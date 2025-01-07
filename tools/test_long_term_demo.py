import os
import cv2
import sys
import torch
import random
import argparse
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.utils import *
from utils.testing_utils import create_mp4_imgs, set_text
from models.model import TrainTransformers
from modules.tokenizers.model_tokenizer import Tokenizer
from utils.config_utils import Config
from utils.running import load_parameters
from datasets.create_dataset import create_test_datasets

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_path', default=None, type=str, help='continue to train, model path')
    parser.add_argument('--resume_iter', default=0, type=int, help='continue to train, iter')
    parser.add_argument('--load_path', default=None, type=str, help='pretrained path')
    parser.add_argument("--config", type=str, required=True, help="configs for training")
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--save_video_path', type=str, required=True)
    
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg

def init_environment(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

def test_sliding_window_img(val_data, model, args, tokenizer):
    condition_frames = args.condition_frames
    model.eval()
    if not os.path.exists(os.path.join(args.save_video_path, args.exp_name)):
        os.makedirs(os.path.join(args.save_video_path, args.exp_name))
    
    with torch.no_grad():
        for i, (img, pose, yaw) in enumerate(val_data):
            video_save_path = os.path.join(args.save_video_path, args.exp_name, 'sliding_'+str(i))
            os.makedirs(video_save_path, exist_ok=True)
            model.eval().cuda()
            img = img.cuda()
            pose = pose.cuda()
            yaw = yaw.cuda()  
            cond_imgs_wpad0 = torch.cat([img[:, :condition_frames, :], torch.zeros_like(img[:, :1, :])], dim=1)
            start_token_wpad0, start_feature_wpad0 = tokenizer.encode_to_z(cond_imgs_wpad0)
            start_token = start_token_wpad0[:, :condition_frames, ...]
            start_feature = tokenizer.vq_model.quantize.embedding(start_token)  # No l2 norm
            save_imgs = []
            for j in range(condition_frames):
                cond_img_j = ((img[0, j, ...].permute(1, 2, 0) + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                save_imgs.append(cond_img_j[:, :, ::-1])
                cv2.imwrite(os.path.join(video_save_path, '%d.png'%(j)), cond_img_j[:, :, ::-1])        
            total_frames = 150
            pose_last_step = pose[:, -1:, :]
            yaw_last_step = yaw[:, -1:, :]
            pose_expand = torch.cat((pose, pose_last_step.repeat(1, total_frames+1, 1)), dim=1)
            yaw_expand = torch.cat((yaw, yaw_last_step.repeat(1, total_frames+1, 1)), dim=1) 
            pose_first = np.array([float(0), 0])
            yaw_first = np.array([float(0)])
            pose_first = torch.from_numpy(pose_first).cuda().unsqueeze(dim=0).unsqueeze(dim=1)
            yaw_first = torch.from_numpy(yaw_first).cuda().unsqueeze(dim=0).unsqueeze(dim=1)
            for t1 in range(total_frames):
                print(i, t1)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pose_now = pose_expand[:, t1:t1+args.condition_frames, ...]
                    pose_now[:, 0:1, ...] = pose_first
                    yaw_now = yaw_expand[:, t1:t1+args.condition_frames, ...]
                    yaw_now[:, 0:1, ...] = yaw_first
                    predict_indices = model.generate_gt_pose_gt_yaw(
                        start_token,
                        start_feature, 
                        pose_now,
                        yaw_now, 
                        pose_expand[:, t1+condition_frames:t1+condition_frames+1, ...],
                        yaw_expand[:, t1+condition_frames:t1+condition_frames+1, ...], 
                        tokenizer.vq_model.quantize.embedding,
                        sampling_mtd = args.sampling_mtd,
                        temperature_k = args.temperature_k,
                        top_k = args.top_k,
                        temperature_p = args.temperature_p,
                        top_p = args.top_p)
                
                predict_indices = rearrange(predict_indices, '(b F) h w -> b F h w', F=1)[:, -1:, ...]
                condition_tokens = rearrange(start_token, 'b F (h w) -> b F h w', h=predict_indices.shape[-2])
                sliding_window_tokens = rearrange(torch.cat([condition_tokens, predict_indices], dim=1), 'b F h w -> (b F) h w', F=condition_frames+1)
                imgs_pred = tokenizer.z_to_clips(sliding_window_tokens).cpu()         
                imgs_pred_1 = (imgs_pred[0].permute(1, 2, 0).numpy() * 255).astype('uint8')[:,:,::-1]
                format_fn = np.vectorize(lambda x: "{:.2f}".format(x))
                imgs_pred_1 = set_text(imgs_pred_1, 
                                str(format_fn(pose_expand[0, t1+condition_frames, 0].cpu().numpy()))+", "+str(format_fn(pose_expand[0, t1+condition_frames, 1].cpu().numpy()))+", "+str(yaw_expand[0, t1+args.condition_frames, 0].cpu().numpy()))
                cv2.imwrite(os.path.join(video_save_path, '%d.png'%(t1 + condition_frames)), imgs_pred_1)
                save_imgs.append(imgs_pred_1)
                predict_indices = rearrange(predict_indices, 'b 1 h w -> b 1 (h w)')
                start_token = torch.cat((start_token[:,1:condition_frames,...], predict_indices), dim=1)
                start_feature = tokenizer.vq_model.quantize.embedding(start_token)    
            create_mp4_imgs(args, save_imgs, video_save_path, fps=args.downsample_fps)
                
def main(args):
    local_rank = 0
    model = TrainTransformers(args, local_rank=local_rank, condition_frames=args.condition_frames)
    checkpoint = torch.load(args.load_path, map_location="cpu")
    model.model = load_parameters(model.model, checkpoint)
    test_data = create_test_datasets(args)
    val_data = DataLoader(test_data, batch_size=1)
    tokenizer = Tokenizer(args, local_rank)
    test_sliding_window_img(val_data, model, args, tokenizer)

if __name__ == "__main__":   
    args = add_arguments()
    init_environment(args) 
    main(args)
    

       