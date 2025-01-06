import os
import cv2
import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class DemoTest(Dataset):
    def __init__(self, data_root, condition_frames=15, downsample_size=16, h=256, w=512):
        self.data_root = data_root
        self.video_path_list = sorted(os.listdir(self.data_root))
        self.condition_frames = condition_frames
        self.h = h
        self.w = w
        self.downsample_size = downsample_size
        print("self.downsample_size, self.condition_frames", self.downsample_size, self.condition_frames)
    
    def __len__(self):
        return len(self.video_path_list)    
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs   

    def aug_seq(self, imgs):
        ih, iw, _ = imgs[0].shape
        assert self.h == 256, self.w == 512
        if iw == 512:
            x = int(ih/2-self.h/2)
            y = 0
        else:
            x = 0
            y = int(iw/2-self.w/2)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+self.h, y:y+self.w, :]
        return imgs   
    
    def getimg(self, index):
        seq_data = self.video_path_list[index]
        video_path = os.path.join(seq_data, video_path)
        frames = glob.glob(os.path.join(video_path, '*.png'))
        poses = np.load(os.path.join(video_path, 'pose_x_y.npy'))
        yaws = np.load(os.path.join(video_path, 'yaw.npy'))
        clip_length = len(frames)
        ims = []
        for i in range(clip_length):   
            im = cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB)
            h, w, _ = im.shape
            if 2*h < w:
                w_1 = round(w / h * self.h)
                im = Image.fromarray(im).resize((w_1, self.h), resample= Image.BICUBIC)
                im = np.array(im)
            else:
                h_1 = round(h / w * self.w)
                im = Image.fromarray(im).resize((self.w, h_1), resample= Image.BICUBIC)
                im = np.array(im)
            ims.append(im)
        return ims, poses, yaws
            
    def __getitem__(self, index):
        imgs, poses, yaws = self.getimg(index)
        imgs = self.aug_seq(imgs)
        imgs_tensor = []
        poses_tensor = []
        yaws_tensor = []
        for img, pose, yaw in zip(imgs, poses, yaws):
            imgs_tensor.append(torch.from_numpy(img.copy()).permute(2, 0, 1))
            poses_tensor.append(torch.from_numpy(pose.copy()))
            yaws_tensor.append(torch.from_numpy(yaw.copy()))
        imgs = self.normalize_imgs(torch.stack(imgs_tensor, 0))
        return imgs, torch.stack(poses_tensor, 0).float(), torch.stack(yaws_tensor, 0).float()
    