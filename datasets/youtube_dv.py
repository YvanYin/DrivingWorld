import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class YoutubeDVTest(Dataset):
    def __init__(self, data_root, meta_path, split='test', downsample_fps=5, height=256, width=512, condition_frames=15, all_frames=150):
        self.data_root = data_root
        self.meta_path = meta_path 
        self.downsampled_fps = downsample_fps  
        self.downsample = 10 // self.downsampled_fps
        self.h = height
        self.w = width
        with open(self.meta_path, 'r') as f:
            annos = json.load(f)
        self.img_path_data = annos[500:]    
        self.condition_frames = condition_frames
        self.all_frames = all_frames
    
    def create_sequences(self, annos, split):
        sequence_list = []      
        for idx, an_item in enumerate(annos):
            video_name = an_item['video_name']
            video_path = os.path.join(self.data_root, f'{split}_images', video_name)
            start = an_item['start']
            end = an_item['end']
            index_list = np.arange(start, end, self.downsample_frame)
            if index_list.size < 100:
                continue
            sequence_list_video_i = [[os.path.join(video_name, '%09d.jpg' % i) for i in index_list[index_list_i:index_list_i+25]] for index_list_i, sq_start in enumerate(index_list[:-25])]
            sequence_list.extend(sequence_list_video_i)
        return sequence_list
    
    def __len__(self):
        return len(self.img_path_data)    
    
    def downsample_sequences(self, img_ts, poses):
        ori_size = len(img_ts)
        assert len(img_ts) == len(poses)
        index_list = np.arange(0, ori_size, step=self.downsample)
        img_ts_downsample =np.array(img_ts)[index_list]
        poses_downsample = np.array(poses)[index_list]
        return img_ts_downsample, poses_downsample

    def aug_seq(self, imgs, h, w):
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
    
    def pre_process_image(self, img):
        up = 50
        down = -50
        left = 100
        right = -100
        out = img[up:down, left:right, :]
        return out

    def getimg(self, index):
        img_paths = os.path.join(self.data_root, self.img_path_data[index]['path'])
        img_names = self.img_path_data[index]['seq']
        clip_length = len(img_names)
        poses = [np.array([np.inf, np.inf])] * clip_length  
                
        img_ts_downsample, _ = self.downsample_sequences(img_names, poses)
        
        ims = []
        ps = []
        ys = []
        
        clip_len = self.all_frames
        for i in range(clip_len):
            if i < len(img_ts_downsample):
                im = cv2.cvtColor(cv2.imread(f"{img_paths}/{img_ts_downsample[i]}"), cv2.COLOR_BGR2RGB)
                im = self.pre_process_image(im)
                h, w, _ = im.shape
                if 2*h < w:
                    w_1 = round(w / h * self.h)

                    im = Image.fromarray(im).resize((w_1, self.h), resample= Image.BICUBIC)
                    im = np.array(im)
                else:
                    h_1 = round(h / w * self.w)

                    im = Image.fromarray(im).resize((self.w, h_1), resample= Image.BICUBIC)
                    im = np.array(im)
            else:
                im = ims[-1]
            ims.append(im)
        ys = [np.array([np.inf, ])] * clip_len
        ps = [np.array([np.inf, np.inf])] * clip_len
        return ims, ps, ys
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs

    def __getitem__(self, index):
        imgs, poses, yaws = self.getimg(index)
        imgs = self.aug_seq(imgs, self.h, self.w)
        imgs_tensor = []
        poses_tensor = []
        yaws_tensor = []
        for img, pose,yaw in zip(imgs, poses, yaws):
            imgs_tensor.append(torch.from_numpy(img.copy()).permute(2, 0, 1))
            poses_tensor.append(torch.from_numpy(pose.copy()))
            yaws_tensor.append(torch.from_numpy(yaw.copy()))
        poses = torch.stack(poses_tensor, 0).float()
        yaws = torch.stack(yaws_tensor, 0).float()
        imgs = self.normalize_imgs(torch.stack(imgs_tensor, 0))
        return imgs, poses, yaws