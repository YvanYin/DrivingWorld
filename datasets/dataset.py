import os
import cv2
import json
import torch
from torch.utils.data import Dataset
from datasets.datasets_utils import reverse_seq_data, get_meta_data

class ValImgDataset(Dataset):
    def __init__(self, nuscenes_path, nuscenes_json_path, condition_frames=3, downsample_fps=3, downsample_size=16, h=256, w=512):
        self.img_path_data = []
        self.pose_data = []
        self.nuscenes_path = nuscenes_path
        with open(nuscenes_json_path, 'r', encoding='utf-8') as file:
            nuscenes_preprocess_data = json.load(file)
        self.ori_fps = 12 
        self.downsample = self.ori_fps // downsample_fps
        nuscenes_keys = sorted(list(nuscenes_preprocess_data.keys()))
        for video_keys in nuscenes_keys:
            tmp_img_path = []
            tmp_pose = []
            img_path_poses = nuscenes_preprocess_data[video_keys]
            for img_path_pose in img_path_poses:
                tmp_img_path.append(os.path.join(nuscenes_path, img_path_pose['data_path']))
                tmp_pose.append(img_path_pose['ego_pose'])
            self.img_path_data.append(tmp_img_path)
            self.pose_data.append(tmp_pose)
        self.h = h
        self.w = w
        self.downsample_size = downsample_size
        self.condition_frames = condition_frames
        print("self.downsample_size, self.condition_frames, self.downsample_fps", self.downsample_size, self.condition_frames, downsample_fps)

    def __len__(self):
        return len(self.img_path_data)    
    
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
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs

    def getimg(self, index):
        img_paths = self.img_path_data[index]
        poses = self.pose_data[index]
        start_index = 0
        ims = []
        poses_new = []
        for i in range(self.condition_frames+1):
            im = cv2.cvtColor(cv2.imread(img_paths[start_index+i*self.downsample]), cv2.COLOR_BGR2RGB)
            h, w, _ = im.shape
            if 2*h < w:
                w_1 = round(w / h * self.h)
                im = cv2.resize(im, (w_1, self.h))
            else:
                h_1 = round(h / w * self.w)
                im = cv2.resize(im, (self.w, h_1))
            ims.append(im)
            poses_new.append(poses[start_index+i*self.downsample])
        pose_dict = get_meta_data(poses=poses_new)
        return ims, pose_dict['rel_poses'], pose_dict['rel_yaws']
            
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


class TestImgDataset(Dataset):
    def __init__(self, nuscenes_path, nuscenes_json_path, condition_frames=3, downsample_fps=3, downsample_size=16, h=256, w=512):
        self.img_path_data = []
        self.pose_data = []
        self.nuscenes_path = nuscenes_path
        with open(nuscenes_json_path, 'r', encoding='utf-8') as file:
            nuscenes_preprocess_data = json.load(file)
        self.ori_fps = 12 
        self.downsample = self.ori_fps // downsample_fps
        nuscenes_keys = sorted(list(nuscenes_preprocess_data.keys()))
        for video_keys in nuscenes_keys:
            tmp_img_path = []
            tmp_pose = []
            img_path_poses = nuscenes_preprocess_data[video_keys]
            for img_path_pose in img_path_poses:
                tmp_img_path.append(os.path.join(nuscenes_path, img_path_pose['data_path']))
                tmp_pose.append(img_path_pose['ego_pose'])
            self.img_path_data.append(tmp_img_path)
            self.pose_data.append(tmp_pose)
        self.h = h
        self.w = w
        self.downsample_size = downsample_size
        self.condition_frames = condition_frames
        print("self.downsample_size, self.condition_frames, self.downsample_fps", self.downsample_size, self.condition_frames, downsample_fps)

    def __len__(self):
        return len(self.img_path_data)    
    
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
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs

    def getimg(self, index):
        img_paths = self.img_path_data[index]
        poses = self.pose_data[index]
        clip_length = len(img_paths)//self.downsample
        ims = []
        poses_new = []
        for i in range(clip_length):
            im = cv2.cvtColor(cv2.imread(img_paths[i*self.downsample]), cv2.COLOR_BGR2RGB)
            h, w, _ = im.shape
            if 2*h < w:
                w_1 = round(w / h * self.h)
                im = cv2.resize(im, (w_1, self.h))
            else:
                h_1 = round(h / w * self.w)
                im = cv2.resize(im, (self.w, h_1))
            ims.append(im)
            poses_new.append(poses[i*self.downsample])
        pose_dict = get_meta_data(poses=poses_new)
        return ims, pose_dict['rel_poses'], pose_dict['rel_yaws']
            
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