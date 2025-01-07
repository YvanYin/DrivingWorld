import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from datasets.datasets_utils import get_meta_data

class NuPlanTest(Dataset):
    def __init__(self, data_root, json_root, condition_frames=3, downsample_fps=5, downsample_size=16, h=256, w=512):
        self.meta_path = f'{json_root}/test_meta_v1.json'
        self.pose_meta_path = f'{json_root}/test_ego_meta_v1'
        self.condition_frames = condition_frames
        self.data_root = data_root 
        self.ori_fps = 10 
        self.downsample = self.ori_fps // downsample_fps
        self.h = h
        self.w = w
        with open(self.meta_path, 'r') as f:
            json_data = json.load(f)
        json_data_filter = []
        for data in json_data:
            if len(data['CAM_F0']) > self.condition_frames * self.downsample:
                json_data_filter.append(data)
        self.sequences = json_data_filter
        self.downsample_size = downsample_size
        print("self.downsample_size, self.condition_frames, self.downsample_fps", self.downsample_size, self.condition_frames, downsample_fps)

    def __len__(self):
        return len(self.sequences)    

    def load_pose(self, pose_path, front_cam_list):
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        front_cam_pose = pose_data['CAM_F0'] if 'CAM_F0' in pose_data else pose_data
        poses = {key:
        [pose_meta['x'],pose_meta['y'],pose_meta['z'],
        pose_meta['qx'],pose_meta['qy'],pose_meta['qz'], pose_meta['qw'],] for key, pose_meta in front_cam_pose.items()}
        poses_filter = np.array([poses[f"CAM_F0/{ts}"] for ts in front_cam_list])
        return poses_filter

    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs
        
    def __loadarray_tum_single(self, array):
        absolute_transforms = np.zeros((4, 4))
        absolute_transforms[3, 3] = 1
        absolute_transforms[:3, :3] = R.from_quat(array[3:7]).as_matrix()
        absolute_transforms[:3, 3] = array[0:3]
        return absolute_transforms

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
    
    def downsample_sequences(self, img_ts, poses):
        ori_size = len(img_ts)
        assert len(img_ts) == len(poses)
        index_list = np.arange(0, ori_size, step=self.downsample)
        img_ts_downsample =np.array(img_ts)[index_list]
        poses_downsample = poses[index_list]
        return img_ts_downsample, poses_downsample

    def getimg(self, index):
        seq_data = self.sequences[index]
        seq_root = os.path.join(self.data_root, seq_data['data_root'])
        seq_db_name = os.path.basename(seq_root)
        pose_path = f"{self.pose_meta_path }/{seq_db_name}.json"
        rgb_front_dir = f"{seq_root}/CAM_F0"
        poses = self.load_pose(pose_path, seq_data['CAM_F0'])
        img_ts_downsample, poses_downsample = self.downsample_sequences(seq_data['CAM_F0'], poses)
        clip_length = len(img_ts_downsample)
        ims = []
        poses_new = []
        for i in range(clip_length):   
            im = cv2.cvtColor(cv2.imread(f"{rgb_front_dir}/{img_ts_downsample[i]}"), cv2.COLOR_BGR2RGB)
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
        poses_new.append(self.__loadarray_tum_single(poses_downsample[i]))
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
