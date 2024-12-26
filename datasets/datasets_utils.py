import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation

def reverse_seq_data(poses, seqs):
    seq_len = len(poses)
    reverse_seq = seqs[::-1]
    start_pose = poses.pop(-1)
    reverse_poses = [-pi for pi in poses]
    reverse_poses = [start_pose, ] + reverse_poses[::-1]
    return reverse_poses, reverse_seq

def data_aug_for_seq(imgs, img_h, img_w):
    seq_len = len(imgs)
    H, W, _ = imgs[0].shape
    random_resize_ratio = np.random.uniform(1.01, 1.4)
    if (img_w/img_h)*H < W:  
        resize_h = round(random_resize_ratio * img_h)
        resize_w = round(W / H * resize_h)
    else:
        resize_w = round(random_resize_ratio * img_w)
        resize_h = round(H / W * resize_w)
    crop_h = np.random.randint(0, resize_h-img_h-1)
    crop_w = np.random.randint(0, resize_w-img_w-1)
    for i, img in enumerate(imgs):
        img_i = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        imgs[i] = img_i[crop_h:crop_h+img_h, crop_w:crop_w+img_w, :]
    return imgs

def radians_to_degrees(radians):
    degrees = radians * (180 / math.pi)
    return degrees

def get_meta_data(poses):
    poses = np.concatenate([poses[0:1], poses], axis=0)
    rel_pose = np.linalg.inv(poses[:-1]) @ poses[1:]
    xyzs = rel_pose[:, :3, 3]
    xys = xyzs[:, :2]
    rel_yaws = radians_to_degrees(Rotation.from_matrix(rel_pose[:,:3,:3]).as_euler('zyx', degrees=False)[:,0])[:, np.newaxis]
    return {
        'rel_poses': xys,
        'rel_yaws': rel_yaws,
    }