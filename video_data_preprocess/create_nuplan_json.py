import os
import json
import tqdm
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper

ROOT = "/root" 
NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', ROOT)
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', f'{ROOT}/maps')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')

valid_trainvaltest_dbs = os.listdir(f'{ROOT}/nuplan-v1.1/sensor_blobs')
valid_trainvaltest_dbs.sort()

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--split_id', default=0, type=int, help='minibatch size')
    parser.add_argument('--split_num', default=2, type=int, help='minibatch size')
    args = parser.parse_args()
    return args

def load_single_log_db_data(log_db_name="2021.05.12.22.28.35_veh-35_00620_01164", log_db=None):
    print('In load_single_log_db')
    lidar_pcs = log_db.lidar_pc
    images_data = log_db.image
    next_img_token = None
    img_root_path = f'{ROOT}/nuplan-v1.1/sensor_blobs/{log_db_name}'
    cameras={
        'CAM_L2':[],
        'CAM_F0':[],
        'CAM_R2':[],
        'CAM_L0':[],
        'CAM_L1':[],
        'CAM_R0':[],
        'CAM_R1':[],
        'CAM_B0':[],
        'data_root': img_root_path,
        }
    seq_tmp = []

    ego_pose = {
        'CAM_L2':{},
        'CAM_F0':{},
        'CAM_R2':{},
        'CAM_L0':{},
        'CAM_L1':{},
        'CAM_R0':{},
        'CAM_R1':{},
        'CAM_B0':{},
    }

    ego_pose_from_cam = {
        'CAM_L2':{},
        'CAM_F0':{},
        'CAM_R2':{},
        'CAM_L0':{},
        'CAM_L1':{},
        'CAM_R0':{},
        'CAM_R1':{},
        'CAM_B0':{},
    }

    pre_scene_token = None
    for idx, img_item in enumerate(images_data):
        camera_channel = img_item.camera.channel
        camera_intrinsic = img_item.camera.intrinsic
        img_name = os.path.basename(img_item.filename_jpg)
        loaded_db_name = img_item.filename_jpg.split('/')[0]
        assert loaded_db_name == log_db_name
        next_img_token = img_item.next_token
        curr_img_token = img_item.token
        scene_token = img_item.lidar_pc.scene_token
        img_path = f'{log_db._data_root}/nuplan-v1.1/sensor_blobs/{img_item.filename_jpg}'
        if not os.path.exists(img_path):
            print(f'!!!!WARNING: {img_path} do not exist.')
            continue
        curr_ego_pose = img_item.lidar_pc.ego_pose
        curr_ego_pose_from_cam = img_item.ego_pose
        ego_pose[camera_channel][f'{camera_channel}/{img_name}'] = {
            'x':  curr_ego_pose.x,
            'y':  curr_ego_pose.y,
            'z':  curr_ego_pose.z,
            'qw': curr_ego_pose.qw,
            'qx': curr_ego_pose.qx,
            'qy': curr_ego_pose.qy,
            'qz': curr_ego_pose.qz,
            'vx': curr_ego_pose.vx,
            'vy': curr_ego_pose.vy,
            'ax': curr_ego_pose.acceleration_x,
            'ay': curr_ego_pose.acceleration_y,
            'timestamp': curr_ego_pose.timestamp,
        }
        ego_pose_from_cam[camera_channel][f'{camera_channel}/{img_name}'] = {
            'x':  curr_ego_pose_from_cam.x,
            'y':  curr_ego_pose_from_cam.y,
            'z':  curr_ego_pose_from_cam.z,
            'qw': curr_ego_pose_from_cam.qw,
            'qx': curr_ego_pose_from_cam.qx,
            'qy': curr_ego_pose_from_cam.qy,
            'qz': curr_ego_pose_from_cam.qz,
            'vx': curr_ego_pose_from_cam.vx,
            'vy': curr_ego_pose_from_cam.vy,
            'ax': curr_ego_pose_from_cam.acceleration_x,
            'ay': curr_ego_pose_from_cam.acceleration_y,
            'timestamp': curr_ego_pose_from_cam.timestamp,
        }

        if (scene_token != pre_scene_token) and (pre_scene_token is not None):
            print(f'idx: {idx}, scene_token:{scene_token}, next_token:{next_img_token}, {camera_channel}, img_path:{img_item.filename_jpg}')
            cameras[camera_channel].append({'seq': seq_tmp, 'scene': pre_scene_token})
            seq_tmp = [img_name, ]
        else:
            seq_tmp.append(img_name)
        pre_scene_token = scene_token
    return cameras, ego_pose

def loop_over_db_files(
    rank=0,
    workers=1,
    split_db_list=None,
    ego_save_dir='',
    seq_save_dir='', nuplandb_wrapper=None):
    total_sequence = []
    sensor_data_db_list = os.listdir(f'{ROOT}/nuplan-v1.1/sensor_blobs')
    sensor_data_db_list.sort()
    print('In Loop over db.')
    for idx, db_i in tqdm.tqdm(enumerate(split_db_list)):
        print('id', idx)
        if db_i not in sensor_data_db_list:
            print(f"!!!! {db_i} donot exist in sensor data.")
            continue
        print(f">>>>Start {db_i}.")
        log_db = nuplandb_wrapper.get_log_db(db_i)
        print(log_db)
        cameras, ego_pose = load_single_log_db_data(db_i, log_db)
        scene_num = len(cameras['CAM_F0'])
        for i in range(scene_num):
            new_seq_meta = {
                    'CAM_F0': cameras['CAM_F0'][i]['seq'],
                    'scene': cameras['CAM_F0'][i]['scene'],
                    'data_root': cameras['data_root'],
                    'pose': f'ego_meta/{db_i}.json',
                }
            total_sequence.append(new_seq_meta)
        ego_meta_path = os.path.join(ego_save_dir, db_i+'.json')
        os.makedirs(os.path.dirname(ego_meta_path), exist_ok=True)
        with open(ego_meta_path, 'w') as f:
            json.dump(ego_pose, f)
        seq_meta_path = os.path.join(seq_save_dir, db_i+'.json')
        os.makedirs(os.path.dirname(seq_meta_path), exist_ok=True)
        with open(seq_meta_path, 'w') as f:
            json.dump(total_sequence, f)
    return total_sequence


def accumulate_results(all_results):
    accumulated_results = []
    for result in all_results:
        accumulated_results.extend(result)
    return accumulated_results

if __name__ == '__main__':
    args = add_arguments()
    split_num = args.split_num
    split_id = args.split_id
    num_processes = args.workers
    print('##############', split_id, split_num)
    split_name = 'trainval'
    ego_save_dir = '' # your path here
    seq_save_dir = '' # your path here
    db_list = valid_trainvaltest_dbs
    db_list.sort()
    split_db_list = db_list[split_id::split_num]
    if split_name == 'trainval':
        split_db_list_update = [db_i for db_i in  split_db_list if os.path.exists(f'{ROOT}/nuplan-v1.1/splits/test/{db_i}.db')]
        db_path_lists = [f'{ROOT}/nuplan-v1.1/splits/trainval/{db_i}.db' for db_i in  split_db_list_update]
    elif split_name == 'test':
        split_db_list_update = [db_i for db_i in  split_db_list if os.path.exists(f'{ROOT}/nuplan-v1.1/splits/test/{db_i}.db')]
        db_path_lists = [f'{ROOT}/nuplan-v1.1/splits/test/{db_i}.db' for db_i in  split_db_list_update]

    nuplandb_wrapper = NuPlanDBWrapper(
        data_root=NUPLAN_DATA_ROOT,
        map_root=NUPLAN_MAPS_ROOT,
        db_files=db_path_lists,
        map_version=NUPLAN_MAP_VERSION,
    )
    print('Sart loop.')
    accumulated_results = loop_over_db_files(0, 1, split_db_list_update, ego_save_dir=ego_save_dir, seq_save_dir=seq_save_dir, nuplandb_wrapper=nuplandb_wrapper)