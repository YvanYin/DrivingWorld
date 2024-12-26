from datasets.youtube_dv import YoutubeDVTest
from datasets.dataset import ValImgDataset
from torch.utils.data import ConcatDataset
from datasets.nuplan import NuPlanVal, NuPlanTest

def create_val_datasets(args, split='val'):
    data_list = args.val_data_list
    dataset_list = []
    for data_name in data_list:
        if data_name == 'nuscense_img':
            dataset = ValImgDataset(
                    args.datasets_paths['nuscense_root'], 
                    args.datasets_paths['nuscense_val_json_path'], 
                    condition_frames=args.condition_frames, downsample_fps=args.downsample_fps)
        elif data_name == 'youtube':
            dataset = YoutubeDVTest(
                args.datasets_paths['youtube_root'], 
                args.datasets_paths['youtube_val_json_root'],
                split=split, 
                condition_frames=args.condition_frames, 
                downsample_fps=args.downsample_fps)
        elif data_name == 'nuplan':
            dataset = NuPlanVal(
                args.datasets_paths['nuplan_root'], 
                args.datasets_paths['nuplan_json_root'], 
                condition_frames=args.condition_frames, downsample_fps=args.downsample_fps)
        dataset_list.append(dataset)
    data_array = ConcatDataset(dataset_list)
    return data_array

def create_test_datasets(args, split='test'):
    data_list = args.val_data_list
    dataset_list = []
    for data_name in data_list:
        if data_name == 'nuscense_img':
            dataset = ValImgDataset(
                    args.datasets_paths['nuscense_root'], 
                    args.datasets_paths['nuscense_val_json_path'], 
                    condition_frames=args.condition_frames, downsample_fps=args.downsample_fps)
        elif data_name == 'youtube':
            dataset = YoutubeDVTest(
                args.datasets_paths['youtube_root'], 
                args.datasets_paths['youtube_val_json_root'],
                split=split, 
                condition_frames=args.condition_frames) 
        elif data_name == 'nuplan':
            dataset = NuPlanTest(
                args.datasets_paths['nuplan_root'], 
                args.datasets_paths['nuplan_json_root'], 
                condition_frames=args.condition_frames, downsample_fps=args.downsample_fps)
        dataset_list.append(dataset)
    data_array = ConcatDataset(dataset_list)
    return data_array