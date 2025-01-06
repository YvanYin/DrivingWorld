from torch.utils.data import ConcatDataset

from datasets.dataset_demo import DemoTest

def create_test_datasets(args, split='test'):
    data_list = args.test_data_list
    dataset_list = []
    for data_name in data_list:
        if data_name == 'demo':
            dataset = DemoTest(
                    args.datasets_paths['demo_root'], 
                    condition_frames=args.condition_frames, downsample_fps=args.downsample_fps)
        dataset_list.append(dataset)
    data_array = ConcatDataset(dataset_list)
    return data_array