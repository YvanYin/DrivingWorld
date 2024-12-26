import torch

def poses_to_indices(poses, x_divisions=128, y_divisions=128):
    """
    Tokenize pose (x, y) to indices.
    Args:
        poses: (x, y), [b, f, 2]
    Return:
        pose dicces: [b, f, 2]
    """
    x_min = 0
    x_range = 8
    y_range = 1
    y_step = y_range / y_divisions
    y_min = -0.5 + y_step / 2
    x, y = poses[:, :, 0], poses[:, :, 1]        
    idx_x = torch.floor(x * x_divisions / x_range).clip(0, x_divisions-1).to(torch.long).unsqueeze(dim=2)
    idx_y = torch.floor((y-y_min) * y_divisions / y_range).clip(0, y_divisions-1).to(torch.long).unsqueeze(dim=2)
    return torch.cat([idx_x, idx_y], dim=2)

def indices_to_pose(idx_x, idx_y, x_division=128, y_divisions=128):
    x_min = 0
    x_range = 8
    y_range = 1
    y_step = y_range / y_divisions
    y_min = -0.5 + y_step / 2
    x_step = x_range / x_division

    x = idx_x * x_step + x_step / 2
    y = idx_y * y_step + y_min + y_step / 2
    return x, y

def yaws_to_indices(yaws, division=512):
    yaw_range = 16
    yaw_step = yaw_range / division
    yaw_min = - yaw_range / 2.0 + yaw_step / 2.0
    idx_yaw = torch.floor((yaws - yaw_min) * division / yaw_range).clip(0, division-1).to(torch.long)
    return idx_yaw

def indices_to_yaws(idx_yaw, division=512):
    yaw_range = 16
    yaw_step = yaw_range / division
    yaw_min = - yaw_range / 2.0 + yaw_step / 2.0
    yaw = idx_yaw * yaw_step + yaw_min + yaw_step / 2.0
    return yaw