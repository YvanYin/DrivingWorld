import argparse
import torch
import torch.nn as nn
import torch.nn.functional
import argparse
from einops import rearrange
from modules.tokenizers.vq_model import VQ_models

class Tokenizer(nn.Module):
    def __init__(self, args, local_rank=-1):
        super().__init__()
        self.args = args
        self.vq_model = VQ_models[args.vq_model](codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim, video_vq_temp_frames=args.video_vq_temp_frames)
        self.vq_model.to(local_rank)
        self.vq_model.eval()
        print("load from " + args.vq_ckpt)
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        self.vq_model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint

    @torch.no_grad()
    def encode_to_z(self, x):
        b, t, _, _, _ = x.shape
        h, w = self.args.image_size[0] // self.args.downsample_size, self.args.image_size[1] // self.args.downsample_size 
        ts = rearrange(x, 'b t c h w -> (b t) c h w')
        with torch.no_grad():
            _, _, [_, _, indices, features] = self.vq_model.encode_hacked(ts)
        indices = rearrange(indices, '(b h w) -> b (h w)', b=b*t, h=h, w=w)
        indices = rearrange(indices, '(b t) L -> b t L', b=b, t=t)

        features = rearrange(features, '(b h w) c -> b (h w) c', b=b*t, h=h, w=w)
        features = rearrange(features, '(b t) L c -> b t L c', b=b, t=t)
        return indices, features

    @torch.no_grad()
    def z_to_image(self, indices):
        b, h, w = indices.shape 
        qzshape = [b, self.args.codebook_embed_dim, h, w]
        image = self.vq_model.decode_code(indices, qzshape)
        image = image / 2 + 0.5
        return image.clip(0, 1)
    
    @torch.no_grad()
    def z_to_clips(self, indices):
        b, h, w = indices.shape 
        qzshape = [b, self.args.codebook_embed_dim, h, w]
        image = self.vq_model.decode_code_temporal(indices, qzshape)
        image = image / 2 + 0.5
        return image.clip(0, 1)
    
    @torch.no_grad()
    def z_to_clips_all(self, indices):
        b, h, w = indices.shape 
        qzshape = [b, self.args.codebook_embed_dim, h, w]
        image = self.vq_model.decode_code_temporal_all(indices, qzshape)
        image = image / 2 + 0.5
        return image.clip(0, 1)


class PoseTokenizer(nn.Module):
    def __init__(self, args, pose_codebook_size=4096):
        super().__init__()
        self.latitute_bins = 32
        self.longtitute_bins = pose_codebook_size // self.latitute_bins 
        self.longti_range = [0, 8] if self.longtitute_bins == 128 else [-8, 8]
        self.lati_range = [-1, 1]
        self.longtitu_seps = torch.arange(self.longti_range[0], self.longti_range[1], 
            (self.longti_range[1] - self.longti_range[0]) / self.longtitute_bins).cuda()
        self.latitute_seps = torch.arange(self.lati_range[0], self.lati_range[1], 
            (self.lati_range[1] - self.lati_range[0]) / self.latitute_bins).cuda()

    def tokenize_pose(self, poses):
        x, y = poses[:, :, 0], poses[:, :, 1]
        x_idx = (torch.searchsorted(self.longtitu_seps, x, side='right') - 1).clip(0, self.longtitute_bins - 1)
        y_idx = (torch.searchsorted(self.latitute_seps, y, side='right') - 1).clip(0, self.latitute_bins - 1)
        indices = x_idx * self.latitute_bins + y_idx
        return indices.to(torch.long).unsqueeze(dim=2)
    
    def poses_to_indices(self, poses):
        x, y = poses[:, :, 0], poses[:, :, 1]
        internal_x, internal_y = torch.floor(x * 16).clip(0, 127), torch.floor((y + 1) * 16).clip(0, 31)
        indices = internal_x * self.latitute_bins + internal_y
        return indices.to(torch.long).unsqueeze(dim=2)
