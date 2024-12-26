import torch
import random
import importlib
import torch.nn as nn
import torch.nn.functional
from einops import rearrange
from utils.generat_utils import top_k_sampling, top_p_sampling, pk_sampling
from modules.tokenizers.pose_tokenizer import poses_to_indices, yaws_to_indices

class TrainTransformers(nn.Module):
    def __init__(
        self, 
        args, 
        local_rank=-1,
        condition_frames=3,):
        super().__init__()
        self.local_rank = local_rank
        self.args = args
        self.condition_frames = condition_frames
        self.codebook_size = self.args.codebook_size
        self.vae_emb_dim = self.args.codebook_embed_dim
        self.image_size = self.args.image_size
        self.latent_size = (self.image_size[0]//self.args.downsample_size,  self.image_size[1]//self.args.downsample_size)
        self.pkeep = args.pkeep
        self.img_token_size = self.latent_size[0] * self.latent_size[1]
        self.pose_x_vocab_size = self.args.pose_x_vocab_size
        self.pose_y_vocab_size = self.args.pose_y_vocab_size
        self.yaw_vocab_size = self.args.yaw_vocab_size
        self.pose_token_size = 2
        self.yaw_token_size = 1
        self.total_token_size = self.img_token_size + self.pose_token_size + self.yaw_token_size
        self.token_size_dict = {
            'img_tokens_size': self.img_token_size,
            'pose_tokens_size': self.pose_token_size,
            'yaw_token_size': self.yaw_token_size,
            'total_tokens_size': self.total_token_size
        }
        transformer_config = {
            "vocab_size": self.codebook_size,
            "block_size": condition_frames*(self.total_token_size), 
            "n_layer": args.n_layer,
            "n_head": 16,
            "n_embd": args.n_embd,
            "pose_x_vocab_size": self.pose_x_vocab_size,
            "pose_y_vocab_size": self.pose_y_vocab_size,
            "yaw_vocab_size": self.yaw_vocab_size,
        } 
        module_path = 'models.' + args.gpt_type
        module = importlib.import_module(f'{module_path}')
        self.model = module.GPT(
            **transformer_config, 
            latent_size=self.latent_size, 
            L=self.img_token_size, 
            local_rank=local_rank, 
            condition_frames=self.condition_frames, 
            token_size_dict=self.token_size_dict,
            vae_emb_dim = self.vae_emb_dim,
            )
    
    def model_forward(self, feature_total, pose_indices_total, yaw_indices_total, targets, posedrop_input_flag, posedrop_gt_flag):
        logits = self.model(feature_total, pose_indices_total, yaw_indices_total, posedrop_input_flag)
        yaw_logits = logits['yaw_logits']
        pose_x_logits = logits['pose_x_logits']
        pose_y_logits = logits['pose_y_logits']
        img_logits = logits['img_logits']
        yaw_logits = rearrange(yaw_logits, 'b n c -> b c n')
        pose_x_logits = rearrange(pose_x_logits, 'b n c -> b c n')
        pose_y_logits = rearrange(pose_y_logits, 'b n c -> b c n')
        img_logits = rearrange(img_logits, 'b n c -> b c n')
        yaw_loss = nn.functional.cross_entropy(yaw_logits, targets[:, 0:self.yaw_token_size], reduction='none')
        pose_x_loss = nn.functional.cross_entropy(pose_x_logits, targets[:, self.yaw_token_size:self.yaw_token_size+1], reduction='none')
        pose_y_loss = nn.functional.cross_entropy(pose_y_logits, targets[:, self.yaw_token_size+1:self.yaw_token_size+2], reduction='none')
        pose_loss_weight = (~posedrop_gt_flag).float()[:, None]
        yaw_loss_weight = yaw_loss * pose_loss_weight
        pose_x_loss_weight = pose_x_loss * pose_loss_weight
        pose_y_loss_weight = pose_y_loss *  pose_loss_weight
        img_loss = nn.functional.cross_entropy(img_logits, targets[:, self.yaw_token_size+self.pose_token_size:], reduction='none')
        loss = torch.cat([yaw_loss_weight, pose_x_loss_weight, pose_y_loss_weight, img_loss], dim=1)
        loss_mean = loss.mean()
        return loss_mean


    def step_train(self, token, feature, pose, yaw, token_gt, feature_gt, pose_gt, yaw_gt, vqvae_codebook=None):
        self.model.train()
        posedrop_input_flag = torch.isinf(pose[:, 0, 0])
        posedrop_gt_flag = torch.isinf(pose[:, -1, 0])
        pose_total = poses_to_indices(torch.cat([pose, pose_gt], dim=1), self.pose_x_vocab_size, self.pose_y_vocab_size)  
        yaw_total = yaws_to_indices(torch.cat([yaw, yaw_gt], dim=1), self.yaw_vocab_size)  
        pro = random.random()
        if  pro < self.args.mask_data: 
            mask = torch.bernoulli(self.pkeep * torch.ones(token.shape, device=token.device))
            mask = mask.round().to(dtype=torch.int64)
            random_token = torch.randint_like(token, self.codebook_size)
            new_token = mask * token + (1 - mask) * random_token
            mask_gt = torch.bernoulli(self.pkeep * torch.ones(token_gt.shape, device=token_gt.device))
            mask_gt = mask_gt.round().to(dtype=torch.int64)
            random_token_gt = torch.randint_like(token_gt, self.codebook_size)
            new_token_gt = mask_gt * token_gt + (1 - mask_gt) * random_token_gt
        else:
            new_token = token
            new_token_gt = token_gt

        feature_masked_total = torch.cat(
            [vqvae_codebook(new_token), vqvae_codebook(new_token_gt)], 
            dim=1)
        feature_masked_query = feature_masked_total[:, 1:, :, :]
        feature_masked = feature_masked_total[:, :-1, :, :]
        pose_indices = pose_total[:, :-1, :]
        yaw_indices = yaw_total[:, :-1, :]
        pose_gt_indices = pose_total[:, -1:, :]
        yaw_gt_indices = yaw_total[:, -1:, :]
        targets = torch.cat(
            [torch.cat([yaw_indices, pose_indices, token], dim=2), 
            torch.cat([yaw_gt_indices, pose_gt_indices, token_gt], dim=2)], 
            dim=1)[:, 1:, :]
        targets = rearrange(targets, 'b F L -> (b F) L')
        loss = self.model_forward(feature_masked_total, pose_total, yaw_total, targets, posedrop_input_flag, posedrop_gt_flag)
        return loss

    @torch.no_grad()
    def step_eval(self, token, feature, pose, yaw, token_gt, feature_gt, pose_gt, yaw_gt, vqvae_codebook):
        self.model.eval()
        pose_indices = self.poses_to_indices(pose) 
        pose_gt_indices = self.poses_to_indices(pose_gt) 
        yaw_indices = self.yaws_to_indices(yaw)
        yaw_gt_indices = self.yaws_to_indices(yaw_gt)
        intra_token_embeddings = self.model.eval_first_second(feature,yaw_indices, pose_indices) 
        targets = torch.cat(
            [torch.cat([yaw_indices, pose_indices, token], dim=2), 
            torch.cat([yaw_gt_indices, pose_gt_indices, token_gt], dim=2)], 
            dim=1)[:, 1:, :]
        targets = targets[:, -1, ...] 
        loss_final = 0
        logits_yaw = self.model.eval_third(intra_token_embeddings[:, :self.yaw_token_size, :])['yaw_logits']
        logits_yaw = logits_yaw[:, 0:self.yaw_token_size, :]
        yaws = yaw_gt_indices[:, 0, :]
        loss_0 = nn.functional.cross_entropy(logits_yaw.reshape(-1, logits_yaw.size(-1)), targets[:, 0:self.yaw_token_size].reshape(-1))
        loss_final += loss_0
        logits_pose = self.model.eval_third(intra_token_embeddings[:, :self.yaw_token_size+self.pose_token_size, :], yaws=yaws)['pose_logits'] 
        logits_pose = logits_pose[:, self.yaw_token_size:self.yaw_token_size+self.pose_token_size, :]
        poses = pose_gt_indices[:, 0, :] 
        loss_1 = nn.functional.cross_entropy(logits_pose.reshape(-1, logits_pose.size(-1)), targets[:, self.yaw_token_size:self.yaw_token_size+self.pose_token_size].reshape(-1))
        loss_final += loss_1
        new_tokens = None
        predict_indices = []
        for i in range(self.img_token_size):
            logits = self.model.eval_third(intra_token_embeddings[:, :i+self.yaw_token_size+self.pose_token_size+1, :], yaws=yaws, poses=poses, new_tokens=new_tokens)['img_logits']
            logits = logits[:, i:i+1, :]
            loss_tmp = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[:, i+self.yaw_token_size+self.pose_token_size:i+self.yaw_token_size+self.pose_token_size+1].reshape(-1))
            loss_final += loss_tmp
            predicts = logits.argmax(dim=-1)
            predict_indices.append(predicts)
            tmp_tokens = vqvae_codebook(predicts)
            if new_tokens is not None:
                new_tokens = torch.cat([new_tokens, tmp_tokens], dim=1)
            else:
                new_tokens = tmp_tokens
        predict_indices = torch.cat(predict_indices, dim=1)
        predict_indices = rearrange(predict_indices, 'b (h w) -> b h w', h=self.latent_size[0], w=self.latent_size[1])
        indices_gt = rearrange(token_gt, 'b 1 (h w) -> b h w', h=self.latent_size[0], w=self.latent_size[1])
        return predict_indices, indices_gt, loss_final/self.total_token_size

    def forward(self, token, feature, pose, yaw, token_gt, feature_gt, pose_gt, yaw_gt, vqvae_codebook=None, eval=False):
        if self.training:
            loss = self.step_train(token, feature, pose, yaw, token_gt, feature_gt, pose_gt, yaw_gt, vqvae_codebook)
            return loss
        else:
            predict_indices, indices_gt, loss_final = self.step_eval(token, feature, pose, yaw, token_gt, feature_gt, pose_gt, yaw_gt, vqvae_codebook)
            return predict_indices, indices_gt, loss_final
    
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def generate_gt_pose_gt_yaw(self, token, feature, pose, yaw, pose_gt, yaw_gt, vqvae_codebook, sampling_mtd = None, temperature_k = 1.0, top_k = 0, temperature_p = 1.0, top_p = 0):
        self.model.eval()
        posedrop_input_flag = torch.isinf(pose[:, 0, 0])    
        pose_indices = poses_to_indices(pose, self.pose_x_vocab_size, self.pose_y_vocab_size) 
        pose_gt_indices = poses_to_indices(pose_gt, self.pose_x_vocab_size, self.pose_y_vocab_size) 
        yaw_indices = yaws_to_indices(yaw, self.yaw_vocab_size)
        yaw_gt_indices = yaws_to_indices(yaw_gt, self.yaw_vocab_size)
        intra_token_embeddings = self.model.eval_first_second(feature, pose_indices, yaw_indices, posedrop_input_flag) 
        yaws = yaw_gt_indices[:, 0:1, :]
        poses = pose_gt_indices[:, 0:1, :]
        new_tokens = None
        predict_indices = []
        for i in range(self.img_token_size):
            logits = self.model.eval_third(intra_token_embeddings[:, :i+self.yaw_token_size+self.pose_token_size+1, :], yaws=yaws, poses=poses, new_tokens=new_tokens, drop_flag=posedrop_input_flag)['img_logits']
            logits = logits[:, i:i+1, :]
            if sampling_mtd == 'top_k':
                idx_next = top_k_sampling(logits, temperature_k = temperature_k, top_k = top_k)
            elif sampling_mtd == 'top_p':
                idx_next = top_p_sampling(logits, temperature_p = temperature_p, top_p = top_p)
            elif sampling_mtd == 'pk':
                idx_next = pk_sampling(logits, temperature_k = temperature_k, top_k = top_k, temperature_p = temperature_p, top_p = top_p)
            else:
                idx_next = logits.argmax(dim=-1)
            predict_indices.append(idx_next)
            tmp_tokens = vqvae_codebook(idx_next)
            if new_tokens is not None:
                new_tokens = torch.cat([new_tokens, tmp_tokens], dim=1)
            else:
                new_tokens = tmp_tokens
        predict_indices = torch.cat(predict_indices, dim=1)
        predict_indices = rearrange(predict_indices, 'b (h w) -> b h w', h=self.latent_size[0], w=self.latent_size[1])
        return predict_indices
    
    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(self.model.state_dict(),'{}/tvar_{}.pkl'.format(path, str(epoch)))  
        
