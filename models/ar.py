import torch
import torch.nn as nn
from einops import rearrange

from utils.rope_2d import *
from utils.embeddings import get_fourier_embeds_from_coordinates

class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class CausalSpaceSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_embd % 32 == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.qk_norm = True
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()   
        self.pose_tokens_num = config.token_size_dict['pose_tokens_size']
        self.img_tokens_num = config.token_size_dict['img_tokens_size']
        self.yaw_token_size = config.token_size_dict['yaw_token_size']
        self.total_tokens_num = config.token_size_dict['total_tokens_size']
        self.patch_size = config.patch_size 
        self.num_tokens = self.total_tokens_num 
        self.freqs_cis_singlescale = compute_axial_cis(dim = config.n_embd  // self.n_head, end_x = self.patch_size[0], end_y = self.patch_size[1], theta = 1000.0)
        
    def forward(self, x, attn_mask):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        if T > self.pose_tokens_num+self.yaw_token_size: 
            q_B_scale2_d = q[:, :, self.pose_tokens_num+self.yaw_token_size:, :]
            k_B_scale2_d = k[:, :, self.pose_tokens_num+self.yaw_token_size:, :]
            q_out, k_out = apply_rotary_emb(q_B_scale2_d, k_B_scale2_d, freqs_cis=self.freqs_cis_singlescale[:T-self.pose_tokens_num-self.yaw_token_size]) 
            q = torch.cat([q[:, :, 0:self.pose_tokens_num+self.yaw_token_size, :], q_out], dim=2)
            k = torch.cat([k[:, :, 0:self.pose_tokens_num+self.yaw_token_size, :], k_out], dim=2)
        if attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None, :, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask = attn_mask.to(q.dtype), dropout_p=self.attn_dropout_rate).transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_drop(self.proj(y))
        return y

class CausalSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSpaceSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),  
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x

class SpaceSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.qk_norm = True
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()
        self.pose_tokens_num = config.token_size_dict['pose_tokens_size']
        self.img_tokens_num = config.token_size_dict['img_tokens_size']
        self.yaw_token_size = config.token_size_dict['yaw_token_size']
        self.total_tokens_num = config.token_size_dict['total_tokens_size']
        self.patch_size = config.patch_size 
        self.num_tokens = self.total_tokens_num 
        self.freqs_cis_singlescale = compute_axial_cis(dim = config.n_embd  // self.n_head, end_x = self.patch_size[0], end_y = self.patch_size[1], theta = 1000.0)
        
    def forward(self, x,attn_mask):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        if T > self.pose_tokens_num+self.yaw_token_size: 
            q_B_scale2_d = q[:, :, self.pose_tokens_num+self.yaw_token_size:, :]
            k_B_scale2_d = k[:, :, self.pose_tokens_num+self.yaw_token_size:, :]
            q_out, k_out = apply_rotary_emb(q_B_scale2_d, k_B_scale2_d, freqs_cis=self.freqs_cis_singlescale[:T-self.pose_tokens_num-self.yaw_token_size]) 
            q = torch.cat([q[:, :, 0:self.pose_tokens_num+self.yaw_token_size, :], q_out], dim=2)
            k = torch.cat([k[:, :, 0:self.pose_tokens_num+self.yaw_token_size, :], k_out], dim=2)
        if attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None, :, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask = attn_mask.to(q.dtype), dropout_p=self.attn_dropout_rate).transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_drop(self.proj(y))
        return y
    
class SpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SpaceSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),  
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x),attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x

class CausalTimeSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.qk_norm = True
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        y = F.scaled_dot_product_attention(q, k, v, attn_mask = attn_mask.to(q.dtype), dropout_p=self.attn_dropout_rate).transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_drop(self.proj(y))
        return y

class CausalTimeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalTimeSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),  
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x

class CausalTimeSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_time_block = CausalTimeBlock(config)
        self.space_block = SpaceBlock(config)
        
    def forward(self, x, time_attn_mask,space_attn_mask):
        b, f, l, c = x.shape
        x = rearrange(x, 'b f l c -> (b l) f c')
        x = self.causal_time_block(x, time_attn_mask)
        x = rearrange(x, '(b l) f c -> b f l c', b=b, l=l, f=f)
        space_attn_mask = space_attn_mask.unsqueeze(1)
        space_attn_mask = space_attn_mask.repeat(1, x.shape[1], 1, 1)
        x = rearrange(x, 'b f l c -> (b f) l c', b=b, f=f)
        space_attn_mask = rearrange(space_attn_mask, 'b f l c -> (b f) l c')
        x = self.space_block(x,space_attn_mask)
        x = rearrange(x, '(b f) l c -> b f l c', b=b, f=f)
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=[12, 6], n_head=8, n_embd=1024,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, 
                 local_rank=0, condition_frames = 3, latent_size = (32, 32),
                 L = 32*32, token_size_dict=None, vae_emb_dim = 8,
                 pose_x_vocab_size=512, pose_y_vocab_size=512, yaw_vocab_size=512,
                 ):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked,
                           patch_size=latent_size,
                           condition_frames=condition_frames,
                           token_size_dict=token_size_dict)
        self.C = n_embd
        self.Cvae = vae_emb_dim 
        self.yaw_pose_emb_dim = 512
        self.pose_x_vocab_num = pose_x_vocab_size
        self.pose_y_vocab_num = pose_y_vocab_size
        self.yaw_vocab_num = yaw_vocab_size
        self.img_projector = nn.Sequential(
            nn.Linear(self.Cvae, self.C // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.C//2, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        self.pose_x_projector = nn.Sequential(
            nn.Linear(self.yaw_pose_emb_dim, self.yaw_pose_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_pose_emb_dim, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        self.pose_y_projector = nn.Sequential(
            nn.Linear(self.yaw_pose_emb_dim, self.yaw_pose_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_pose_emb_dim, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        self.yaw_projector = nn.Sequential(
            nn.Linear(self.yaw_pose_emb_dim, self.yaw_pose_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_pose_emb_dim, self.C, bias=False),
            nn.LayerNorm(self.C)
        )
        self.causal_time_space_num = config.n_layer[0]
        self.auto_regressive_num = config.n_layer[1]
        print("self.causal_time_space_num, self.auto_regressive_num", self.causal_time_space_num, self.auto_regressive_num)
        self.local_rank = local_rank
        self.img_token_size = token_size_dict['img_tokens_size']
        self.total_token_size = token_size_dict['total_tokens_size']
        self.yaw_token_size = token_size_dict['yaw_token_size']
        self.pose_token_size = token_size_dict['pose_tokens_size']
        self.condition_frames = condition_frames
        self.blank_pose_emb = torch.zeros((1, self.yaw_pose_emb_dim)).cuda()
        self.blank_yaw_emb = torch.zeros((1, self.yaw_pose_emb_dim)).cuda()
        self.sos_emb = nn.Parameter(torch.zeros(1, 1, self.C)) 
        nn.init.normal(self.sos_emb.data, mean=0, std=0.02)
        self.time_emb = nn.Parameter(torch.zeros(50, self.C)) 
        nn.init.normal(self.time_emb.data, mean=0, std=0.02)
        self.begin_ends = []
        self.causal_time_space_blocks = nn.Sequential(*[CausalTimeSpaceBlock(config) for _ in range(self.causal_time_space_num)])
        self.causal_space_blocks = nn.Sequential(*[CausalSpaceBlock(config) for _ in range(self.auto_regressive_num)])
        self.head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.vocab_size // 2, bias=False),
            nn.GELU(),
            nn.Linear(config.vocab_size // 2, config.vocab_size, bias=False)
        )
        self.pose_x_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, self.pose_x_vocab_num, bias=False),
            nn.GELU(),
            nn.Linear(self.pose_x_vocab_num, self.pose_x_vocab_num, bias=False),
        )
        self.pose_y_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, self.pose_y_vocab_num, bias=False),
            nn.GELU(),
            nn.Linear(self.pose_y_vocab_num, self.pose_y_vocab_num, bias=False)
        )
        self.yaw_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, self.yaw_vocab_num, bias=False),
            nn.GELU(),
            nn.Linear(self.yaw_vocab_num, self.yaw_vocab_num, bias=False)
        )
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        matrix = torch.tril(torch.ones(condition_frames, condition_frames))
        time_causal_mask = torch.where(matrix==0, float('-inf'), matrix)
        time_causal_mask = torch.where(matrix==1, 0, time_causal_mask)
        self.mask_time = time_causal_mask.contiguous().cuda()
        matrix_1 = torch.tril(torch.ones(self.total_token_size, self.total_token_size))
        seq_causal_mask = torch.where(matrix_1==0, float('-inf'), matrix_1)
        seq_causal_mask = torch.where(matrix_1==1, 0, seq_causal_mask)
        self.mask_ar = seq_causal_mask.contiguous().cuda()
        self.mask_ar_droppose = torch.clone(self.mask_ar)
        self.mask_ar_droppose[:, 1:1+self.pose_token_size+self.yaw_token_size] = float('-inf')
        mask_spatial = torch.ones(self.total_token_size, self.total_token_size)
        self.mask_spatial = mask_spatial.cuda()
        self.mask_spatial_droppose = torch.clone(self.mask_spatial)
        self.mask_spatial_droppose[:, :3] = float('-inf')
        self.mask_spatial_droppose[:3, :] = float('-inf')
        self.mask_spatial_droppose[0, 0] = 1.0
        self.mask_spatial_droppose[1, 1] = 1.0
        self.mask_spatial_droppose[2, 2] = 1.0

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_yaw_pose_emb(self, pose_indices, yaw_indices):
        if pose_indices == None:
            yaw_indices_normalize = yaw_indices / self.yaw_vocab_num
            yaw_emb = get_fourier_embeds_from_coordinates( 
                self.yaw_pose_emb_dim,
                yaw_indices_normalize)
            return yaw_emb, None, None
        elif pose_indices is not None and (pose_indices.shape[-1]==1):
            yaw_indices_normalize = yaw_indices / self.yaw_vocab_num
            pose_x_indices_normalize = pose_indices[:, :, 0:1] / self.pose_x_vocab_num
            yaw_pose_emb = get_fourier_embeds_from_coordinates(
                self.yaw_pose_emb_dim,
                torch.cat([yaw_indices_normalize, pose_x_indices_normalize], dim=-1), )
            yaw_emb, pose_x_emb = torch.split(yaw_pose_emb, dim=2, split_size_or_sections=1)
            return yaw_emb, pose_x_emb, None
        else :
            yaw_indices_normalize = yaw_indices / self.yaw_vocab_num
            pose_x_indices_normalize = pose_indices[:, :, 0:1] / self.pose_x_vocab_num
            pose_y_indices_normalize = pose_indices[:, :, 1:2] / self.pose_y_vocab_num
            yaw_pose_emb = get_fourier_embeds_from_coordinates(
                self.yaw_pose_emb_dim,
                torch.cat([yaw_indices_normalize, pose_x_indices_normalize, pose_y_indices_normalize], dim=-1), 
                )
            yaw_emb, pose_x_emb, pose_y_emb = torch.split(yaw_pose_emb, dim=2, split_size_or_sections=1)
            return yaw_emb, pose_x_emb, pose_y_emb
    
    def organize_attn_mask(self, drop_flg):
        mask_spatial = []
        mask_ar = []
        for flg_i in drop_flg:
            if flg_i:
                mask_spatial.append(self.mask_spatial_droppose)
                mask_ar.append(self.mask_ar_droppose)
            else:
                mask_spatial.append(self.mask_spatial)
                mask_ar.append(self.mask_ar)
        return torch.stack(mask_spatial, dim=0), torch.stack(mask_ar, dim=0)
            
    def forward(self, feature_total, pose_indices_total, yaw_indices_total, drop_flag=False):
        mask_spatial_curr, mask_ar_curr = self.organize_attn_mask(drop_flag)
        yaw_emb_total, pose_x_emb_total, pose_y_emb_total = self.get_yaw_pose_emb(pose_indices_total, yaw_indices_total)
        pose_x_token_embeddings = self.pose_x_projector(pose_x_emb_total) 
        pose_y_token_embeddings = self.pose_y_projector(pose_y_emb_total)  
        yaw_token_embeddings = self.yaw_projector(yaw_emb_total) 
        feature_embeddings = self.img_projector(feature_total)  
        input_pose_x_token_embeddings = pose_x_token_embeddings[:, :-1, ...]
        input_pose_y_token_embeddings = pose_y_token_embeddings[:, :-1, ...]
        input_yaw_token_embeddings = yaw_token_embeddings[:, :-1, ...]
        input_feature_embeddings = feature_embeddings[:, :-1, ...]
        add_pose_x_embeddings = pose_x_token_embeddings[:, 1:, ...]
        add_pose_y_embeddings = pose_y_token_embeddings[:, 1:, ...]
        add_yaw_embeddings = yaw_token_embeddings[:, 1:, ...]
        add_feature_embeddings = feature_embeddings[:, 1:, ...]
        yaw_pose_scale_token_embeddings = torch.cat([input_yaw_token_embeddings, input_pose_x_token_embeddings, input_pose_y_token_embeddings, input_feature_embeddings], dim=2)
        B, F, _, _ = yaw_pose_scale_token_embeddings.shape
        time_emb_F = self.time_emb[:F, :].unsqueeze(0)  
        time_emb_F = torch.repeat_interleave(time_emb_F[:, :, None, :], self.total_token_size, dim=2)  
        time_space_token_embeddings = yaw_pose_scale_token_embeddings + time_emb_F 
        for i in range(self.causal_time_space_num):
            time_space_token_embeddings = self.causal_time_space_blocks[i](time_space_token_embeddings, self.mask_time, mask_spatial_curr)
        auto_regressive_token_embeddings = rearrange(time_space_token_embeddings, 'B F L C -> (B F) L C', B=B, F=F) 
        sos_token_embeddings = self.sos_emb.unsqueeze(0).repeat((B, F, 1, 1))  
        query_embeddings = torch.cat([sos_token_embeddings, add_yaw_embeddings, add_pose_x_embeddings, add_pose_y_embeddings, add_feature_embeddings[:, :, :-1, :]], dim=2)
        intra_query_embeddings = rearrange(query_embeddings, 'B F L C -> (B F) L C', B=B, F=F)
        auto_regressive_token_embeddings = intra_query_embeddings + auto_regressive_token_embeddings
        for i in range(self.auto_regressive_num):
            auto_regressive_token_embeddings = self.causal_space_blocks[i](auto_regressive_token_embeddings, mask_ar_curr)
        out_img_embed = auto_regressive_token_embeddings[:, -self.img_token_size:, :]
        img_logits = self.head(out_img_embed)
        out_pose_x_embed = auto_regressive_token_embeddings[:, 1:2, :]
        pose_x_logits = self.pose_x_head(out_pose_x_embed)
        out_pose_y_embed = auto_regressive_token_embeddings[:, 2:3, :]
        pose_y_logits = self.pose_y_head(out_pose_y_embed)
        out_yaw_embed = auto_regressive_token_embeddings[:, 0:1, :]
        yaw_logits = self.yaw_head(out_yaw_embed)
        out = {
            'yaw_logits': yaw_logits,
            'pose_x_logits': pose_x_logits,
            'pose_y_logits': pose_y_logits,
            'img_logits': img_logits
        }
        return out
    
    @torch.no_grad()
    def eval_first_second(self, feature, pose_indices, yaw_indices, drop_flag):
        mask_spatial_curr, mask_ar_curr = self.organize_attn_mask(drop_flag)
        yaw_emb, pose_x_emb, pose_y_emb = self.get_yaw_pose_emb(pose_indices, yaw_indices) 
        pose_x_token_embeddings = self.pose_x_projector(pose_x_emb) 
        pose_y_token_embeddings = self.pose_y_projector(pose_y_emb) 
        yaw_token_embeddings = self.yaw_projector(yaw_emb) 
        feature_embeddings = self.img_projector(feature)
        yaw_pose_scale_token_embeddings = torch.cat([yaw_token_embeddings, pose_x_token_embeddings, pose_y_token_embeddings, feature_embeddings], dim=2)
        B, F, _, _ = yaw_pose_scale_token_embeddings.shape
        time_emb_F = self.time_emb[:F, :].unsqueeze(0)  
        time_emb_F = torch.repeat_interleave(time_emb_F[:, :, None, :], self.total_token_size, dim=2)  
        time_space_token_embeddings = yaw_pose_scale_token_embeddings + time_emb_F
        for i in range(self.causal_time_space_num):
            time_space_token_embeddings = self.causal_time_space_blocks[i](time_space_token_embeddings, self.mask_time, mask_spatial_curr)     
        auto_regressive_token_embeddings = time_space_token_embeddings[:, -1:, :, :]  
        auto_regressive_token_embeddings = rearrange(auto_regressive_token_embeddings, 'B F L C -> (B F) L C', B=B, F=1)
        return auto_regressive_token_embeddings

    @torch.no_grad()
    def eval_third(self, auto_regressive_token_embeddings, yaws=None, poses=None, new_tokens=None, drop_flag=False):
        mask_spatial_curr, mask_ar_curr = self.organize_attn_mask(drop_flag)
        B, L1, C = auto_regressive_token_embeddings.shape
        sos_embeddings =self.sos_emb.repeat((B, 1, 1))
        if (yaws is not None) and (poses is None) and (new_tokens is None):
            yaw_emb, _, _ = self.get_yaw_pose_emb(None, yaws)
            yaw_embeddings = self.yaw_projector(yaw_emb[:, :, 0, :])
            intra_query_embeddings = torch.cat([sos_embeddings, yaw_embeddings], dim=1)
        elif (yaws is not None) and (poses is not None)  and (new_tokens is None):
            yaw_emb, pose_x_emb, pose_y_emb = self.get_yaw_pose_emb(poses, yaws)
            yaw_embeddings = self.yaw_projector(yaw_emb[:, :, 0, :])
            pose_x_embeddings = self.pose_x_projector(pose_x_emb[:, :, 0, :])
            if poses.shape[2] == 2:
                pose_y_embeddings = self.pose_y_projector(pose_y_emb[:, :, 0, :])
                intra_query_embeddings = torch.cat([sos_embeddings, yaw_embeddings, pose_x_embeddings, pose_y_embeddings], dim=1)
            else:
                intra_query_embeddings = torch.cat([sos_embeddings, yaw_embeddings, pose_x_embeddings], dim=1)
        elif (yaws is not None) and (poses is not None) and (new_tokens is not None):
            yaw_emb, pose_x_emb, pose_y_emb = self.get_yaw_pose_emb(poses, yaws)
            yaw_embeddings = self.yaw_projector(yaw_emb[:, :, 0, :])
            pose_x_embeddings = self.pose_x_projector(pose_x_emb[:, :, 0, :])
            pose_y_embeddings = self.pose_y_projector(pose_y_emb[:, :, 0, :])
            new_embeddings = self.img_projector(new_tokens)
            intra_query_embeddings = torch.cat([sos_embeddings, yaw_embeddings, pose_x_embeddings, pose_y_embeddings, new_embeddings], dim=1)
        else:
            intra_query_embeddings = sos_embeddings
        _, L, _ = intra_query_embeddings.shape  
        assert L == L1
        auto_regressive_token_embeddings = intra_query_embeddings + auto_regressive_token_embeddings
        for i in range(self.auto_regressive_num):
            auto_regressive_token_embeddings = self.causal_space_blocks[i](auto_regressive_token_embeddings, mask_ar_curr[:, :L1, :L1])
        img_logits = None
        pose_x_logits = None
        pose_y_logits = None
        yaw_embed = auto_regressive_token_embeddings[:, 0:self.yaw_token_size, :]
        yaw_logits = self.yaw_head(yaw_embed)
        if L1 > 3:
            img_embed = auto_regressive_token_embeddings[:, (self.yaw_token_size + self.pose_token_size):, :]
            img_logits = self.head(img_embed)
        if L1 > 2:
            pose_y_embed = auto_regressive_token_embeddings[:, self.yaw_token_size+1:self.yaw_token_size + self.pose_token_size, :]
            pose_y_logits = self.pose_x_head(pose_y_embed)
        if L1 > 1:
            pose_x_embed = auto_regressive_token_embeddings[:, self.yaw_token_size:self.yaw_token_size+1, :]
            pose_x_logits = self.pose_x_head(pose_x_embed)
        out = {
            'yaw_logits': yaw_logits,
            'pose_x_logits': pose_x_logits,
            'pose_y_logits': pose_y_logits,
            'img_logits': img_logits
        }
        return out