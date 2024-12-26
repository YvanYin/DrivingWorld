import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import imageio
import os
import json
import logging
logger = logging.getLogger('base')

def init_optimizer(model, lr=1e-4, weight_decay=1e-3):
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim


def init_lr_schedule(optimizer, milstones=[1000000, 1500000, 2000000], gamma=0.5):
    scheduler = MultiStepLR(optimizer, milestones=milstones, gamma=gamma)
    return scheduler


def save_model(self, path, epoch, rank=0):
    if rank == 0:
        save
        torch.save(self.model.state_dict(),'{}/tvar_{}.pkl'.format(path, str(epoch)))  


def save_ckpt(args, path, model, optimizer=None, scheduler=None, curr_iter=0, curr_epoch=None):
    """
    Save the model, optimizer, lr scheduler.
    """

    ckpt = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),

    )
    
    ckpt_path = '{}/tvar_{}.pkl'.format(path, str(curr_iter))

    torch.save(ckpt, ckpt_path)

    print(f'#### Save model: {ckpt_path}')

def resume_ckpt(local_rank, args, model, optimizer=None):
    
    resume_load_path = '{}/tvar_{}.pkl'.format(args.save_model_path, str(args.resume_step))
    print(local_rank,": loading...... ", resume_load_path)
    ckpt_file = torch.load(resume_load_path, map_location="cpu")
    
    if 'optimizer_state_dict' in ckpt_file:
        if optimizer is not None:
            optimizer.load_state_dict(ckpt_file['optimizer_state_dict'])
        print(local_rank, f'Rank: {local_rank}, Successfully loaded optimizer from {resume_load_path}.')
    if 'model_state_dict' in ckpt_file:


        model = load_parameters(model, ckpt_file)
        print(local_rank, f'Rank: {local_rank}, Successfully loaded model from {resume_load_path}.')
    else:

        model = load_parameters(model, ckpt_file)
        print(local_rank, f'Rank: {local_rank}, Successfully loaded model from {resume_load_path}.')
    return model, optimizer



def load_parameters(model, load_ckpt_file):
    if 'model_state_dict' in load_ckpt_file:
        ckpt = load_ckpt_file['model_state_dict']
    else:
        ckpt = load_ckpt_file
    ckpt_state_dict = {}
    for key, val in ckpt.items():
        if key in model.state_dict() and val.shape == model.state_dict()[key].shape:
            ckpt_state_dict[key] = val
        elif key not in model.state_dict():
            print(f"!!!! {key} not exists in model.")

            continue
        elif val.shape != model.state_dict()[key].shape:
            print(f"!!!! Shape of ckpt's {key} is {val.shape}, but model's shape is {model.state_dict()[key].shape}")
            if key == 'pos_emb':
                ckpt_state_dict[key] = model.state_dict()[key]
                B, H, W = val.shape
                B1, H1, W1 = ckpt_state_dict[key].shape
                B, H, W = min(B, B1), min(H, H1), min(W, W1)
                ckpt_state_dict[key][:B, :H, :W] = val[:B, :H, :W]
                print(f"!!!! load {B} {H} {W}")
            elif key == 'img_projector.0.weight':
                ckpt_state_dict[key] = torch.zeros_like(model.state_dict()[key])
                H, W = val.shape
                H1, W1 = ckpt_state_dict[key].shape
                H, W = min(H, H1), min(W, W1)
                ckpt_state_dict[key][:H, :W] = val[:H, :W]
                print(f"!!!! load {H} {W}")
            else:
                print(f"!!!! no weight loaded for {key}")
                ckpt_state_dict[key] = model.state_dict()[key]
    newparas_not_in_ckpt = set(list(model.state_dict().keys())).difference(list(ckpt.keys()))
    for key in newparas_not_in_ckpt:
        print(f"!!!! {key} required by the model does not exist in ckpt. Shape is {model.state_dict()[key].shape}")
        ckpt_state_dict[key] = model.state_dict()[key]
    model.load_state_dict(ckpt_state_dict, strict=True)
    return model


def save_ckpt_deepspeed(args, path, model, optimizer=None, scheduler=None, curr_iter=0, curr_epoch=None):
    """
    Save the model, optimizer, lr scheduler.
    """

    client_sd = dict(
        curr_iter=curr_iter,
    )
    torch.distributed.barrier()
    os.makedirs(path, exist_ok=True)
    ckpt_path = path
    print(f'#### Deepspeed, Save model to {ckpt_path}')
    model.save_checkpoint(os.path.abspath(ckpt_path), curr_iter, client_sd, save_latest=True) 



def load_from_deepspeed_ckpt(args, model):
    if args.load_from_deepspeed is not None:
        print('#### Before deepspeed load ckpt, img_projector.0.weight sum:', torch.sum(model.model.state_dict()['img_projector.0.weight']))
        load_path, client_sd = model.load_checkpoint(args.load_from_deepspeed, load_module_strict=False, load_module_only=True)
        if load_path is None or client_sd is None:
            if args.load_from_deepspeed.endswith("/"):
                args.load_from_deepspeed = args.load_from_deepspeed[:-1]
            tag = os.path.split(args.load_from_deepspeed)[-1]
            resume_raw_dir = os.path.dirname(args.load_from_deepspeed) 
            load_path, client_sd = model.load_checkpoint(resume_raw_dir, tag, load_module_strict=False, load_module_only=True)
        print('#### After deepspeed load ckpt, img_projector.0.weight sum:', torch.sum(model.model.state_dict()['img_projector.0.weight']))
    if args.resume_step > 0: 
        print('#### Before deepspeed resume ckpt, img_projector.0.weight sum:', torch.sum(model.model.state_dict()['img_projector.0.weight']))
        resume_load_path = '{}/{}'.format(args.save_model_path, str(args.resume_step))
        load_path, client_sd = model.load_checkpoint(resume_load_path)
        if load_path is None or client_sd is None:
            if resume_load_path.endswith("/"):
                resume_load_path = resume_load_path[:-1]
            tag = os.path.split(resume_load_path)[-1]
           
            load_path, client_sd = model.load_checkpoint(args.save_model_path, tag)
        
        print('#### After deepspeed resume, img_projector.0.weight sum:', torch.sum(model.model.state_dict()['img_projector.0.weight']))
    return model


def log_dict_args(dict_args):
    """
    Logs a dictionary of arguments.
    
    Args:
        dict_args (dict): The dictionary of arguments to be logged.
    """

    json_args = json.dumps(dict_args, indent=4)
    

    logging.info(f"Arguments: {json_args}")