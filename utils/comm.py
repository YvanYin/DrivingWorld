import os
import socket
import torch

def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port 

def _is_free_port(port):
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)
    
def _init_dist_envi(args):
    num_gpus_per_node = torch.cuda.device_count()
    if num_gpus_per_node > 1:
        args.distributed = True
    else:
        args.distributed = False
    if 'MASTER_PORT' not in os.environ:    
        if _is_free_port(14500):
            master_port = '14500'
        else:
            master_port = str(_find_free_port())
        os.environ['MASTER_PORT'] = master_port
    if 'MASTER_ADDR' not in os.environ:
        master_addr = '127.0.0.1' 
        os.environ['MASTER_ADDR'] = master_addr
    if 'WORLD_SIZE' not in os.environ:
        num_nodes = 1
        world_size = num_nodes * num_gpus_per_node
        os.environ['WORLD_SIZE'] = str(world_size)