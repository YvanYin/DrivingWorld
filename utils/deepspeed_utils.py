import os

def get_deepspeed_config(args):
        config_params = {
            'train_batch_size': int(os.environ['WORLD_SIZE']) * args.batch_size,
        }
        config_params['flops_profiler'] = {
            'enabled': False,
            'profile_step': 1,
            'module_depth': -1,
            'top_modules': 3,
            'detailed': True,
        }



        

        config_params["zero_optimization"] = {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size":  1e6, 
            "overlap_comm": True, 
            "reduce_scatter": True,
            "reduce_bucket_size": 1e6, 
            "contiguous_gradients": False, 
        }


















        config_params['bf16'] = {
            "enabled": True,
        }
        config_params['zero_allow_untested_optimizer'] = True

        return config_params