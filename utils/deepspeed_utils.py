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
        # config_params['zero_optimization'] ={
        #     'stage': 1,
        # }
        
        # adjust following parameters with comments.
        config_params["zero_optimization"] = {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size":  1e6, #3e8, #
            "overlap_comm": True, #False,
            "reduce_scatter": True,
            "reduce_bucket_size": 1e6, #3e8, #
            "contiguous_gradients": False, #True, #
        }


        # config_params["train_micro_batch_size_per_gpu"] = int(args.batch_size),
        # # config_params["train_batch_size"]="auto",
        # config_params["gradient_accumulation_steps"]="auto",
        # config_params['zero_optimization'] = {
        #     "stage": 3,
        #     "overlap_comm": True, 
        #     "contiguous_gradients": True, 
        #     "sub_group_size": 1e9, 
        #     "reduce_bucket_size": "auto", 
        #     "stage3_prefetch_bucket_size": "auto",
        #     "stage3_param_persistence_threshold": "auto",
        #     "stage3_max_live_parameters": 1e9,
        #     "stage3_max_reuse_distance": 1e9,
        #     "stage3_gather_16bit_weights_on_model_save": True

        # }
        config_params['bf16'] = {
            "enabled": True,
        }
        config_params['zero_allow_untested_optimizer'] = True

        return config_params
