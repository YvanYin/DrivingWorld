cd ..

CUDA_VISIBLE_DEVICES=0 python3 tools/test_long_term_nuplan.py \
--config "configs/drivingworld_v1/gen_videovq_conf_nuplan.py" \
--exp_name "demo_test" \
--load_path "" \
--save_video_path "./outputs/long_term"