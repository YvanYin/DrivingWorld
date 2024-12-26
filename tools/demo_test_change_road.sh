cd ..

CUDA_VISIBLE_DEVICES=6 python3 tools/test_change_road.py \
--config "configs/drivingworld_v1/gen_videovq_conf_nuplan.py" \
--exp_name "demo_dest" \
--load_path "" \
--save_video_path "./outputs/change_road"