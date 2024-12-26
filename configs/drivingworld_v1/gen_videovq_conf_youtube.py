seed = 1234


datasets_paths=dict(    
    nuplan_json_root=  '/path/json', 
    nuplan_root= '/path/images', 

    youtube_root='/path/json',
    youtube_val_json_root='/path/images',
)

val_data_list=['youtube']
image_size=[256, 512]
downsample_fps=5


n_layer=[12, 6]
n_embd=1536
gpt_type= "ar" 

pkeep = 0.7
condition_frames=15
pose_x_vocab_size=128
pose_y_vocab_size=128
yaw_vocab_size=512


codebook_size=16384
codebook_embed_dim=32
downsample_size=16
vq_model="VideoVQ-16"
vq_ckpt="./pretrained_models/vqvae.pt"
video_vq_temp_frames=condition_frames + 1



sampling_mtd="top_k"  

top_k=30
temperature_k=1.0

top_p=0.8
temperature_p=1.0
