# seed
seed = 1234

# dataset
datasets_paths=dict(    
    demo_root = './data/'
)
test_data_list=['demo']
image_size=[256, 512]

# model
n_layer=[12, 6]
n_embd=1536
gpt_type= "ar" 
pkeep = 0.7
condition_frames=15
pose_x_vocab_size=128
pose_y_vocab_size=128
yaw_vocab_size=512

# video vqvae
codebook_size=16384
codebook_embed_dim=32
downsample_size=16
vq_model="VideoVQ-16"
vq_ckpt="./pretrained_models/vqvae.pt"
video_vq_temp_frames=condition_frames + 1

# sampling
sampling_mtd="top_k"  
top_k=30
temperature_k=1.0
top_p=0.8
temperature_p=1.0
