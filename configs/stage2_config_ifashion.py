project_name = 'SD3-joint-stage2'
run_name = 'joint-stage2'

stage2_data_config = dict(
    train_data_path='/mnt/raid1/mzyu/dataset/ifashion/processed/train.npy',
    item_info_path='processed_info/stage2_ifashion/item_info.npy',
    preference_path='processed_info/stage2_ifashion/train_preference.npy',
    id_cate_dict_path='/mnt/raid1/mzyu/dataset/ifashion/id_cate_dict.npy',
    image_paths_path='processed_info/stage2_ifashion/all_item_image_paths.npy',
    image_root='/mnt/raid1/mzyu/dataset/ifashion/semantic_category',
    resolution=512,
    max_length=256,
    mask_target_keys=["Color", "Material", "Design features", "Clothing Fashion Style"],
    num_context_items=3,
)

# resume_from_legacy = 'pretrained_models/dual_diff_sd3_512_base/transformer/diffusion_pytorch_model.safetensors'
resume_from_legacy = 'output/ifashion/stage1_models_t5/0020001/transformer/diffusion_pytorch_model.safetensors'
pretrained_mask_emb = 'pretrained_models/aligned_t5_mask_emb/mask_token_emb.00-of-01.pth'
noise_scheduler_pretrained = 'pretrained_models/stable-diffusion-3-medium-diffusers/scheduler'
sd3_pipeline_load_from = 'pretrained_models/stable-diffusion-3-medium-diffusers'

training = dict(
    sampling_eps=1e-3,
    antithetic_sampling=True,
    importance_sampling=False,
    ignore_padding=False,
    caption_training_weight=0.2,
)

num_workers = 8
global_batch_size = 1
micro_batch_size = 1

grad_clip = 2.0

lr = 3.e-5
wd = 1.e-2
num_warmup_steps = 2000
max_steps = 800_000
ema_steps = 100_000
log_every = 25
ckpt_every = 2500

ema_rate = 0.9995
seed = 1234
