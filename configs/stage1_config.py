project_name = 'SD3-joint'
run_name = 'joint-pretrain'

# t2i
t2i_json_lst = ['text_to_image_30k.json']
t2i_data_config = dict(
    roots=[
            './processed_info/stage1_ifashion/',
            ],
    json_lst=t2i_json_lst,
    resolution=512,
    org_caption_key=['org_caption'],
    re_caption_key=['re_caption'],
    max_length=256
)

i2t_json_lst = ['image_to_text_30k.json']

# i2t
i2t_data_config = dict(
    roots=[
        './processed_info/stage1_ifashion/',
    ],
    json_lst=i2t_json_lst,
    resolution=512,
    org_caption_key=['caption'],
    re_caption_key=['caption'],
    max_length=256,
)


resume_from_legacy = 'pretrained_models/dual_diff_sd3_512_base/transformer/diffusion_pytorch_model.safetensors'
pretrained_mask_emb = 'pretrained_models/aligned_t5_mask_emb/mask_token_emb.00-of-01.pth'
noise_scheduler_pretrained = 'pretrained_models/stable-diffusion-3-medium-diffusers/scheduler'
sd3_pipeline_load_from = 'pretrained_models/stable-diffusion-3-medium-diffusers'

# resume_from_legacy = None
# pretrained_mask_emb = '/mnt/bn/us-aigc-temp/zjl_data/mask_token_emb.00-of-01.pth'
# noise_scheduler_pretrained = '/mnt/bn/us-aigc-temp/huggingface_model/sd3_scheduler'
# sd3_pipeline_load_from = '/mnt/bn/us-aigc-temp/huggingface_model/sd3_pipeline'


training = dict(
    sampling_eps=1e-3,
    antithetic_sampling=True,
    importance_sampling=False,
    ignore_padding=False,
    caption_training_weight=0.2,
)

# training setting
num_workers = 8
global_batch_size = 4
micro_batch_size = 2

grad_clip = 2.0

lr = 3.e-5
# lr = 5.e-6
wd = 1.e-2
num_warmup_steps=2000
max_steps = 800_000
ema_steps = 100_000
log_every = 25
ckpt_every = 2500

train_real_cap_ratio = -1.0

# mixed_precision = 'no'
ema_rate = 0.9995
seed = 1234
