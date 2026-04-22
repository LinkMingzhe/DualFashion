project_name = "SD3-stage3"
run_name = "stage3-text-only"

stage3_data_config = dict(
    train_data_path="{your_path}/processed_info/stage3_data_augmentation/data_augmentation.npy",
    id_cate_dict_path="{your_path}/dataset/ifashion/id_cate_dict.npy",
    max_length=256,
    mask_target_keys=["Color", "Material", "Design features", "Clothing Fashion Style"],
)

resume_from_legacy = "{your_path}/{checkpoint_path}/transformer/diffusion_pytorch_model.safetensors"
sd3_pipeline_load_from = "{your_path}/pretrained_models/stable-diffusion-3-medium-diffusers"

training = dict(
    sampling_eps=1e-3,
    antithetic_sampling=True,
    importance_sampling=False,
    ignore_padding=False,
)

num_workers = 4
global_batch_size = 1

grad_clip = 2.0

lr = 3e-5
wd = 1e-2
num_warmup_steps = 2000
max_steps = 100_000
log_every = 50
