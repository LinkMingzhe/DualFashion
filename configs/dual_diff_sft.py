project_name = 'SD3-joint-finetune'
run_name = 'test'

resolution = 512
# t2i
t2i_json_lst = ['meta_data_all.json',
            'meta_data_all.json',
            'meta_data_all.json',
            ]
t2i_data_config = dict(
    roots=[
            '/mnt/bn/aigc-us/zjl/recap_laion_en_aesthetic/',
            '/mnt/bn/datacompv6/zijie/data/recap_datacomp_1b/aesthetic_subset/',
            '/mnt/bn/aigc-us/zjl/recap_datacomp_aesthetic_subset2/',
            ],
    json_lst=t2i_json_lst,
    resolution=512,
     org_caption_key=['org_caption',
                      'org_caption',
                      'org_caption',
                      ],
    re_caption_key=['re_caption',
                     're_caption',
                     're_caption',
                     ],
    max_length=256

)


load_pretrained_model = False
freeze_vocab = False

resume_from_legacy = None#'/mnt/bn/us-aigc-temp/zjl_new_experiments/joint_diffusion_sd3/oct6_with_text/checkpoints/epoch_0_step_950000.pth'

noise_scheduler_pretrained = '/mnt/bn/us-aigc-temp/huggingface_model/sd3_scheduler'

training = dict(
    sampling_eps=1e-3,
    antithetic_sampling=True,
    importance_sampling=False,
    ignore_padding=False,
    caption_training_weight=1.0,
)

# training setting
num_workers = 4 # not a bottleneck
global_batch_size = 512     
micro_batch_size = 8      
grad_clip = 2.0

# optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr = 3.e-5
wd = 1.e-2
num_warmup_steps=2000
max_steps = 90_000 # 90k
ema_steps = 5_000
log_every = 25
ckpt_every = 2_500
text_ratio = 1

train_real_cap_ratio = 0.1

# mixed_precision = 'no'
ema_rate = 0.9995  
seed = 1234
