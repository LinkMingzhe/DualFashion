project_name = 'SD3-joint'
run_name = 'joint-pretrain'

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

i2t_json_lst = [
            'meta_data.json',
            'meta_data.json',
            ]

# i2t
i2t_data_config = dict(
    roots=[
            '/mnt/bn/aigc-us/zjl/openimages/data/',
            '/mnt/bn/aigc-us/zjl/sharegpt4v_processed_data/data/',

            ],
    json_lst=i2t_json_lst,
    resolution=512,
    org_caption_key=['caption',
                      'caption',],
    re_caption_key=['caption',
                     'caption',
                     ],
    max_length=256,

)

resume_from_legacy = None
pretrained_mask_emb = '/mnt/bn/us-aigc-temp/zjl_data/mask_token_emb.00-of-01.pth'
noise_scheduler_pretrained = '/mnt/bn/us-aigc-temp/huggingface_model/sd3_scheduler'
sd3_pipeline_load_from = '/mnt/bn/us-aigc-temp/huggingface_model/sd3_pipeline'

training = dict(
    sampling_eps=1e-3,
    antithetic_sampling=True,
    importance_sampling=False,
    ignore_padding=False,
    caption_training_weight=0.2,
)

# training setting
num_workers = 8
global_batch_size = 256 
micro_batch_size = 4    

grad_clip = 2.0 

lr = 3.e-5
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
