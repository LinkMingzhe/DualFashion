
"""
A minimal training script using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
import json
import logging
import os
import random
import socket
from time import time
import warnings
import gc
import sys
from sd3_modules.dual_diff_pipeline import DualDiffSD3Pipeline
from safetensors.torch import load_file
import shutil


# byte-wandb huggingface
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
old_metadata = importlib_metadata.metadata

def new_metadata(name):
    if name == 'wandb':
        name =  'byted-wandb'
    return old_metadata(name)

importlib_metadata.metadata = new_metadata

from PIL import Image
from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from transformers import AutoTokenizer, get_constant_schedule_with_warmup

from grad_norm import calculate_l2_grad_norm, get_model_parallel_dim_dict, scale_grad
from parallel import distributed_init, get_intra_node_process_group

import wandb
from mmcv import Config

# from custom_dataset.ifashion_data import get_dataset
from custom_dataset.stage2_ifashion_data import Stage2IFashionData

from sd3_modules.stage2_sd3_loss_utils_v5 import ImageFlowMatchingLoss, TextMaskedDiffusionLoss
from diffusers import StableDiffusion3Pipeline as SDPipe
from sd3_modules.sd3_model import SD3JointModelFlexible

import warnings
warnings.filterwarnings("ignore")  # ignore warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MASK_TOKEN_IDS = 32099 # <'extra_id0'>, currently hard-coded

print_every = 200

MATCHING_LAMBDA = 0.1


class MatchingMLP(nn.Module):
    """MLP that matches context latents to target latent space."""

    def __init__(self, dropout_p: float = 0.1, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim
        self.flatten = nn.Flatten()
        self.hidden_layer: nn.Linear | None = None
        self.output_layer: nn.Linear | None = None
        self.activation: nn.Module | None = None
        self.dropout: nn.Module | None = None

    def _build_layers(self, input_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        hidden_dim = self.hidden_dim
        self.hidden_layer = nn.Linear(input_dim, hidden_dim).to(device=device, dtype=dtype)
        self.output_layer = nn.Linear(hidden_dim, input_dim).to(device=device, dtype=dtype)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_p)
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.flatten(x)
        if self.hidden_layer is None or self.output_layer is None:
            self._build_layers(flat.shape[-1], flat.device, flat.dtype)
        assert self.hidden_layer is not None and self.output_layer is not None
        assert self.activation is not None and self.dropout is not None
        hidden = self.hidden_layer(flat)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        out = self.output_layer(hidden)
        out = torch.tanh(out)
        return out.view_as(x)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999, copy=False):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())
    if not copy:
        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    else:
        # for initialization

        for name, param in model_params.items():
            ema_params[name].data.copy_(param.data)

def average_tensor(t: torch.Tensor, gradient_accumulation_steps: int):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return t / gradient_accumulation_steps

# initialize the learnable embedding and MLP
def ensure_stage2_image_conditioning_modules(model: nn.Module, latent_channels: int) -> None:
    if not hasattr(model, "stage2_target_role_embedding"):
        model.stage2_target_role_embedding = nn.Parameter(
            torch.randn(1, latent_channels, 1, 1) * 0.01
        )
    if not hasattr(model, "stage2_context_role_embedding"):
        model.stage2_context_role_embedding = nn.Parameter(
            torch.randn(1, latent_channels, 1, 1) * 0.01
        )
    if not hasattr(model, "stage2_matching_mlp"):
        model.stage2_matching_mlp = MatchingMLP()

def prepare_stage2_matching_mlp_from_state_dict(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    """Ensure the lazily-built stage2 matching MLP has materialized layers before loading weights."""
    mlp = getattr(model, "stage2_matching_mlp", None)
    if not isinstance(mlp, MatchingMLP):
        return
    if mlp.hidden_layer is not None and mlp.output_layer is not None:
        return

    hidden_weight_key = "stage2_matching_mlp.hidden_layer.weight"
    output_weight_key = "stage2_matching_mlp.output_layer.weight"
    if hidden_weight_key not in state_dict or output_weight_key not in state_dict:
        return

    example_param = next((param for param in model.parameters()), None)
    device = example_param.device if example_param is not None else torch.device("cpu")
    dtype = example_param.dtype if example_param is not None else torch.float32

    hidden_out_features, input_features = state_dict[hidden_weight_key].shape
    mlp.hidden_dim = hidden_out_features
    mlp._build_layers(input_features, device, dtype)

def initialize_stage2_matching_mlp_parameters(
    model: nn.Module,
    vae_config,
    device: torch.device,
) -> None:
    """Eagerly materialize the matching MLP parameters so optimizers capture their state."""
    mlp = getattr(model, "stage2_matching_mlp", None)
    if not isinstance(mlp, MatchingMLP):
        return
    if mlp.hidden_layer is not None and mlp.output_layer is not None:
        return

    latent_channels = getattr(vae_config, "latent_channels", None)
    sample_size = getattr(vae_config, "sample_size", None)
    block_out_channels = getattr(vae_config, "block_out_channels", None)
    if latent_channels is None or sample_size is None:
        return

    downsample_factor = 1
    if isinstance(block_out_channels, (list, tuple)) and len(block_out_channels) > 0:
        downsample_factor = 2 ** len(block_out_channels)
    latent_resolution = sample_size // downsample_factor
    if latent_resolution <= 0:
        return

    input_features = latent_channels * latent_resolution * latent_resolution
    example_param = next((param for param in model.parameters()), None)
    dtype = example_param.dtype if example_param is not None else torch.float32
    mlp._build_layers(input_features, device, dtype)

def reconcile_optimizer_state_dict(
    optimizer: torch.optim.Optimizer,
    loaded_state: dict,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Align a loaded optimizer state with the current optimizer structure.
    Resets states for any newly introduced parameters while preserving existing ones.
    """
    current_state = optimizer.state_dict()
    current_groups = current_state.get("param_groups", [])
    loaded_groups = loaded_state.get("param_groups", [])
    loaded_states = loaded_state.get("state", {})

    new_state_entries: dict[int, dict] = {}
    mismatch = False

    for group_idx, current_group in enumerate(current_groups):
        current_params = current_group.get("params", [])
        loaded_params: list[int] = []
        if group_idx < len(loaded_groups):
            loaded_params = loaded_groups[group_idx].get("params", [])
        if len(current_params) != len(loaded_params):
            mismatch = True
        for param_pos, current_param_id in enumerate(current_params):
            if param_pos < len(loaded_params):
                loaded_param_id = loaded_params[param_pos]
                new_state_entries[current_param_id] = deepcopy(
                    loaded_states.get(loaded_param_id, {})
                )
            else:
                new_state_entries[current_param_id] = {}

    if len(loaded_groups) > len(current_groups):
        mismatch = True

    if mismatch and logger is not None:
        loaded_param_total = sum(len(g.get("params", [])) for g in loaded_groups)
        current_param_total = sum(len(g.get("params", [])) for g in current_groups)
        logger.warning(
            "Loaded optimizer state has %d parameters; current optimizer expects %d. "
            "Parameters without saved state will be reset.",
            loaded_param_total,
            current_param_total,
        )

    return {"state": new_state_entries, "param_groups": deepcopy(current_groups)}

# get the incomplete information and the latent
def apply_stage2_image_conditioning(
    model: nn.Module, target_latents: torch.Tensor, context_latents: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float]:
    target_emb = target_latents + model.stage2_target_role_embedding.to(target_latents.dtype)
    context_summary = context_latents.mean(dim=1)
    context_emb = context_summary + model.stage2_context_role_embedding.to(context_summary.dtype)
    context_proj = model.stage2_matching_mlp(context_emb)
    return target_emb, context_proj, MATCHING_LAMBDA

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir, rank):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def save_model_index_json_file(target_file_path):
    source_file_path = "{your_path}/output/ifashion/model_index.json"
    new_folder_path = os.path.join(target_file_path, "model_index.json")
    shutil.copy(source_file_path, new_folder_path)


def read_config(file):
    # solve config loading conflict when multi-processes
    import time
    while True:
        config = Config.fromfile(file)
        if len(config) == 0:
            time.sleep(0.1)
            continue
        break
    return config

def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.encoder.block), # this assumes huggingface T5Encodermodel
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model

def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        # process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "h_sdp": ShardingStrategy._HYBRID_SHARD_ZERO2,
            "h_fsdp": ShardingStrategy.HYBRID_SHARD,
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(config):
    if config.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif config.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


#############################################################################
#                                Training Loop                              #
#############################################################################


def main(args):
    """
    Trains a MMDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    config = read_config(args.config)

    if args.training == "multiple":
        # parallel.py
        distributed_init(args)
        dp_world_size = fs_init.get_data_parallel_world_size()
        dp_rank = fs_init.get_data_parallel_rank()
        mp_world_size = fs_init.get_model_parallel_world_size()
        mp_rank = fs_init.get_model_parallel_rank()

        assert config.global_batch_size % dp_world_size == 0, "Batch size must be divisible by data parrallel world size."
        local_batch_size = config.global_batch_size // dp_world_size
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        setup_mixed_precision(args)
        print(f"Starting rank={rank}, seed={seed}, "
              f"world_size={dist.get_world_size()}.")

    if args.training == "single":
        # single GPU
        dp_world_size = 1
        dp_rank = 0
        mp_world_size = 1
        mp_rank = 0
        assert config.global_batch_size % dp_world_size == 0
        local_batch_size = config.global_batch_size
        rank = 0
        device = args.device
        seed = args.global_seed
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        setup_mixed_precision(args)

        gpu_index = args.device
        device_str = f"cuda:{gpu_index}"
        device = torch.device(device_str)
        print(f"Starting rank={rank}, seed={seed}, world_size=1.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir, rank)
        logger.info(f"Experiment directory: {args.results_dir}")
        # initialize wandb
        wandb.init(
            project = config.project_name,
            name = config.run_name,
        )
    else:
        logger = create_logger(None, rank)
    # logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))


    # load sd3 pipe
    sd_model_pipe = SDPipe.from_pretrained(config.sd3_pipeline_load_from, torch_dtype=torch.bfloat16).to(device)
    # sd_model_pipe = DualDiffSD3Pipeline.from_pretrained("{your_path}/pretrained_models/dual_diff_sd3_512_base",
    #                                                      torch_dtype=torch.bfloat16).to(device)
    sd_model_pipe.tokenizer_3.pad_token = sd_model_pipe.tokenizer_3.eos_token   # change padding token to eos token
    orig_sd_transformer = sd_model_pipe.transformer

    text_tokenizer = sd_model_pipe.tokenizer_3

    logger.info('sd pipeline loaded, text encoder was also prepared')
    logger.info(f"Creating text diffusion model from SD3")

    # the model load the config of transformer, is the core of training.
    model = SD3JointModelFlexible(
        len(sd_model_pipe.tokenizer_3),
        **orig_sd_transformer.config
    ).train()

    ensure_stage2_image_conditioning_modules(model, sd_model_pipe.vae.config.latent_channels)


    if args.training == "single":
        model.to(device)
        model = model.to(torch.bfloat16)
        initialize_stage2_matching_mlp_parameters(model, sd_model_pipe.vae.config, device)
    else:
        init_device = next((param.device for param in model.parameters()), torch.device("cpu"))
        initialize_stage2_matching_mlp_parameters(model, sd_model_pipe.vae.config, init_device)

    logger.info(f"Model trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # remove unnecessary stuff from sd3 model pipe to save memory
    # delete all the encoder/weight not used in sd model pipe
    sd_model_pipe.transformer = None
    sd_model_pipe.text_encoder_1 = None
    sd_model_pipe.text_encoder_2 = None
    sd_model_pipe.text_encoder_3.requires_grad_(False)
    sd_model_pipe.text_encoder_3.eval()

    gc.collect()
    torch.cuda.empty_cache()

    if config.resume_from_legacy:
        resume_path = config.resume_from_legacy
        logger.info(f'Resuming legacy model from {resume_path}')

        legacy_state_dict = load_file(
            resume_path,
            device="cpu",
        )
        prepare_stage2_matching_mlp_from_state_dict(model, legacy_state_dict)
        missing, unexpect = model.load_state_dict(
            legacy_state_dict,
            strict=False
        )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpect}')
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f'Pretrained model loaded from {resume_path}')

        if config.pretrained_mask_emb is not None:
            logger.info(f'Resuming t5 embedding from {config.pretrained_mask_emb}')
            mask_emb = torch.load(config.pretrained_mask_emb, map_location="cpu")
            sd_model_pipe.text_encoder_3.shared.weight.data[MASK_TOKEN_IDS] = mask_emb
            del mask_emb


    model_parallel_dim_dict = get_model_parallel_dim_dict(model)

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            logger.warning(f"Could not find existing checkpoints in {checkpoint_dir}")
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model)
    if args.resume:
        logger.info(f"Resuming model weights from: {args.resume}")
        resume_state_path = os.path.join(
            args.resume,
            f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
        )
        resume_state_dict = torch.load(
            resume_state_path,
            map_location="cpu",
        )
        prepare_stage2_matching_mlp_from_state_dict(model, resume_state_dict)
        model.load_state_dict(
            resume_state_dict,
            strict=True,
        )

        if os.path.exists(os.path.join(
                args.resume,
                f"consolidated_ema.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
        )):

            logger.info(f"Resuming ema weights from: {args.resume}")
            ema_state_path = os.path.join(
                args.resume,
                f"consolidated_ema.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
            )
            ema_state_dict = torch.load(
                ema_state_path,
                map_location="cpu",
            )
            prepare_stage2_matching_mlp_from_state_dict(model_ema, ema_state_dict)
            model_ema.load_state_dict(
                ema_state_dict,
                strict=True,
            )

    if args.training == "multiple":
        dist.barrier()

        model = setup_fsdp_sync(model, args)
        model_ema = setup_fsdp_sync(model_ema, args)
        # sd_model_pipe.text_encoder_3 = setup_lm_fsdp_sync(sd_model_pipe.text_encoder_3) # no need for this, quite slow

    # add t5 vocab embedding to optimizer
    # optimizer update two parts' parameter:model and sd_model_pipe.text_encoder_3.get_input_embeddings()
    opt = torch.optim.AdamW \
        (list(model.parameters() ) +list(sd_model_pipe.text_encoder_3.get_input_embeddings().parameters()), lr=config.lr, weight_decay=config.wd)
    scheduler = get_constant_schedule_with_warmup(opt,
                                                  num_warmup_steps=config.num_warmup_steps)
    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        if opt_state_world_size != get_world_size():
            logger.info(
                f"Resuming from a checkpoint with unmatched world size "
                f"({get_world_size()} vs. {opt_state_world_size}) "
                f"is currently not supported."
            )
        else:
            logger.info(f"Resuming optimizer states from: {args.resume}")
            opt.load_state_dict(
                reconcile_optimizer_state_dict(
                    opt,
                    torch.load(
                        os.path.join(
                            args.resume,
                            f"optimizer.{get_rank():05d}-of-{get_world_size():05d}.pth",
                        ),
                        map_location="cpu",
                    ),
                    logger,
                )
            )
            for param_group in opt.param_groups:
                param_group["lr"] = config.lr
                param_group["weight_decay"] = config.wd

        # resume scheduler
        scheduler_state = torch.load(
            os.path.join(
                args.resume,
                f"scheduler.pth",
            ),
            map_location="cpu",
        )
        scheduler.load_state_dict(scheduler_state)

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    # Setup data:
    logger.info("Creating dataset...")



    stage2_dataset = Stage2IFashionData(
        tokenizer=sd_model_pipe.tokenizer_3,
        **config.stage2_data_config,
    )
    if args.training == "multiple":
        stage2_sampler = DistributedSampler(stage2_dataset)
    else:
        stage2_sampler = RandomSampler(stage2_dataset)
    logger.info("Stage2 dataset created")
    logger.info(f"{config.stage2_data_config}")
    logger.info("*****************************************")

    stage2_dataloader = torch.utils.data.DataLoader(
        stage2_dataset,
        batch_size=int(local_batch_size),
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=None,
        sampler=stage2_sampler,
        drop_last=True,
        persistent_workers=True,
    )
    stage2_data_iter = iter(stage2_dataloader)

    # good, now we setup the loss function
    text_diffusion_loss_module = TextMaskedDiffusionLoss(
        config,
        model_pipe = sd_model_pipe,
    )

    image_diffusion_loss_module = ImageFlowMatchingLoss(
        model_pipe = sd_model_pipe,
        text_max_length=256,
    )


    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    # steps_chunk = config.total_steps_chunk
    # text_steps = int(steps_chunk * config.training.text_training_weight)
    # image_steps = steps_chunk - text_steps

    # logger.info(f'For every {steps_chunk}, text steps: {text_steps}, image steps: {image_steps}')

    step = 0
    # if not config.disable_skip_data:
    #     logger.info(f'Skipping {resume_step} steps in dataloader')
    #     while step < resume_step:
    #         if step % 100 ==0:
    #             logger.info(f'Skipped step {step} in dataloader')
    #             gc.collect()
    #             torch.cuda.empty_cache()
    #
    #         try:
    #             _ = next(stage2_data_iter)
    #         except StopIteration:
    #             stage2_data_iter = iter(stage2_dataloader)
    #             _ = next(stage2_data_iter)
    #         step += 1
    # else:
    #     step = resume_step

    running_caption_loss = 0.0
    running_image_loss = 0.0

    logger.info(f"Training for {config.max_steps:,} steps...")

    # save the untrained model:
    if step == 0:
        model_dir = os.path.join(args.results_dir, args.model_dir)
        model_save_dir = os.path.join(model_dir, f"{step + 1:07d}")
        os.makedirs(model_save_dir, exist_ok=True)

        model.save_pretrained(os.path.join(model_save_dir, "transformer"))
        sd_model_pipe.text_encoder_3.save_pretrained(os.path.join(model_save_dir, "text_encoder"))
        sd_model_pipe.tokenizer_3.save_pretrained(os.path.join(model_save_dir, "tokenizer"))
        sd_model_pipe.vae.save_pretrained(os.path.join(model_save_dir, "vae"))
        sd_model_pipe.scheduler.save_pretrained(os.path.join(model_save_dir, "scheduler"))
        save_model_index_json_file(model_save_dir)
        mask_emb = sd_model_pipe.text_encoder_3.shared.weight.data[MASK_TOKEN_IDS].clone().cpu()
        torch.save(mask_emb, os.path.join(model_save_dir, "mask_emb.pt"))

        logger.info(f"Saved pipeline weights to {model_save_dir}")

    while step < config.max_steps:

        try:
            stage2_batch = next(stage2_data_iter)
        except StopIteration:
            stage2_data_iter = iter(stage2_dataloader)
            stage2_batch = next(stage2_data_iter)

        target_images = stage2_batch["target_image"].to(device, non_blocking=True)
        context_images = stage2_batch["context_images"].to(device, non_blocking=True)

        caption_tokens = stage2_batch["caption_tokens"]
        caption_ids = caption_tokens["input_ids"].squeeze(1).to(device)
        caption_label_mask = caption_tokens["label_mask"].squeeze(1).to(device)
        caption_attention = caption_tokens.get("attention_mask")
        if caption_attention is not None:
            caption_attention = caption_attention.squeeze(1).to(device)

        preference_tokens = stage2_batch["preference_tokens"]
        preference_ids = preference_tokens["input_ids"].squeeze(1).to(device)
        preference_attention = preference_tokens.get("attention_mask")
        if preference_attention is not None:
            preference_attention = preference_attention.squeeze(1).to(device)

        sentence_tokens = stage2_batch["sentence_tokens"]
        sentence_ids = sentence_tokens["input_ids"].squeeze(1).to(device)
        sentence_attention = sentence_tokens.get("attention_mask")
        if sentence_attention is not None:
            sentence_attention = sentence_attention.squeeze(1).to(device)

        batch_size = target_images.size(0)
        num_context = context_images.size(1)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                target_latents = sd_model_pipe.vae.encode(target_images).latent_dist.sample()
                target_latents = (
                    target_latents - sd_model_pipe.vae.config.shift_factor
                ) * sd_model_pipe.vae.config.scaling_factor

                context_flat = context_images.reshape(
                    batch_size * num_context,
                    *target_images.shape[1:],
                )
                context_latents = sd_model_pipe.vae.encode(context_flat).latent_dist.sample()
                context_latents = (
                    context_latents - sd_model_pipe.vae.config.shift_factor
                ) * sd_model_pipe.vae.config.scaling_factor

            target_latents = target_latents.detach()
            context_latents = context_latents.detach().reshape(
                batch_size, num_context, *target_latents.shape[1:]
            )

        loss_item = 0.0

        opt.zero_grad()

        # the micro batch size controls gradient accumulation
        for mb_idx in range((local_batch_size - 1) // config.micro_batch_size + 1):
            mb_st = mb_idx * config.micro_batch_size
            mb_ed = min((mb_idx + 1) * config.micro_batch_size, local_batch_size)
            last_mb = mb_ed == local_batch_size

            target_latents_mb = target_latents[mb_st:mb_ed]
            context_latents_mb = context_latents[mb_st:mb_ed]
            caption_ids_mb = caption_ids[mb_st:mb_ed]
            caption_label_mask_mb = caption_label_mask[mb_st:mb_ed]
            if caption_label_mask_mb.ndim == 1:
                caption_label_mask_mb = caption_label_mask_mb.unsqueeze(0)

            if caption_attention is not None:
                caption_attention_mb = caption_attention[mb_st:mb_ed]
            else:
                caption_attention_mb = None

            preference_ids_mb = preference_ids[mb_st:mb_ed]
            if preference_attention is not None:
                preference_attention_mb = preference_attention[mb_st:mb_ed]
            else:
                preference_attention_mb = None

            sentence_ids_mb = sentence_ids[mb_st:mb_ed]
            if sentence_attention is not None:
                sentence_attention_mb = sentence_attention[mb_st:mb_ed]
            else:
                sentence_attention_mb = None

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                with torch.inference_mode():
                    preference_emb_mb = sd_model_pipe.text_encoder_3(
                        preference_ids_mb, attention_mask=preference_attention_mb
                    )[0].detach()
                    sentence_emb_mb = sd_model_pipe.text_encoder_3(
                        sentence_ids_mb, attention_mask=sentence_attention_mb
                    )[0].detach()
                    caption_emb_mb = sd_model_pipe.text_encoder_3(
                        caption_ids_mb, attention_mask=caption_attention_mb
                    )[0].detach()

                conditioning_embeds = torch.cat(
                    [
                        preference_emb_mb.to(model.dtype),
                        sentence_emb_mb.to(model.dtype),
                    ],
                    dim=1,
                )

                avg_context_latents_mb = context_latents_mb.mean(dim=1).to(target_latents_mb.dtype)

                # caption loss
                caption_diffusion_loss = text_diffusion_loss_module.compute_loss(
                    model,
                    caption_ids_mb, 
                    # target_latents_mb,
                    avg_context_latents_mb,
                    None,
                    use_dummy_loss=False,
                    disable_t5_grad=False,
                    label_mask=caption_label_mask_mb,
                    conditioning_embeds=conditioning_embeds,
                )

                target_emb_mb, context_proj_mb, matching_lambda = apply_stage2_image_conditioning(
                    model, target_latents_mb, context_latents_mb
                )

                # 50% random drop matching condition
                context_dropout_mask = (torch.rand(
                    target_emb_mb.shape[0], device=target_emb_mb.device
                ) >= 0.5).float().view(-1, 1, 1, 1).to(context_proj_mb.dtype)
                context_proj_for_loss = context_proj_mb * context_dropout_mask

                # 50% random drop preference information
                drop_preference = torch.rand((), device=preference_emb_mb.device) < 0.5
                if drop_preference:
                    image_text_embeds = sentence_emb_mb.to(model.dtype)
                else:
                    image_text_embeds = torch.cat(
                        [
                            preference_emb_mb,
                            sentence_emb_mb,
                            # caption_emb_mb,
                        ],
                        dim=1,
                    ).to(model.dtype)

                image_diffusion_loss = image_diffusion_loss_module.compute_loss(
                    model,
                    image_text_embeds,
                    target_emb_mb,
                    context_proj=context_proj_for_loss,
                    lamda=matching_lambda,
                    gt=target_latents_mb,
                )

                loss = image_diffusion_loss + \
                       caption_diffusion_loss * config.training.caption_training_weight

            if args.training == "multiple":
                with model.no_sync() if args.data_parallel in ['h_sdp', 'h_fsdp', "sdp",
                                                           "fsdp"] and not last_mb else contextlib.nullcontext():
                    loss.backward()
            if args.training == "single":
                loss.backward()

            running_caption_loss += caption_diffusion_loss.item()
            running_image_loss += image_diffusion_loss.item()

            loss_item += loss.item()

        grad_norm = calculate_l2_grad_norm(model, model_parallel_dim_dict)
        if grad_norm > config.grad_clip:
            scale_grad(model, config.grad_clip / grad_norm)

        opt.step()
        scheduler.step()

        step += 1
        if step >= config.ema_steps:
            update_ema(model_ema, model)
        elif step == config.ema_steps - 1:
            logger.info(f"Initalized EMA tracking with current model weight")
            update_ema(model_ema, model, decay=0.0, copy=True)

        # Log loss values:
        running_loss += loss_item
        log_steps += 1
        if step % config.log_every == 0:
            gradient_accumulation_steps = (local_batch_size - 1) // config.micro_batch_size + 1
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            secs_per_step = (end_time - start_time) / log_steps
            imgs_per_sec = config.global_batch_size * log_steps / (end_time - start_time)

            # Reduce loss history over all processes:
            # add judgement to single and multiple
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            avg_loss = average_tensor(avg_loss, gradient_accumulation_steps).item()

            avg_caption_loss = torch.tensor(running_caption_loss / log_steps, device=device)
            avg_caption_loss = average_tensor(avg_caption_loss, gradient_accumulation_steps).item()

            avg_image_loss = torch.tensor(running_image_loss / log_steps, device=device)
            avg_image_loss = average_tensor(avg_image_loss, gradient_accumulation_steps).item()

            # # Reduce loss history over all processes:
            # avg_loss = torch.tensor(running_loss / log_steps, device=device)
            # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            # avg_loss = avg_loss.item() / dist.get_world_size() / gradient_accumulation_steps
            #
            # avg_caption_loss = torch.tensor(running_caption_loss / log_steps, device=device)
            # dist.all_reduce(avg_caption_loss, op=dist.ReduceOp.SUM)
            # avg_caption_loss = avg_caption_loss.item() / dist.get_world_size() / gradient_accumulation_steps
            #
            # avg_image_loss = torch.tensor(running_image_loss / log_steps, device=device)
            # dist.all_reduce(avg_image_loss, op=dist.ReduceOp.SUM)
            # avg_image_loss = avg_image_loss.item() / dist.get_world_size() / gradient_accumulation_steps
            logger.info(
                f"(Step={step + 1:07d}) "
                f"Image Loss: {avg_image_loss:.4f}, "
                f"Caption Loss: {avg_caption_loss:.4f}, "
                f"Train Secs/Step: {secs_per_step:.2f}, "
                f"Train Imgs/Sec: {imgs_per_sec:.2f}, "
                f"Train grad norm: {grad_norm:.2f},"
            )
            # call wandb log on main rank
            if rank == 0:
                # collect text loss, image loss, grad norm, and lr
                wandb.log({"train_loss": avg_loss,
                           "train_caption_loss": avg_caption_loss,
                           "train_image_loss": avg_image_loss,
                           "lr": opt.param_groups[0]["lr"],
                           "grad_norm": grad_norm,
                           }, step=step)

            # Reset monitoring variables:
            running_loss = 0
            running_image_loss = 0
            running_caption_loss = 0

            log_steps = 0

            start_time = time()

        # Save DiT checkpoint:
        if step % config.ckpt_every == 0 or step == config.max_steps:

            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            # save model parameter
            if dist.is_available() and dist.is_initialized():
                # multiple
                with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    consolidated_model_state_dict = model.state_dict()
                    if fs_init.get_data_parallel_rank() == 0:
                        consolidated_fn = (
                            "consolidated."
                            f"{fs_init.get_model_parallel_rank():02d}-of-"
                            f"{fs_init.get_model_parallel_world_size():02d}"
                            ".pth"
                        )
                        torch.save(
                            consolidated_model_state_dict,
                            os.path.join(checkpoint_path, consolidated_fn),
                        )
                dist.barrier()
                del consolidated_model_state_dict
                logger.info(f"Saved consolidated to {checkpoint_path}.")
            else:
                # single
                torch.save(model.state_dict(), os.path.join(checkpoint_path, "consolidated.00-of-01.pth"))
                logger.info(f"Saved model (single GPU) to {checkpoint_path}.")


            # save EMA model
            if step > config.ema_steps:
                # multiple
                if dist.is_available() and dist.is_initialized():
                    with FSDP.state_dict_type(
                            model_ema,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    ):
                        consolidated_ema_state_dict = model_ema.state_dict()
                        if fs_init.get_data_parallel_rank() == 0:
                            consolidated_ema_fn = (
                                "consolidated_ema."
                                f"{fs_init.get_model_parallel_rank():02d}-of-"
                                f"{fs_init.get_model_parallel_world_size():02d}"
                                ".pth"
                            )
                            torch.save(
                                consolidated_ema_state_dict,
                                os.path.join(checkpoint_path, consolidated_ema_fn),
                            )
                    dist.barrier()
                    del consolidated_ema_state_dict
                    logger.info(f"Saved consolidated_ema to {checkpoint_path}.")
                # single
                torch.save(model_ema.state_dict(), os.path.join(checkpoint_path, "consolidated_ema.00-of-01.pth"))
                logger.info(f"Saved EMA model (single GPU) to {checkpoint_path}.")


            # save optimizer
            if dist.is_available() and dist.is_initialized():
                with FSDP.state_dict_type(
                        model_ema,
                        StateDictType.LOCAL_STATE_DICT,
                ):
                    opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                    torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
                dist.barrier()
                logger.info(f"Saved optimizer to {checkpoint_path}.")
            else:
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, "optimizer.00000-of-00001.pth"))
                logger.info(f"Saved optimizer (single GPU) to {checkpoint_path}.")


            # just save scheduler on the main rank
            if rank == 0:
                scheduler_state_fn = f"scheduler.pth"
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, scheduler_state_fn))
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            logger.info(f"Saved scheduler to {checkpoint_path}.")

            # save model parameter
            # if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            if dist.is_available() and dist.is_initialized():
                if dist.get_rank() == 0:
                    torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                    with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                        f.write(str(step + 1))
                    logger.info(f"Saved training arguments to {checkpoint_path}.")
            else:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    f.write(str(step + 1))
                logger.info(f"Saved training arguments to {checkpoint_path}.")
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        # save model
        if step % 2500 == 0:
            model_dir = os.path.join(args.results_dir, args.model_dir)
            model_save_dir = os.path.join(model_dir, f"{step + 1:07d}")
            os.makedirs(model_save_dir, exist_ok=True)

            model.save_pretrained(os.path.join(model_save_dir, "transformer"))
            sd_model_pipe.text_encoder_3.save_pretrained(os.path.join(model_save_dir, "text_encoder"))
            sd_model_pipe.tokenizer_3.save_pretrained(os.path.join(model_save_dir, "tokenizer"))
            sd_model_pipe.vae.save_pretrained(os.path.join(model_save_dir, "vae"))
            sd_model_pipe.scheduler.save_pretrained(os.path.join(model_save_dir, "scheduler"))
            save_model_index_json_file(model_save_dir)
            mask_emb = sd_model_pipe.text_encoder_3.shared.weight.data[MASK_TOKEN_IDS].clone().cpu()
            torch.save(mask_emb, os.path.join(model_save_dir, "mask_emb.pt"))

    model.eval()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="{your_path}/configs/stage2_config_ifashion.py")
    parser.add_argument("--results_dir", type=str, default="{your_path}/output/ifashion")
    parser.add_argument("--checkpoint_dir", type=str, default="stage2_checkpoints")
    parser.add_argument("--model_dir", type=str, default="stage2_models")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")

    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["h_sdp", "h_fsdp", "sdp", "fsdp"], default="h_sdp")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"], default='bf16')

    parser.add_argument("--global_seed", type=int, default=971011)
    parser.add_argument("--training", type=str, choices=["single", "multiple"], default="single")
    parser.add_argument("--device", default=3)

    args = parser.parse_args()

    main(args)
