#!/usr/bin/env python3
"""
Stage-3 text-only training using augmented captions.

Differences from stage2:
- Uses augmented dataset: {your_path}/processed_info/stage3_data_augmentation/data_augmentation.npy
- Trains only the text branch (no image loss)
- Loads pretrained weights from stage2 (see {your_path}/configs/stage3_config.py)
"""
import argparse
import os
import random
from datetime import datetime
from time import time

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_constant_schedule_with_warmup
from mmcv import Config
from safetensors.torch import load_file
from diffusers import StableDiffusion3Pipeline as SDPipe

from sd3_modules.sd3_model import SD3JointModelFlexible
from sd3_modules.stage3_sd3_loss_utils import TextMaskedDiffusionLoss
from custom_dataset.stage3_data_augmentation import Stage3DataAugmentation

torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="{your_path}/configs/stage3_config.py",
        help="Path to config file.",
    )
    parser.add_argument(
        "--results-dir",
        default="{your_path}/output/ifashion/stage3_checkpoints",
        help="Directory to save checkpoints/logs.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="stage3_models",
        help="Subdirectory under results-dir to store full model exports.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--save-every",
        type=int,
        default=2000,
        help="Save every N steps.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "tf32", "fp16", "bf16"],
        default="bf16",
        help="Computation precision for forward pass.",
    )
    parser.add_argument(
        "--grad-precision",
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Autocast dtype for forward/backward.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    ckpt_dir,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(ckpt_dir, "ckpt.pt"),
    )

def save_model_export(model, sd_model_pipe, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    model.save_pretrained(os.path.join(export_dir, "transformer"))
    if sd_model_pipe is not None:
        sd_model_pipe.text_encoder_3.save_pretrained(os.path.join(export_dir, "text_encoder"))
        sd_model_pipe.tokenizer_3.save_pretrained(os.path.join(export_dir, "tokenizer"))


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    set_seed(args.seed if args.seed is not None else config.get("seed", 1234))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # precision setup
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model_dtype = torch.float32
    elif args.precision == "fp16":
        torch.backends.cuda.matmul.allow_tf32 = False
        model_dtype = torch.float16
    elif args.precision == "bf16":
        torch.backends.cuda.matmul.allow_tf32 = False
        model_dtype = torch.bfloat16
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        model_dtype = torch.float32

    grad_dtype = None
    if args.grad_precision == "fp16":
        grad_dtype = torch.float16
    elif args.grad_precision == "bf16":
        grad_dtype = torch.bfloat16

    # load sd3 pipe (only text encoder + tokenizer used)
    sd_model_pipe = SDPipe.from_pretrained(
        config.sd3_pipeline_load_from,
        torch_dtype=model_dtype,
    ).to(device)
    sd_model_pipe.text_encoder_3.to(model_dtype)
    sd_model_pipe.tokenizer_3.pad_token = sd_model_pipe.tokenizer_3.eos_token
    orig_sd_transformer = sd_model_pipe.transformer

    model = SD3JointModelFlexible(
        len(sd_model_pipe.tokenizer_3),
        **orig_sd_transformer.config,
    ).to(device).to(model_dtype).train()

    # freeze encoders
    sd_model_pipe.transformer = None
    sd_model_pipe.text_encoder_1 = None
    sd_model_pipe.text_encoder_2 = None
    sd_model_pipe.text_encoder_3.requires_grad_(False)
    sd_model_pipe.text_encoder_3.eval()

    if config.resume_from_legacy:
        legacy_state_dict = load_file(config.resume_from_legacy, device="cpu")
        missing, unexpected = model.load_state_dict(legacy_state_dict, strict=False)
        print(f"[resume] loaded from {config.resume_from_legacy}")
        if missing:
            print(f"[resume] missing keys: {missing}")
        if unexpected:
            print(f"[resume] unexpected keys: {unexpected}")

    train_dataset = Stage3DataAugmentation(
        tokenizer=sd_model_pipe.tokenizer_3,
        **config.stage3_data_config,
    )
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.global_batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(train_loader)

    text_loss = TextMaskedDiffusionLoss(config, model_pipe=sd_model_pipe)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=config.num_warmup_steps)

    step = 0
    running_loss = 0.0
    log_every = config.log_every
    max_steps = config.max_steps
    start_time = time()

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        caption_tokens = batch["caption_tokens"]
        # caption tensors shape: [batch, num_aug, seq]
        caption_ids = caption_tokens["input_ids"].to(device)
        caption_label_mask = caption_tokens["label_mask"].to(device)
        caption_attention = caption_tokens.get("attention_mask")
        if caption_attention is not None:
            caption_attention = caption_attention.to(device)

        preference_tokens = batch["preference_tokens"]
        preference_ids = preference_tokens["input_ids"].to(device)
        preference_attention = preference_tokens.get("attention_mask")
        if preference_attention is not None:
            preference_attention = preference_attention.to(device)

        sentence_tokens = batch["sentence_tokens"]
        sentence_ids = sentence_tokens["input_ids"].to(device)
        sentence_attention = sentence_tokens.get("attention_mask")
        if sentence_attention is not None:
            sentence_attention = sentence_attention.to(device)

        # flatten batch and augmentation dims
        bsz, num_aug, seq_len = caption_ids.shape
        flat = lambda x: x.view(-1, seq_len) if x is not None else None
        caption_ids = flat(caption_ids)
        caption_label_mask = flat(caption_label_mask)
        caption_attention = flat(caption_attention)

        flat_pref = lambda x: x.view(-1, x.shape[-1]) if x is not None else None
        preference_ids = flat_pref(preference_ids)
        preference_attention = flat_pref(preference_attention)
        sentence_ids = flat_pref(sentence_ids)
        sentence_attention = flat_pref(sentence_attention)

        if grad_dtype is not None and device.type == "cuda":
            autocast_ctx = torch.cuda.amp.autocast(dtype=grad_dtype)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            with torch.no_grad():
                preference_emb = sd_model_pipe.text_encoder_3(
                    preference_ids, attention_mask=preference_attention
                )[0]
                sentence_emb = sd_model_pipe.text_encoder_3(
                    sentence_ids, attention_mask=sentence_attention
                )[0]

            conditioning_embeds = torch.cat(
                [preference_emb.to(model.dtype), sentence_emb.to(model.dtype)],
                dim=1,
            )

            loss = text_loss.compute_loss(
                model,
                caption_ids,
                image_condition=None,
                attention_mask=caption_attention,
                use_dummy_loss=False,
                disable_t5_grad=False,
                label_mask=caption_label_mask,
                conditioning_embeds=conditioning_embeds,
            )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1

        if step % log_every == 0:
            avg_loss = running_loss / log_every
            elapsed = time() - start_time
            print(
                f"[step {step}] loss={avg_loss:.4f} elapsed={elapsed/60:.1f}m",
                flush=True,
            )
            running_loss = 0.0
            start_time = time()

        if step % args.save_every == 0 or step == max_steps:
            ckpt_dir = os.path.join(args.results_dir, f"{step:07d}")
            model_export_dir = os.path.join(args.results_dir, args.model_dir, f"{step:07d}")
            save_checkpoint(model, opt, scheduler, step, ckpt_dir)
            save_model_export(model, sd_model_pipe, model_export_dir)
            print(f"[save] checkpoint -> {ckpt_dir}", flush=True)
            print(f"[save] model -> {model_export_dir}", flush=True)


if __name__ == "__main__":
    main()
