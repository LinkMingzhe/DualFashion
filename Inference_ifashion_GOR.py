import argparse
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
from safetensors.torch import load_file

from custom_dataset.stage2_ifashion_data import _normalize_value, _safe_json_dumps
from sd3_modules.dual_diff_pipeline import DualDiffSD3Pipeline

MASK_TOKEN_ID = 32099
MATCHING_LAMBDA = 0.1


def apply_stage2_image_conditioning(
    model: torch.nn.Module, target_latents: torch.Tensor, context_latents: torch.Tensor
) -> torch.Tensor:
    gate = torch.sigmoid(model.stage2_context_gate).to(target_latents.dtype)
    target_emb = target_latents + model.stage2_target_role_embedding.to(target_latents.dtype)

    context_summary = context_latents.mean(dim=1)
    context_emb = context_summary + model.stage2_context_role_embedding.to(context_summary.dtype)
    context_proj = model.stage2_context_projector(context_emb)

    return target_emb + gate * context_proj


def compute_stage2_context_offset(
    model: torch.nn.Module, context_latents: torch.Tensor
) -> torch.Tensor:
    target_template = torch.zeros_like(context_latents[:, 0])
    return apply_stage2_image_conditioning(model, target_template, context_latents)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 GOR inference for iFashion.")
    parser.add_argument(
        "--models_root",
        type=str,
        default="output/ifashion/stage2_models_v5",
        help="Directory containing trained stage-2 checkpoints.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output/QualitativeResults/stage2_inference_GOR",
        help="Directory to store sampling results.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/mnt/raid1/mzyu/dataset/ifashion/semantic_category",
        help="Root directory for raw fashion item images.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sequential outfit samples to draw per (uid, oid) pair.",
    )
    parser.add_argument(
        "--max_inference",
        type=int,
        default=None,
        help="Optional upper bound on sequential samples drawn per (uid, oid) pair.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Upper bound on the number of GOR test entries to evaluate per model.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps during sampling.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame",
        help="Negative prompt applied when classifier-free guidance is enabled.",
    )
    parser.add_argument(
        "--text-guidance-scale",
        type=float,
        default=8.0,
        help="Sentence - Classifier-free guidance scale applied to the sentence-only prompt.",
    )
    parser.add_argument(
        "--context-guidance-scale",
        type=float,
        default=7.0,
        help="Matching - Additional guidance scale applied to context (mutual) conditioning.",
    )
    parser.add_argument(
        "--preference-guidance-scale",
        type=float,
        default=8.0,
        help="Preference - Classifier-free guidance scale used during image sampling.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token length per textual condition segment.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Render resolution for generated images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--caption_steps",
        type=int,
        default=64,
        help="Diffusion steps used for caption sampling.",
    )
    return parser.parse_args()


def load_numpy_dict(path: str) -> Dict:
    return np.load(path, allow_pickle=True).item()


def build_preference_text(
    preferences: Dict,
    uid: int,
    category_id: int,
    mask_keys: List[str],
) -> str:
    default_payload = {key: "Unknown" for key in mask_keys}
    user_pref = preferences.get(int(uid), {})
    cate_pref = user_pref.get(int(category_id))
    if not cate_pref:
        return _safe_json_dumps(default_payload)

    freq = cate_pref.get("Frequency") or cate_pref.get("frequency")
    if not isinstance(freq, dict):
        return _safe_json_dumps(default_payload)

    payload = {key: _normalize_value(freq.get(key, "Unknown")) for key in mask_keys}
    return _safe_json_dumps(payload)


def build_caption_text(item_info: Dict, item_id: int, mask_keys: List[str]) -> str:
    item_payload = item_info.get(int(item_id))
    if not item_payload:
        return _safe_json_dumps({key: "Unknown" for key in mask_keys})

    elements = item_payload.get("elements", {})
    payload = {key: _normalize_value(elements.get(key, "Unknown")) for key in mask_keys}
    return _safe_json_dumps(payload)


def _find_value_spans(text: str, mask_keys: List[str]):
    spans = []
    for key in mask_keys:
        pattern = rf'"{key}"\s*:\s*"([^"]*)"'
        for match in re.finditer(pattern, text):
            spans.append((match.start(1), match.end(1)))
    return spans


def _build_label_mask(token_info, raw_text: str, mask_keys: List[str]) -> torch.Tensor:
    if "offset_mapping" not in token_info:
        return torch.zeros_like(token_info["input_ids"])
    offsets = token_info["offset_mapping"][0]
    mask = torch.zeros_like(token_info["input_ids"])
    spans = _find_value_spans(raw_text, mask_keys)
    for start, end in spans:
        token_hits = (offsets[:, 0] < end) & (offsets[:, 1] > start)
        if token_hits.any():
            mask[0, token_hits] = 1
    if "attention_mask" in token_info:
        mask = mask * token_info["attention_mask"]
    return mask


def prepare_masked_caption_inputs(
    tokenizer,
    caption_text: str,
    mask_keys: List[str],
    max_length: int,
    device: torch.device,
):
    token_info = tokenizer(
        caption_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = token_info["input_ids"].to(device)
    attention_mask = token_info["attention_mask"].to(device)
    label_mask = _build_label_mask(token_info, caption_text, mask_keys).to(device)

    masked_input_ids = input_ids.clone()
    masked_positions = label_mask.bool()
    masked_input_ids[masked_positions] = MASK_TOKEN_ID

    prompt_mask = attention_mask.bool() & (~masked_positions)

    token_info.pop("offset_mapping", None)
    return {
        "input_ids": input_ids,
        "masked_input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "label_mask": label_mask,
        "prompt_mask": prompt_mask,
    }


@torch.no_grad()
def encode_text(
    tokenizer,
    text_encoder,
    text: str,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    use_autocast = device.type == "cuda" and text_encoder.dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type=device.type, enabled=use_autocast, dtype=text_encoder.dtype):
        embeds = text_encoder(input_ids, attention_mask=attention_mask)[0]
    return embeds


def prepend_pad_segment(embeds: torch.Tensor, pad_segment: torch.Tensor, target_length: int) -> torch.Tensor:
    seq_len = embeds.shape[1]
    if seq_len >= target_length:
        return embeds
    pad_len = target_length - seq_len
    segment_len = pad_segment.shape[1]
    if segment_len <= 0:
        raise ValueError("pad_segment must have positive sequence length.")
    pad_segments: List[torch.Tensor] = []
    while pad_len > 0:
        take = min(segment_len, pad_len)
        if take == segment_len:
            pad_segments.append(pad_segment)
        else:
            pad_segments.append(pad_segment[:, :take])
        pad_len -= take
    padding = torch.cat(pad_segments, dim=1)
    return torch.cat([padding, embeds], dim=1).contiguous()


def enable_text_sampler_conditioning(text_sampler) -> None:
    if getattr(text_sampler, "_conditioning_patched", False):
        return

    text_sampler.conditioning_embeds = None
    original_clear = text_sampler.clear_condition

    def register_condition(y, x, x_mask, conditioning_embeds=None):
        device = y.device
        text_sampler.y = y
        text_sampler.x = x.to(device)
        text_sampler.x_mask = x_mask.to(device)
        text_sampler.conditioning_embeds = conditioning_embeds

    def clear_condition():
        original_clear()
        text_sampler.conditioning_embeds = None

    def forward(xt):
        cond = text_sampler.y
        mask_index = text_sampler.mask_index
        text_hidden_states = text_sampler.prepare_text_inputs(xt, attention_mask=None)
        seq_len = xt.shape[1]
        if text_sampler.conditioning_embeds is not None:
            cond_embeds = text_sampler.conditioning_embeds.to(
                text_hidden_states.device, dtype=text_hidden_states.dtype
            )
            text_hidden_states = torch.cat([text_hidden_states, cond_embeds], dim=1)

        with torch.cuda.amp.autocast(enabled=True):
            logits = text_sampler.model(
                hidden_states=cond,
                timestep=torch.zeros(xt.shape[0], device=xt.device),
                encoder_hidden_states=text_hidden_states.detach(),
                pooled_projections=None,
            )[1]

        logits = logits.float()
        if logits.shape[1] != seq_len:
            logits = logits[:, :seq_len, :]
        logits[:, :, mask_index] += text_sampler.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = (xt != mask_index)
        logits[unmasked_indices] = text_sampler.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _sample_prior(*batch_dims):
        if text_sampler.x is None or text_sampler.x_mask is None:
            raise RuntimeError("Text sampler conditions are not registered.")
        target_device = text_sampler.y.device if text_sampler.y is not None else text_sampler.x.device
        if text_sampler.x.device != target_device:
            text_sampler.x = text_sampler.x.to(target_device)
        if text_sampler.x_mask.device != target_device:
            text_sampler.x_mask = text_sampler.x_mask.to(target_device)
        masked = text_sampler.mask_index * torch.ones(
            *batch_dims, dtype=torch.int64, device=target_device
        )
        return torch.where(text_sampler.x_mask, text_sampler.x, masked)

    text_sampler.register_condition = register_condition
    text_sampler.clear_condition = clear_condition
    text_sampler.forward = forward
    text_sampler._sample_prior = _sample_prior
    text_sampler._conditioning_patched = True


def decode_and_save_image(
    pipe: DualDiffSD3Pipeline,
    latent: torch.Tensor,
    output_path: str,
) -> None:
    use_autocast = latent.device.type == "cuda"
    with torch.autocast(device_type=latent.device.type, enabled=use_autocast, dtype=pipe.transformer.dtype):
        latents = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
    image = (image.clamp(-1, 1) + 1) / 2
    save_image(image, output_path)


def ensure_stage2_modules(transformer: torch.nn.Module, latent_channels: int) -> None:
    ref_param = next(transformer.parameters())
    device = ref_param.device
    dtype = ref_param.dtype

    if not hasattr(transformer, "stage2_target_role_embedding"):
        transformer.stage2_target_role_embedding = nn.Parameter(
            torch.randn(1, latent_channels, 1, 1, device=device, dtype=dtype) * 0.01
        )

    if not hasattr(transformer, "stage2_context_role_embedding"):
        transformer.stage2_context_role_embedding = nn.Parameter(
            torch.randn(1, latent_channels, 1, 1, device=device, dtype=dtype) * 0.01
        )

    if not hasattr(transformer, "stage2_context_projector"):
        projector = nn.Conv2d(latent_channels, latent_channels, kernel_size=1).to(device=device, dtype=dtype)
        nn.init.zeros_(projector.weight)
        nn.init.zeros_(projector.bias)
        transformer.stage2_context_projector = projector

    if not hasattr(transformer, "stage2_context_gate"):
        transformer.stage2_context_gate = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))


def restore_stage2_parameters(transformer: torch.nn.Module, model_dir: str) -> None:
    transformer_path = os.path.join(model_dir, "transformer", "diffusion_pytorch_model.safetensors")
    if not os.path.exists(transformer_path):
        return

    target_device = next(transformer.parameters()).device
    load_device = "cpu" if target_device.type != "cuda" or not torch.cuda.is_available() else "cpu"
    weights = load_file(transformer_path, device=load_device)
    for key in [
        "stage2_target_role_embedding",
        "stage2_context_role_embedding",
        "stage2_context_projector.weight",
        "stage2_context_projector.bias",
        "stage2_context_gate",
    ]:
        module = transformer
        if "." in key:
            attr, sub = key.split(".")
            module = getattr(transformer, attr, None)
            if module is None:
                continue
            target_tensor = getattr(module, sub)
        else:
            target_tensor = getattr(transformer, key, None)
        if target_tensor is None or key not in weights:
            continue
        target_tensor.data.copy_(weights[key].to(target_tensor.device, dtype=target_tensor.dtype))


def save_ground_truth_images(
    outfit_item_ids: List[int],
    image_root: str,
    image_paths: np.ndarray,
    sample_dir: str,
) -> None:
    for item_id in outfit_item_ids:
        if item_id <= 0 or item_id >= len(image_paths):
            continue
        rel_path = image_paths[item_id]
        image_path = os.path.join(image_root, rel_path)
        if not os.path.exists(image_path):
            continue
        with Image.open(image_path).convert("RGB") as img:
            img.save(os.path.join(sample_dir, f"grd_{item_id}.png"))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_root, exist_ok=True)

    # Load data resources
    test_grd = load_numpy_dict("/mnt/raid1/mzyu/dataset/ifashion/test_grd_dict.npy")
    id_to_category = load_numpy_dict("/mnt/raid1/mzyu/dataset/ifashion/id_cate_dict.npy")
    test_preferences = load_numpy_dict("processed_info/stage2_ifashion_test/test_preference.npy")
    test_item_info = load_numpy_dict("processed_info/stage2_ifashion_test/item_info.npy")
    test_image_paths = np.load("processed_info/stage2_ifashion_test/all_item_image_paths.npy", allow_pickle=True)

    mask_keys = ["Color", "Material", "Design features", "Clothing Fashion Style"]

    model_names = sorted(
        [name for name in os.listdir(args.models_root) if os.path.isdir(os.path.join(args.models_root, name))]
    )
    if not model_names:
        raise ValueError(f"No models found under {args.models_root}")

    total_samples = len(test_grd["uids"])
    if args.max_samples is not None and args.max_samples > 0:
        total_samples = min(total_samples, args.max_samples)

    for model_name in model_names:
        # best ckpt
        if model_name != '0057501':
            continue
        model_path = os.path.join(args.models_root, model_name)
        output_model_dir = os.path.join(args.output_root, model_name)
        if os.path.exists(output_model_dir) and os.listdir(output_model_dir):
            print(f"Skipping {model_name} because outputs already exist at {output_model_dir}")
            continue
        os.makedirs(output_model_dir, exist_ok=True)

        pipe = DualDiffSD3Pipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        pipe = pipe.to(device)
        pipe.transformer.eval()
        ensure_stage2_modules(pipe.transformer, pipe.vae.config.latent_channels)
        restore_stage2_parameters(pipe.transformer, model_path)
        enable_text_sampler_conditioning(pipe.text_sampler)

        latent_channels = pipe.vae.config.latent_channels
        latent_height = args.resolution // pipe.vae_scale_factor
        latent_width = args.resolution // pipe.vae_scale_factor
        zero_context = torch.zeros(
            1,
            1,
            latent_channels,
            latent_height,
            latent_width,
            device=device,
            dtype=pipe.transformer.dtype,
        )

        scheduler = pipe.scheduler
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.set_sampling_mode("t2i")

        text_guidance_scale = float(args.text_guidance_scale) if args.text_guidance_scale is not None else 0.0
        context_guidance_scale = float(args.context_guidance_scale) if args.context_guidance_scale is not None else 0.0
        preference_guidance_scale = (
            float(args.preference_guidance_scale) if args.preference_guidance_scale is not None else 0.0
        )
        do_classifier_free_guidance = any(
            scale != 0.0 for scale in (text_guidance_scale, context_guidance_scale, preference_guidance_scale)
        )

        negative_prompt_segment = None
        if do_classifier_free_guidance:
            negative_prompt_text = args.negative_prompt or ""
            negative_prompt_segment = encode_text(
                pipe.tokenizer, pipe.text_encoder, negative_prompt_text, args.max_length, device
            ).to(device=device, dtype=pipe.transformer.dtype)


        max_repeats = args.num_samples if args.max_inference is None else min(args.num_samples, args.max_inference)

        for idx in tqdm(range(total_samples), desc=f"Model {model_name}", unit="sample"):
            uid = int(test_grd["uids"][idx])
            oid = int(test_grd["oids"][idx])
            outfit_item_ids = [int(item) for item in test_grd["outfits"][idx]]
            category_ids = [int(cat) for cat in test_grd["category"][idx]]

            folder_name = f"{uid}_{oid}"
            sample_dir = os.path.join(output_model_dir, folder_name)
            os.makedirs(sample_dir, exist_ok=True)

            save_ground_truth_images(outfit_item_ids, args.image_root, test_image_paths, sample_dir)

            prompt_configs: List[Dict[str, torch.Tensor]] = []
            for item_id, category_id in zip(outfit_item_ids, category_ids):
                preference_text = build_preference_text(test_preferences, uid, category_id, mask_keys)
                category_name = id_to_category.get(category_id, "item")
                sentence_text = f"Recommend a fashion {category_name} item, on white background."
                caption_text = build_caption_text(test_item_info, item_id, mask_keys)

                preference_emb = encode_text(
                    pipe.tokenizer, pipe.text_encoder, preference_text, args.max_length, device
                ).to(device=device, dtype=pipe.transformer.dtype)
                sentence_emb = encode_text(
                    pipe.tokenizer, pipe.text_encoder, sentence_text, args.max_length, device
                ).to(device=device, dtype=pipe.transformer.dtype)

                caption_inputs = prepare_masked_caption_inputs(
                    pipe.tokenizer,
                    caption_text,
                    mask_keys,
                    args.max_length,
                    device,
                )
                with torch.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                    dtype=pipe.transformer.dtype,
                ):
                    masked_caption_emb = pipe.text_encoder(
                        caption_inputs["masked_input_ids"],
                        attention_mask=caption_inputs["attention_mask"],
                    )[0]

                conditioning_embeds = torch.cat(
                    [
                        preference_emb.to(pipe.transformer.dtype),
                        sentence_emb.to(pipe.transformer.dtype),
                        masked_caption_emb.to(pipe.transformer.dtype),
                    ],
                    dim=1,
                ).contiguous()

                text_prompt_segments = [sentence_emb]
                preference_prompt_segments = [preference_emb, sentence_emb]
                text_prompt_embeds_base = torch.cat(text_prompt_segments, dim=1).contiguous()
                preference_prompt_embeds_base = torch.cat(preference_prompt_segments, dim=1).contiguous()

                uncond_prompt_embeds_base = None
                if do_classifier_free_guidance:
                    if negative_prompt_segment is None:
                        raise RuntimeError(
                            "Negative prompt embeddings were not prepared despite guidance being enabled."
                        )
                    uncond_prompt_embeds_base = torch.cat(
                        [negative_prompt_segment] * len(preference_prompt_segments),
                        dim=1,
                    ).contiguous()
                    text_prompt_embeds_base = prepend_pad_segment(
                        text_prompt_embeds_base,
                        negative_prompt_segment,
                        preference_prompt_embeds_base.shape[1],
                    )

                prompt_configs.append(
                    {
                        "text_prompt": text_prompt_embeds_base,
                        "preference_prompt": preference_prompt_embeds_base,
                        "uncond_prompt": uncond_prompt_embeds_base,
                        "caption_text": caption_text,
                        "caption_inputs": caption_inputs,
                        "conditioning_embeds": conditioning_embeds,
                        "preference_text": preference_text,
                        "sentence_text": sentence_text,
                    }
                )

            caption_entries: List[Dict[str, str]] = []

            for sample_idx in range(max_repeats):
                generator = torch.Generator(device=device)
                generator.manual_seed(torch.randint(0, 10_000_000, (1,)).item())

                if len(outfit_item_ids) > 1:
                    cpu_generator = torch.Generator(device="cpu")
                    cpu_generator.manual_seed(generator.initial_seed())
                    order = torch.randperm(len(outfit_item_ids), generator=cpu_generator).tolist()
                    ordered_item_ids = [outfit_item_ids[i] for i in order]
                    ordered_prompt_configs = [prompt_configs[i] for i in order]
                else:
                    ordered_item_ids = outfit_item_ids
                    ordered_prompt_configs = prompt_configs

                previous_latents: List[torch.Tensor] = []

                for item_id, prompt_config in zip(ordered_item_ids, ordered_prompt_configs):
                    text_prompt_base = prompt_config["text_prompt"]
                    preference_prompt_base = prompt_config["preference_prompt"]
                    uncond_prompt_base = prompt_config["uncond_prompt"]

                    if not previous_latents:
                        context_latents = zero_context
                    else:
                        context_latents = torch.stack(previous_latents, dim=1)

                    context_offset = compute_stage2_context_offset(pipe.transformer, context_latents)

                    latent_state = pipe.prepare_latents(
                        batch_size=1,
                        num_channels_latents=latent_channels,
                        height=args.resolution,
                        width=args.resolution,
                        dtype=pipe.transformer.dtype,
                        device=device,
                        generator=generator,
                    )

                    with torch.no_grad():
                        scheduler.set_timesteps(args.num_inference_steps, device=device)
                        for t in scheduler.timesteps:
                            if do_classifier_free_guidance:
                                if uncond_prompt_base is None:
                                    raise RuntimeError(
                                        "Negative prompt embeddings were not prepared despite guidance being enabled."
                                    )
                                latent_model_input = torch.cat(
                                    [latent_state, latent_state, latent_state, latent_state],
                                    dim=0,
                                )
                                context_offsets = torch.cat(
                                    [
                                        torch.zeros_like(context_offset),
                                        torch.zeros_like(context_offset),
                                        context_offset,
                                        context_offset,
                                    ],
                                    dim=0,
                                )
                                latent_model_input = latent_model_input + MATCHING_LAMBDA * context_offsets
                                guidance_prompt_embeds = torch.cat(
                                    [
                                        uncond_prompt_base,
                                        text_prompt_base,
                                        text_prompt_base,
                                        preference_prompt_base,
                                    ],
                                    dim=0,
                                ).contiguous()
                                timestep = t.expand(latent_model_input.shape[0]).to(device)
                            else:
                                latent_model_input = latent_state + MATCHING_LAMBDA * context_offset
                                guidance_prompt_embeds = preference_prompt_base
                                timestep = t.expand(latent_state.shape[0]).to(device)

                            use_autocast = device.type == "cuda"
                            with torch.autocast(
                                device_type=device.type, enabled=use_autocast, dtype=pipe.transformer.dtype
                            ):
                                noise_pred = pipe.transformer(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=guidance_prompt_embeds,
                                    pooled_projections=None,
                                    return_dict=False,
                                )[0]

                            if do_classifier_free_guidance:
                                (
                                    noise_pred_uncond,
                                    noise_pred_text,
                                    noise_pred_match,
                                    noise_pred_preference,
                                ) = noise_pred.chunk(4)
                                noise_pred = (
                                    noise_pred_uncond
                                    + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                                    + context_guidance_scale * (noise_pred_match - noise_pred_text)
                                    + preference_guidance_scale * (noise_pred_preference - noise_pred_match)
                                )

                            latent_state = scheduler.step(noise_pred, t, latent_state, return_dict=False)[0]

                    sample_output_path = os.path.join(sample_dir, f"{sample_idx}_{item_id}.png")
                    decode_and_save_image(pipe, latent_state, sample_output_path)

                    caption_inputs = prompt_config["caption_inputs"]
                    conditioning_embeds = prompt_config["conditioning_embeds"]
                    if not previous_latents:
                        caption_condition = zero_context.squeeze(1)
                    else:
                        caption_condition = context_latents.mean(dim=1)

                    pipe.text_sampler.register_condition(
                        caption_condition.detach(),
                        caption_inputs["masked_input_ids"],
                        caption_inputs["prompt_mask"],
                        conditioning_embeds=conditioning_embeds,
                    )
                    pred_tokens = pipe.text_sampler.sample(
                        args.max_length,
                        args.caption_steps,
                        batch_size_per_gpu=1,
                        device=device,
                    )
                    pipe.text_sampler.clear_condition()

                    generated_caption = pipe.tokenizer.decode(pred_tokens[0], skip_special_tokens=True)
                    caption_entries.append(
                        {
                            "sample_idx": str(sample_idx),
                            "item_id": str(item_id),
                            "image_path": os.path.abspath(sample_output_path),
                            "preference_text": prompt_config["preference_text"],
                            "sentence_text": prompt_config["sentence_text"],
                            "caption_template": prompt_config["caption_text"],
                            "generated_caption": generated_caption,
                        }
                    )

                    previous_latents.append(latent_state.detach())
                    if len(previous_latents) > 3:
                        previous_latents = previous_latents[-3:]

            caption_record = {
                "uid": uid,
                "oid": oid,
                "generated_captions": caption_entries,
            }
            with open(os.path.join(sample_dir, "generated_captions.json"), "w", encoding="utf-8") as f:
                json.dump(caption_record, f, ensure_ascii=False, indent=2)

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
