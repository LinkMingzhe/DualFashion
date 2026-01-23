import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

from custom_dataset.stage2_ifashion_data import _normalize_value, _safe_json_dumps
from sd3_modules.stage2_pipeline import DualDiffSD3Pipeline

MASK_TOKEN_ID = 32099

MATCHING_LAMBDA = 0.1


class MatchingMLP(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int = 1024,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden_layer: nn.Linear | None = None
        self.output_layer: nn.Linear | None = None
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.default_hidden_dim = hidden_dim
        self.latent_channels = latent_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.flatten(x)
        if self.hidden_layer is None or self.output_layer is None:
            self._build_layers(flat.shape[-1], device=flat.device, dtype=flat.dtype)
        hidden = self.hidden_layer(flat)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        out = self.output_layer(hidden)
        out = torch.tanh(out)
        return out.view_as(x)

    def _build_layers(self, input_dim: int, device: torch.device, dtype: torch.dtype, hidden_dim: int | None = None) -> None:
        if self.hidden_layer is not None and self.output_layer is not None:
            return
        hidden_dim = hidden_dim or self.default_hidden_dim
        self.hidden_layer = nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype)
        self.output_layer = nn.Linear(hidden_dim, input_dim, device=device, dtype=dtype)
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)


def apply_stage2_image_conditioning(
    model: torch.nn.Module, target_latents: torch.Tensor, context_latents: torch.Tensor
) -> torch.Tensor:
    target_emb = target_latents + model.stage2_target_role_embedding.to(target_latents.dtype)

    context_summary = context_latents.mean(dim=1)
    context_emb = context_summary + model.stage2_context_role_embedding.to(context_summary.dtype)
    context_proj = model.stage2_matching_mlp(context_emb)

    return target_emb, context_proj


def compute_stage2_context_offset(
    model: torch.nn.Module, context_latents: torch.Tensor
) -> torch.Tensor:
    target_template = torch.zeros_like(context_latents[:, 0])
    _, context_proj = apply_stage2_image_conditioning(model, target_template, context_latents)
    return context_proj


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

    sample_size = getattr(transformer.config, "sample_size", 64)

    if hasattr(transformer, "stage2_context_projector"):
        delattr(transformer, "stage2_context_projector")
    if hasattr(transformer, "stage2_context_gate"):
        delattr(transformer, "stage2_context_gate")

    if not hasattr(transformer, "stage2_matching_mlp"):
        transformer.stage2_matching_mlp = MatchingMLP(latent_channels=latent_channels, hidden_dim=1024, dropout_p=0.1)


def restore_stage2_parameters(transformer: torch.nn.Module, model_dir: str) -> None:
    transformer_path = os.path.join(model_dir, "transformer", "diffusion_pytorch_model.safetensors")
    if not os.path.exists(transformer_path):
        return

    target_device = next(transformer.parameters()).device
    target_dtype = next(transformer.parameters()).dtype
    load_device = "cpu"
    weights = load_file(transformer_path, device=load_device)

    for attr_name in (
        "stage2_target_role_embedding",
        "stage2_context_role_embedding",
    ):
        if attr_name not in weights or not hasattr(transformer, attr_name):
            continue
        tensor = getattr(transformer, attr_name)
        tensor.data.copy_(weights[attr_name].to(device=target_device, dtype=tensor.dtype))

    mlp = getattr(transformer, "stage2_matching_mlp", None)
    if isinstance(mlp, MatchingMLP):
        hidden_w = weights.get("stage2_matching_mlp.hidden_layer.weight")
        hidden_b = weights.get("stage2_matching_mlp.hidden_layer.bias")
        output_w = weights.get("stage2_matching_mlp.output_layer.weight")
        output_b = weights.get("stage2_matching_mlp.output_layer.bias")

        if hidden_w is not None and output_w is not None:
            mlp._build_layers(
                input_dim=hidden_w.shape[1],
                hidden_dim=hidden_w.shape[0],
                device=target_device,
                dtype=target_dtype,
            )

        if mlp.hidden_layer is not None and hidden_w is not None:
            mlp.hidden_layer.weight.data.copy_(hidden_w.to(device=target_device, dtype=mlp.hidden_layer.weight.dtype))
        if mlp.hidden_layer is not None and hidden_b is not None:
            mlp.hidden_layer.bias.data.copy_(hidden_b.to(device=target_device, dtype=mlp.hidden_layer.bias.dtype))
        if mlp.output_layer is not None and output_w is not None:
            mlp.output_layer.weight.data.copy_(output_w.to(device=target_device, dtype=mlp.output_layer.weight.dtype))
        if mlp.output_layer is not None and output_b is not None:
            mlp.output_layer.bias.data.copy_(output_b.to(device=target_device, dtype=mlp.output_layer.bias.dtype))


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
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=text_encoder.dtype):
        embeds = text_encoder(input_ids, attention_mask=attention_mask)[0]
    return embeds


def prepend_pad_segment(embeds: torch.Tensor, pad_segment: torch.Tensor, target_length: int) -> torch.Tensor:
    """Prepend pad_segment copies so that embeds reach the desired sequence length."""
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


def load_image_tensor(
    item_id: int,
    image_paths: np.ndarray,
    image_root: str,
    transform: T.Compose,
) -> torch.Tensor:
    if item_id <= 0 or item_id >= len(image_paths):
        return torch.zeros(3, 512, 512)

    rel_path = image_paths[item_id]
    image_path = os.path.join(image_root, rel_path)
    if not os.path.exists(image_path):
        return torch.zeros(3, 512, 512)

    with Image.open(image_path).convert("RGB") as img:
        return transform(img)


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
    if image.dim() == 4 and image.shape[0] == 1:
        image = image[0]
    save_image(image, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 PFITB inference for iFashion with caption generation.")
    parser.add_argument(
        "--models_root",
        type=str,
        default="output/ifashion/stage2_models_v5",
        help="Directory containing trained stage-2 checkpoints.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output/QualitativeResults/test",
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
        default=1,
        help="Number of image samples to draw per (uid, oid) pair.",
    )
    parser.add_argument(
        "--num_caption_samples",
        type=int,
        default=1,
        help="Number of caption samples to draw per (uid, oid) pair (defaults to num_samples).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Upper bound on the number of FITB test entries to evaluate per model.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps during image sampling.",
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
        "--negative-prompt",
        type=str,
        default="low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame",
        help="Negative prompt applied when classifier-free guidance is enabled.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token length per textual condition segment.",
    )
    parser.add_argument(
        "--caption_steps",
        type=int,
        default=64,
        help="Diffusion steps used for caption sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_root, exist_ok=True)
    
    caption_sample_count = args.num_caption_samples if args.num_caption_samples is not None else args.num_samples
    caption_sample_count = max(0, int(caption_sample_count))

    # Load data resources
    test_fitb = load_numpy_dict("/mnt/raid1/mzyu/dataset/ifashion/processed/fitb_test.npy")
    test_grd = load_numpy_dict("/mnt/raid1/mzyu/dataset/ifashion/test_grd_dict.npy")
    id_to_category = load_numpy_dict("/mnt/raid1/mzyu/dataset/ifashion/id_cate_dict.npy")
    test_preferences = load_numpy_dict("processed_info/stage2_ifashion_test/test_preference.npy")
    test_item_info = load_numpy_dict("processed_info/stage2_ifashion_test/item_info.npy")
    test_image_paths = np.load("processed_info/stage2_ifashion_test/all_item_image_paths.npy", allow_pickle=True)

    mask_keys = ["Color", "Material", "Design features", "Clothing Fashion Style"]

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(512),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    # Enumerate models
    model_names = sorted(
        [name for name in os.listdir(args.models_root) if os.path.isdir(os.path.join(args.models_root, name))]
    )
    if not model_names:
        raise ValueError(f"No models found under {args.models_root}")

    total_samples = len(test_fitb["uids"])
    if args.max_samples is not None and args.max_samples > 0:
        total_samples = min(total_samples, args.max_samples)

    for model_name in model_names:
        # best ckpt
        if model_name != '0057501':
            continue
        # if int(model_name) % 10000 != 0:
        #     continue
        model_path = os.path.join(args.models_root, model_name)
        output_model_dir = os.path.join(args.output_root, model_name)
        if os.path.exists(output_model_dir):
            continue
        os.makedirs(output_model_dir, exist_ok=True)

        gen_data = defaultdict(dict)
        grd_data = defaultdict(dict)

        pipe = DualDiffSD3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        pipe = pipe.to(device)
        pipe.transformer.eval()
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.set_sampling_mode("t2i")

        ensure_stage2_modules(pipe.transformer, pipe.vae.config.latent_channels)
        restore_stage2_parameters(pipe.transformer, model_path)

        scheduler = pipe.scheduler

        for idx in tqdm(range(total_samples), desc=f"{model_name} | FITB", unit="sample"):
            uid = int(test_fitb["uids"][idx])
            oid = int(test_fitb["oids"][idx])
            folder_name = f"{uid}_{oid}"
            sample_dir = os.path.join(output_model_dir, folder_name)
            os.makedirs(sample_dir, exist_ok=True)

            outfit_incomplete = test_fitb["outfits"][idx].tolist()
            outfit_full = test_grd["outfits"][idx]
            categories = test_fitb["category"][idx].tolist()

            missing_indices = [i for i, item_id in enumerate(outfit_incomplete) if int(item_id) == 0]
            if not missing_indices:
                continue

            target_index = missing_indices[0]
            target_item_id = int(outfit_full[target_index])
            target_category_id = int(categories[target_index])

            context_item_ids = [int(outfit_full[i]) for i in range(len(outfit_full)) if i != target_index]

            preference_text = build_preference_text(test_preferences, uid, target_category_id, mask_keys)
            category_name = id_to_category.get(target_category_id, "item")
            sentence_text = f"Recommend a fashion {category_name} item, on white background."
            caption_text = build_caption_text(test_item_info, target_item_id, mask_keys)

            preference_guidance_scale = float(args.preference_guidance_scale) if args.preference_guidance_scale is not None else 0.0
            text_guidance_scale = float(args.text_guidance_scale) if args.text_guidance_scale is not None else 0.0
            context_guidance_scale = float(args.context_guidance_scale) if args.context_guidance_scale is not None else 0.0
            do_classifier_free_guidance = any(
                scale != 0.0 for scale in (text_guidance_scale, context_guidance_scale, preference_guidance_scale)
            )

            # Encode text components
            with torch.no_grad():
                preference_emb = encode_text(
                    pipe.tokenizer, pipe.text_encoder, preference_text, args.max_length, device
                ).to(device=device, dtype=pipe.transformer.dtype)
                sentence_emb = encode_text(
                    pipe.tokenizer, pipe.text_encoder, sentence_text, args.max_length, device
                ).to(device=device, dtype=pipe.transformer.dtype)

                text_prompt_segments = [sentence_emb]
                preference_prompt_segments = [preference_emb, sentence_emb]
                text_prompt_embeds_base = torch.cat(text_prompt_segments, dim=1).contiguous()
                preference_prompt_embeds_base = torch.cat(preference_prompt_segments, dim=1).contiguous()
                negative_prompt_text = args.negative_prompt or ""
                negative_segment = encode_text(
                    pipe.tokenizer, pipe.text_encoder, negative_prompt_text, args.max_length, device
                ).to(device=device, dtype=pipe.transformer.dtype)
                uncond_prompt_embeds_base = torch.cat(
                    [negative_segment] * len(preference_prompt_segments), dim=1
                ).contiguous()
                text_prompt_embeds_base = prepend_pad_segment(
                    text_prompt_embeds_base, negative_segment, preference_prompt_embeds_base.shape[1]
                )

            # Prepare context images and latents
            context_images = torch.stack(
                [load_image_tensor(item_id, test_image_paths, args.image_root, transform) for item_id in context_item_ids],
                dim=0,
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=pipe.transformer.dtype):
                    context_flat = context_images.reshape(-1, *context_images.shape[2:])
                    context_latents = pipe.vae.encode(context_flat).latent_dist.sample()
                    context_latents = (
                        context_latents - pipe.vae.config.shift_factor
                    ) * pipe.vae.config.scaling_factor

            context_latents = context_latents.reshape(1, context_images.size(1), *context_latents.shape[1:])
            context_latents = context_latents.to(device=device, dtype=pipe.transformer.dtype)
            avg_context_latents = context_latents.mean(dim=1, keepdim=True)
            context_offset = compute_stage2_context_offset(pipe.transformer, context_latents)

            # Image sampling loop
            latent_shape = (
                context_latents.shape[2],
                context_latents.shape[3],
                context_latents.shape[4],
            )

            num_samples = args.num_samples
            if num_samples <= 0:
                continue

            text_prompt_embeds = text_prompt_embeds_base.repeat(num_samples, 1, 1).contiguous()
            preference_prompt_embeds = preference_prompt_embeds_base.repeat(num_samples, 1, 1).contiguous()
            if do_classifier_free_guidance:
                uncond_prompt_embeds = uncond_prompt_embeds_base.repeat(num_samples, 1, 1).contiguous()

            generator = torch.Generator(device=device)
            generator.manual_seed(torch.randint(0, 10_000_000, (1,), device=device).item())

            latent_state = torch.randn(
                (num_samples, *latent_shape),
                generator=generator,
                device=device,
                dtype=pipe.transformer.dtype,
            )
 
            context_base = context_offset.expand(num_samples, *context_offset.shape[1:]).contiguous()
            context_base = context_base.to(device=device, dtype=pipe.transformer.dtype)

            with torch.no_grad():
                scheduler.set_timesteps(args.num_inference_steps, device=device)
                timesteps = scheduler.timesteps.to(device)
                context_zero = torch.zeros_like(context_base)

                for t in timesteps:
                    if do_classifier_free_guidance:
                        latent_model_input = torch.cat(
                            [latent_state, latent_state, latent_state, latent_state],
                            dim=0,
                        )
                        context_offsets = torch.cat(
                            [context_zero, context_zero, context_base, context_base],
                            dim=0,
                        )
                        latent_model_input = latent_model_input + MATCHING_LAMBDA * context_offsets
                        guidance_prompt_embeds = torch.cat(
                            [
                                uncond_prompt_embeds,
                                text_prompt_embeds,
                                text_prompt_embeds,
                                preference_prompt_embeds,
                            ],
                            dim=0,
                        )
                        timestep = t.expand(latent_model_input.shape[0]).to(device)
                    else:
                        latent_model_input = latent_state + MATCHING_LAMBDA * context_base
                        guidance_prompt_embeds = preference_prompt_embeds
                        timestep = t.expand(num_samples).to(device)

                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=pipe.transformer.dtype):
                        noise_pred = pipe.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=guidance_prompt_embeds,
                            pooled_projections=None,
                            return_dict=False,
                        )[0]

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text, noise_pred_match, noise_pred_preference = noise_pred.chunk(4)
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + context_guidance_scale * (noise_pred_match - noise_pred_text)
                            + preference_guidance_scale * (noise_pred_preference - noise_pred_match)
                        )

                    latent_state = scheduler.step(noise_pred, t, latent_state, return_dict=False)[0]

            gen_image_paths = []
            for sample_idx in range(num_samples):
                sample_output_path = os.path.join(
                    sample_dir, f"{target_item_id}_{sample_idx}.png"
                )
                decode_and_save_image(pipe, latent_state[sample_idx : sample_idx + 1], sample_output_path)
                gen_image_paths.append(os.path.abspath(sample_output_path))

            # Caption sampling
            caption_inputs = prepare_masked_caption_inputs(
                pipe.tokenizer,
                caption_text,
                mask_keys,
                args.max_length,
                device,
            )
            attention_mask = caption_inputs["attention_mask"]
            masked_input_ids = caption_inputs["masked_input_ids"]
            prompt_mask = caption_inputs["prompt_mask"]
            caption_batch = caption_sample_count if caption_sample_count > 0 else 1

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=pipe.transformer.dtype):
                    masked_caption_emb = pipe.text_encoder(
                        masked_input_ids,
                        attention_mask=attention_mask,
                    )[0]

            conditioning_embeds = torch.cat(
                [
                    preference_emb.to(pipe.transformer.dtype),
                    sentence_emb.to(pipe.transformer.dtype),
                    masked_caption_emb.to(pipe.transformer.dtype),
                ],
                dim=1,
            )

            caption_condition = avg_context_latents.squeeze(1).to(device=device, dtype=pipe.transformer.dtype)
            if caption_condition.shape[0] != caption_batch:
                caption_condition = caption_condition.expand(caption_batch, *caption_condition.shape[1:]).contiguous()

            if masked_input_ids.shape[0] != caption_batch:
                masked_input_ids = masked_input_ids.expand(caption_batch, -1).contiguous()
                prompt_mask = prompt_mask.expand(caption_batch, -1).contiguous()

            if conditioning_embeds.shape[0] != caption_batch:
                conditioning_embeds = conditioning_embeds.expand(caption_batch, -1, -1).contiguous()

            pipe.text_sampler.register_condition(
                caption_condition,
                masked_input_ids,
                prompt_mask,
                conditioning_embeds=conditioning_embeds,
            )
            pred_tokens = pipe.text_sampler.sample(
                args.max_length,
                args.caption_steps,
                batch_size_per_gpu=caption_batch,
                device=device,
            )
            pipe.text_sampler.clear_condition()

            caption_entries = []
            for caption_idx in range(pred_tokens.shape[0]):
                generated_caption = pipe.tokenizer.decode(pred_tokens[caption_idx], skip_special_tokens=True)
                entry = {
                    "caption_sample_idx": caption_idx,
                    "generated_caption": generated_caption,
                }
                if caption_idx < len(gen_image_paths):
                    entry["image_path"] = gen_image_paths[caption_idx]
                caption_entries.append(entry)

            caption_record = {
                "uid": uid,
                "oid": oid,
                "target_item_id": target_item_id,
                "preference_text": preference_text,
                "sentence_text": sentence_text,
                "caption_template": caption_text,
                "generated_captions": caption_entries,
            }

            with open(os.path.join(sample_dir, "generated_caption.json"), "w", encoding="utf-8") as f:
                json.dump(caption_record, f, ensure_ascii=False, indent=2)

            # Save ground truth and context images
            gt_rel_path = test_image_paths[target_item_id] if target_item_id < len(test_image_paths) else None
            ground_image_path = None
            if gt_rel_path:
                gt_source_path = os.path.join(args.image_root, gt_rel_path)
                if os.path.exists(gt_source_path):
                    with Image.open(gt_source_path).convert("RGB") as img:
                        grd_dest_path = os.path.join(sample_dir, f"{target_item_id}_grd.png")
                        img.save(grd_dest_path)
                        ground_image_path = os.path.abspath(grd_dest_path)

            context_image_paths = []
            for context_id in context_item_ids:
                rel_path = test_image_paths[context_id] if context_id < len(test_image_paths) else None
                if rel_path:
                    source_path = os.path.join(args.image_root, rel_path)
                    if os.path.exists(source_path):
                        with Image.open(source_path).convert("RGB") as img:
                            dest_path = os.path.join(sample_dir, f"incomplete_outfit_{context_id}.png")
                            img.save(dest_path)
                            context_image_paths.append(os.path.abspath(dest_path))

            if not gen_image_paths:
                print(f"[WARN] Skipping FITB entry {uid}_{oid}: generated images missing.")
                continue
            if ground_image_path is None:
                print(f"[WARN] Skipping FITB entry {uid}_{oid}: ground truth image missing.")
                continue

            outfit_array = np.asarray(outfit_full, dtype=np.int64)
            gen_record = {
                "image_paths": gen_image_paths,
                "cates": np.full(len(gen_image_paths), target_category_id, dtype=np.int64),
                "incomplete_outfit": list(context_image_paths),
                "outfits": outfit_array,
            }
            grd_record = {
                "image_paths": [ground_image_path],
                "cates": np.array([target_category_id], dtype=np.int64),
                "incomplete_outfit": list(context_image_paths),
                "outfits": outfit_array,
            }
            gen_data[uid][oid] = gen_record
            grd_data[uid][oid] = grd_record

        # free resources before next model
        del pipe
        torch.cuda.empty_cache()

        gen_payload = {uid: dict(oid_map) for uid, oid_map in gen_data.items()}
        grd_payload = {uid: dict(oid_map) for uid, oid_map in grd_data.items()}
        np.save(os.path.join(output_model_dir, "gen.npy"), gen_payload)
        np.save(os.path.join(output_model_dir, "grd.npy"), grd_payload)


if __name__ == "__main__":
    main()
