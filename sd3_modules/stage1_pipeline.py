
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Union

import PIL

import torch
import numpy as np
from transformers import (
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from sd3_modules.sd3_model import SD3JointModelFlexible


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

TARGET_MASK_KEYS = [
    "Color",
    "Material",
    "Design features",
    "Clothing Fashion Style",
]


def _find_value_spans(text: str):
    spans = []
    for key in TARGET_MASK_KEYS:
        pattern = rf'"{key}"\s*:\s*"([^"]*)"'
        for match in re.finditer(pattern, text):
            spans.append((match.start(1), match.end(1)))
    return spans


def _build_label_mask(token_info, raw_text: str):
    if "offset_mapping" not in token_info:
        return torch.zeros_like(token_info["input_ids"])
    offsets = token_info["offset_mapping"][0]
    mask = torch.zeros_like(token_info["input_ids"])
    spans = _find_value_spans(raw_text)
    for start, end in spans:
        token_hits = (offsets[:, 0] < end) & (offsets[:, 1] > start)
        if token_hits.any():
            mask[0, token_hits] = 1
    if "attention_mask" in token_info:
        mask = mask * token_info["attention_mask"]
    return mask


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



class ImageProcessor:
    def __init__(self, resolution):
        self.resolution = resolution
        self.interpolate_model = InterpolationMode.BICUBIC
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ])
    
    def change_resolution(self, resolution):
        self.resolution = resolution
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ])

    def preprocess(self, img):
        # transform pil image to pytorch tensor
        return self.transform(img)[None] # add batch dimension


def _sample_categorical(categorical_probs):
    gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)



class LogLinearNoise(torch.nn.Module):
    """Log Linear noise schedule.

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and
    ~1 when t varies from 0 to 1. Total noise is
    -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def importance_sampling_transformation(self, t):
        f_T = torch.log1p(- torch.exp(- self.sigma_max))
        f_0 = torch.log1p(- torch.exp(- self.sigma_min))
        sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
        t = - torch.expm1(- sigma_t) / (1 - self.eps)
        return t

    def forward(self, t):
        # Assume time goes from 0 to 1
        return self.total_noise(t), self.rate_noise(t)

@torch.no_grad()
def prepare_text_inputs(model_pipe,
                         t5_input_ids,
                         attention_mask):
    t5_embeds = model_pipe.text_encoder_3(t5_input_ids, attention_mask=attention_mask)[0]
    return t5_embeds


class ConditionalMaskedDiffusionSampler:

    def __init__(self, mask_index, text_encoder, text_denoiser):
    # the model needs to be wrapped with SUBS parameterization at first
        self.mask_index = mask_index
        self.noise = LogLinearNoise()
        self.text_encoder = text_encoder

        self.noise_removal = True
        self.sampler = 'ddpm_cache'
        self.neg_infinity = -1000000.0
        self.y = None
        self.model = text_denoiser

    def register_condition(self, y, x, x_mask):
        self.y = y
        self.x = x
        self.x_mask = x_mask
    
    def clear_condition(self):
        self.y = None
        self.x = None
        self.x_mask = None

    def prepare_text_inputs(self, xt, attention_mask):
        t5_embeds = self.text_encoder(xt, attention_mask=attention_mask)[0]
        return t5_embeds

    def forward(self, xt):
        cond = self.y
        mask_index = self.mask_index
        text_hidden_states = self.prepare_text_inputs(xt, attention_mask=None)
        
        with torch.cuda.amp.autocast(enabled=True):  
            # note that xt is indices   
            logits = self.model(hidden_states=cond,
                            timestep=torch.zeros(xt.shape[0], device=xt.device), # note that, this time embedding is for image, we don't use time embedding for text
                            encoder_hidden_states=text_hidden_states.detach(),
                            pooled_projections=None)[1]
            
        logits = logits.float()

        # log prob at the mask index = - infinity
        logits[:, :, mask_index] += self.neg_infinity
        
        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1,
                                        keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits
    
    def _sample_prior(self, *batch_dims):
        masked = self.mask_index * torch.ones(
        * batch_dims, dtype=torch.int64)
        prior = torch.where(self.x_mask, self.x, masked)
        return prior

    def _ddpm_update(self, x, t, dt):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        log_p_x0 = self.forward(x)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x
    
    def _ddpm_caching_update(self, x, t, dt, p_x0):
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self.forward(x).exp()
        
        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    @torch.no_grad()
    def sample(self,
               max_length,  
               num_steps,
               eps=1e-5,
               batch_size_per_gpu=1, 
               device='cuda'):
        assert self.y is not None, "Please register the image condition first"
        batch_size_per_gpu = batch_size_per_gpu

        x = self._sample_prior(
            batch_size_per_gpu,
            max_length).to(device)

        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None


        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            if self.sampler == 'ddpm':
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == 'ddpm_cache':
                p_x0_cache, x_next = self._ddpm_caching_update(
                                    x, t, dt, p_x0=p_x0_cache)
                if not torch.allclose(x_next, x):
                    # Disable caching
                    p_x0_cache = None
                x = x_next

        if self.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                            device=device)
            
            x = self.forward(x).argmax(dim=-1)
        return x


class DualDiffSD3Pipeline(DiffusionPipeline, FromSingleFileMixin):

    def __init__(
        self,
        transformer: SD3JointModelFlexible,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = ImageProcessor(512)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 64  # corresponds to image resolution of 512
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

        # Define the parameters for T5 and masked diffusion process
        self.neg_infinity = -1000000.0
        self.mask_index = 32099  # corresponds to the mask token "<'extra_id0'>" in T5

        self.text_sampler = ConditionalMaskedDiffusionSampler(
            mask_index=self.mask_index,
            text_encoder=self.text_encoder,
            text_denoiser=self.transformer,
        )

        self.set_sampling_mode('t2i')  # text to image by default


    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]

        dtype = self.text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # this is for image generation
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 256,
    ):

        device = device or self._execution_device


        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
           

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            prompt_embeds = t5_prompt_embed

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_prompt_embeds = t5_negative_prompt_embed

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    def prepare_image_inputs(self, image_tsr, device):
        device = device or self._execution_device
        image_tsr = image_tsr.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            image_vae_feat = self.vae.encode(image_tsr).latent_dist.sample()
            image_vae_feat = (image_vae_feat - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_vae_feat

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def skip_guidance_layers(self):
        return self._skip_guidance_layers

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def sampling_mode(self):
        return self._sampling_mode

    def set_sampling_mode(self, sampling_mode: str):
        self._sampling_mode = sampling_mode
        assert sampling_mode in ["t2i", "i2t"], "Sampling mode must be either 't2i' (text to image) or 'i2t' (image to text)."

    def text_to_image_sampling_loop(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "image",       # image or latent
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 256,
        mu: Optional[torch.Tensor] = None,
        ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device


        (
            prompt_embeds,
            negative_prompt_embeds
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if self.do_classifier_free_guidance:

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                with torch.cuda.amp.autocast(enabled=True):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=None,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]


                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            images = latents

        else: # image
            with torch.cuda.amp.autocast(enabled=True):

                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                images = self.vae.decode(latents, return_dict=False)[0]

        return images

    def image_to_text_sampling_loop(
        self,
        image: PIL.Image,
        prompt: Optional[str] = None,   # you can put visual question here
        sequence_length: Optional[int] = 128,
        num_inference_steps: Optional[int] = 64,
        resolution: Optional[int] = 512,
        device: Optional[torch.device] = None,
        ):
        # currently we do not use batch sampling by default 
        # because we use ddpm_cache for masked diffusion which is only efficient under batch size 1
        max_length = sequence_length
        if prompt is not None:
            token_info = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            input_ids = token_info["input_ids"]
            attention_mask = token_info["attention_mask"]
            label_mask = _build_label_mask(token_info, prompt)

            masked_input_ids = input_ids.clone()
            masked_positions = label_mask.bool()
            masked_input_ids[masked_positions] = self.mask_index

            prompt_mask = attention_mask.bool() & (~masked_positions)
            text = masked_input_ids
        else:
            text = torch.zeros((1, max_length), dtype=torch.long)
            prompt_mask = torch.zeros((1, max_length), dtype=torch.bool)

        if resolution != self.image_processor.resolution:
            self.image_processor.change_resolution(resolution)
        img_tsr = self.image_processor.preprocess(image)
        device = self._execution_device

        img_feat = self.prepare_image_inputs(img_tsr, device)
        self.text_sampler.register_condition(img_feat, text, prompt_mask)
        pred = self.text_sampler.sample(
                             max_length,
                             num_inference_steps,
                             batch_size_per_gpu=img_feat.shape[0], device=device)

        # decode with tokenizer
        text = self.tokenizer.decode(pred[0]).replace('</s>', '')
        return text


    @torch.no_grad()
    def __call__(
        self,
        *args,
        **kwargs,
    ):
        if self.sampling_mode == "t2i":
            return self.text_to_image_sampling_loop(*args, **kwargs)
        elif self.sampling_mode == "i2t":
            return self.image_to_text_sampling_loop(*args, **kwargs)
        else:
            raise ValueError("Sampling mode must be either 't2i' (text to image) or 'i2t' (image to text).")
        

        
