from diffusers.models.transformers import SD3Transformer2DModel
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from diffusers.models.modeling_outputs import Transformer2DModelOutput

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)

        conditioning = timesteps_emb

        return conditioning


class SD3JointModelFlexible(SD3Transformer2DModel):
    @register_to_config
    def __init__(
        self,
        vocab_size,    # the output token space
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        **kwargs # hold dummy argument here
        ):
        super().__init__(
            sample_size, patch_size, in_channels, num_layers, attention_head_dim, num_attention_heads, joint_attention_dim,
            caption_projection_dim, pooled_projection_dim, out_channels, pos_embed_max_size
        )

        # need to re-initialize the lasr transformer blocks to make it not context_pre_only
        self.transformer_blocks[-1] = \
                        JointTransformerBlock(
                                    dim=self.inner_dim,
                                    num_attention_heads=self.config.num_attention_heads,
                                    attention_head_dim=self.config.attention_head_dim,
                                    context_pre_only=False,
                                )
        self.norm_out = None
        self.proj_out = None
        # re-initialize the image time embedder, exclude the the pooled text embedding part
        self.time_text_embed = TimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.image_norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.image_proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.text_norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.text_proj_out = nn.Linear(self.inner_dim, vocab_size, bias=True)

        self.null_image_embd = nn.Parameter(torch.randn(1, 8, self.inner_dim)/ (self.inner_dim**0.5), requires_grad=True)

    def forward(
        self,
        hidden_states: torch.FloatTensor, # image
        encoder_hidden_states: torch.FloatTensor = None, # text
        pooled_projections: torch.FloatTensor = None, # not used
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        if hidden_states is not None:
            height, width = hidden_states.shape[-2:]
            hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
            text_only = False
        else:
            hidden_states = self.null_image_embd.repeat(encoder_hidden_states.shape[0], 1, 1)
            text_only = True

        temb = self.time_text_embed(timestep)
        
        # the text embedding are injected into same dimension with the image embedding.
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # breakpoint()

        # Attention 
        # 1) hidden_states - self attention
        # 2) Q:hidden_states(image), KV: encoder_hidden_states(text)
        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        if not text_only:
            hidden_states = self.image_norm_out(hidden_states, temb)
            hidden_states = self.image_proj_out(hidden_states)

            # unpatchify
            patch_size = self.config.patch_size
            height = height // patch_size
            width = width // patch_size
            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
            )
        else:
            hidden_states = hidden_states # this is used to create dummy gradient


        encoder_hidden_states = self.text_norm_out(encoder_hidden_states)
        encoder_hidden_states = self.text_proj_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states # more like a logits
    
    def get_fsdp_wrap_module_list(self):
        return self.transformer_blocks


        