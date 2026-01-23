import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from diffusers import FlowMatchEulerDiscreteScheduler
from .sd3_training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


def prepare_text_inputs(model_pipe,
                         t5_input_ids,
                         attention_mask,
                         disable_grad=True):
    if not disable_grad:
        t5_embeds = model_pipe.text_encoder_3(t5_input_ids, attention_mask=attention_mask)[0]
        return t5_embeds
    else:
        with torch.no_grad():
            t5_embeds = model_pipe.text_encoder_3(t5_input_ids, attention_mask=attention_mask)[0]
        return t5_embeds.detach()


# this normalize the gradient of the loss by its gradient norm
# taken from: https://github.com/cloneofsimo/vqgan-training
class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):

        grad_output_norm = torch.norm(grad_output)

        grad_output_normalized = grad_output / (grad_output_norm + 1e-8)

        return grad_output_normalized


def gradnorm(x):
    return GradNormFunction.apply(x)


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

class TextMaskedDiffusionLoss:
    def __init__(self, config, model_pipe, grad_norm=False):
        self.noise = LogLinearNoise()
        self.sampling_eps = config.training.sampling_eps
        self.antithetic_sampling = config.training.antithetic_sampling
        self.importance_sampling = config.training.importance_sampling
        self.ignore_padding = config.training.ignore_padding
        self.mask_index = 32099             # the token id for <extra_id 0> in t5 model, hard-coded
        self.neg_infinity = -1000000.0
        self.model_pipe = model_pipe
        self.grad_norm = grad_norm

    @torch.no_grad()
    def get_null_embeds(self):
        # not needed for now
        _, _, pooled_embeds, _ = self.model_pipe.encode_prompt(
                            prompt=' ',
                            prompt_2=None,
                            prompt_3=None,
                            max_sequence_length=33 # hard-coded
                            )
        self.null_pooled_embeds = pooled_embeds.detach()
        
    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
        _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input. 
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        bs, seq_len = x.shape
        move_indices = torch.rand(
                    * x.shape, device=x.device) < move_chance
    
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity
        
        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1,
                                        keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0

        return logits


    def _forward_pass_diffusion(
        self,
        model,
        x0,  # un-masked indices
        image=None,  # image conditioning
        attention_mask=None,
        return_dummy_output=True,
        disable_t5_grad=True,
        cond_mask=None,
        conditioning_embeds=None, # text conditioning
    ):
        assert attention_mask is None
        t = self._sample_t(x0.shape[0], x0.device)

        sigma, dsigma = self.noise(t)
        # sigma_conditioning = sigma[:, None] # not needed because the text diffusion part is time-agnostic
        move_chance = 1 - torch.exp(-sigma[:, None])

        if cond_mask is not None:
            ## replace with the chance
            xt = x0 * (1 - cond_mask) + cond_mask * self.q_xt(x0, move_chance)
            
            # mask all value
            # mask_token = torch.full_like(x0, self.mask_index)
            # xt = torch.where(cond_mask.bool(), mask_token, x0)
        else:
            # assert False
            xt = self.q_xt(x0, move_chance) # randomly mask
        text_hidden_states = prepare_text_inputs(self.model_pipe,
                                                 xt,
                                                 attention_mask=None, # disable attention mask for padding toekns
                                                 disable_grad=disable_t5_grad
                                                 )
        # if torch.any(torch.isnan(text_hidden_states)):
        #     ratio = torch.sum(torch.isnan(text_hidden_states)) / (text_hidden_states.shape[0] * text_hidden_states.shape[1])
        #     logger.info(f'nan in text embedding, ratio: {ratio}')
        
        # Here the text condition embeddding is concat with the initial text embedding and fed it into the joint transformer during caption diffsion
        if conditioning_embeds is not None:
            text_hidden_states = torch.cat(
                [text_hidden_states, conditioning_embeds.to(text_hidden_states.dtype)],
                dim=1,
            )
        if image is not None:
            image = image.detach()

        if return_dummy_output:
            dummy_output, model_output = model(hidden_states=image,
                                    timestep=torch.zeros(t.shape[0], device=t.device), # note that, this time embedding is for image, we don't use time embedding for text
                                    encoder_hidden_states=text_hidden_states,
                                    pooled_projections=None, # save up compute, just set it to None
                                    )
        else:
            model_output = model(hidden_states=image,
                                    timestep=torch.zeros(t.shape[0], device=t.device),
                                    encoder_hidden_states=text_hidden_states,
                                    pooled_projections=None,
                                    )[1]
        seq_len = x0.shape[1]
        if model_output.shape[1] != seq_len:
            # drop extra positions produced by concatenated conditioning embeds
            model_output = model_output[:, :seq_len, :]
        if self.grad_norm:
            model_output = gradnorm(model_output)
        
        # SUBS parameterization, continuous time.
        logits = self._subs_parameterization(model_output, xt)

        # compare the predicted token distribution with groundtruth token distribution
        log_p_theta = torch.gather(
                input=logits[:, :],
                dim=-1,
                index=x0[:, :, None]).squeeze(-1)
        if return_dummy_output:
            return - log_p_theta * (
                    dsigma / torch.expm1(sigma))[:, None], dummy_output
        else:
            return - log_p_theta * (
                    dsigma / torch.expm1(sigma))[:, None]

    def compute_loss(self, model, 
                    input_tokens, image_condition, attention_mask, use_dummy_loss=True,
                    label_mask=None, conditioning_embeds=None, **kwargs):
        if use_dummy_loss:
            loss, dummy_output = self._forward_pass_diffusion(
                model,
                input_tokens,
                image_condition,
                attention_mask,
                return_dummy_output=True,
                conditioning_embeds=conditioning_embeds,
                **kwargs,
            )
        else:
            loss = self._forward_pass_diffusion(
                model,
                input_tokens,
                image_condition,
                attention_mask,
                return_dummy_output=False,
                cond_mask=label_mask,
                conditioning_embeds=conditioning_embeds,
                **kwargs,
            )

        if label_mask is not None:
            loss = loss * label_mask
        
        if self.ignore_padding:
            nlls = loss * attention_mask
            count = attention_mask.sum()
        else:
            nlls = loss
            count = input_tokens.shape[0] * input_tokens.shape[1]

        batch_nll = nlls.sum()
        if use_dummy_loss:
            token_nll = batch_nll / count + (dummy_output - dummy_output).mean()
        else:
            token_nll = batch_nll / count

        return token_nll


class ImageFlowMatchingLoss:
    # flow matching loss following sd3 implementation
    def __init__(self, model_pipe, text_max_length=256, grad_norm=False):
        # get noise scheduler from model pipe
        # self.noise_scheduler =  copy.deepcopy(model_pipe.scheduler)
        # the original scheduler contains a shift for higher resolution images
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()

        # cache for null embeddings at different sequence lengths
        self.model_pipe = model_pipe
        self.null_embeds_cache = {}
        self._register_null_embeds(text_max_length)
        # print out how many steps are there
        # print(f'total steps: {self.noise_scheduler.config.num_train_timesteps}')

        self.grad_norm = grad_norm

    def _register_null_embeds(self, seq_len):
        if seq_len in self.null_embeds_cache:
            return
        null_embeds = self.model_pipe._get_t5_prompt_embeds(
            prompt=' ',
            max_sequence_length=seq_len
        )
        self.null_embeds_cache[seq_len] = null_embeds

    def _get_null_embeds(self, seq_len, device, dtype):
        if seq_len not in self.null_embeds_cache:
            self._register_null_embeds(seq_len)
        return self.null_embeds_cache[seq_len].to(device=device, dtype=dtype)

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def compute_loss(self, model,
                      text_emb,  # clean text embedding
                      image,
                      context_proj: torch.Tensor | None = None,
                      lamda: float = 0.0,
                      gt: torch.Tensor | None = None,
                      ):
        # Sample noise that we'll add to the latents
        model_input = image
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme='logit_normal', # perform best in sd3 paper
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=model_input.device)

        sigmas = self.get_sigmas(timesteps,
                                 device=model_input.device,
                                 n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        lamda_value = float(lamda)
        if lamda_value != 0.0:
            lamda_tensor = torch.tensor(lamda_value, device=model_input.device, dtype=model_input.dtype)
            scale_factor = 1.0 - lamda_tensor
        else:
            lamda_tensor = None
            scale_factor = 1.0
        if context_proj is not None:
            context_proj = context_proj.to(device=model_input.device, dtype=model_input.dtype)
            noisy_model_input = scale_factor * noisy_model_input + (lamda_tensor if lamda_tensor is not None else 0.0) * context_proj
        else:
            noisy_model_input = scale_factor * noisy_model_input

        # dropout text_emb with 0.1
        dropout_idx = torch.rand(bsz, device=model_input.device) < 0.1
        if dropout_idx.any():
            null_embeds = self._get_null_embeds(
                seq_len=text_emb.shape[1],
                device=model_input.device,
                dtype=text_emb.dtype,
            )
            null_embeds = null_embeds.expand(text_emb.shape[0], -1, -1)
            text_emb = torch.where(
                dropout_idx[:, None, None],
                null_embeds,
                text_emb,
            )

        # ensure embeddings keep gradient
        text_emb_input = text_emb
        if not text_emb_input.requires_grad:
            text_emb_input = text_emb_input.clone().detach().requires_grad_(True)

        # do the velocity prediction
        model_pred = model(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=text_emb_input,
                    pooled_projections=None,    # disable pooled projection
                    return_dict=False,
                )[0]
        if self.grad_norm:
            model_pred = gradnorm(model_pred)
        
        model_pred = model_pred * (-sigmas) + noisy_model_input

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme='logit_normal', sigmas=sigmas)

        target = gt # reconstruction loss
        # target = GT # reconstruction loss

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        return loss
