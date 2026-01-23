import torch
import torch.nn as nn


def prepare_text_inputs(model_pipe, t5_input_ids, attention_mask, disable_grad=True):
    if not disable_grad:
        t5_embeds = model_pipe.text_encoder_3(t5_input_ids, attention_mask=attention_mask)[0]
        return t5_embeds
    else:
        with torch.no_grad():
            t5_embeds = model_pipe.text_encoder_3(t5_input_ids, attention_mask=attention_mask)[0]
        return t5_embeds.detach()


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
        f_T = torch.log1p(-torch.exp(-self.sigma_max))
        f_0 = torch.log1p(-torch.exp(-self.sigma_min))
        sigma_t = -torch.log1p(-torch.exp(t * f_T + (1 - t) * f_0))
        t = -torch.expm1(-sigma_t) / (1 - self.eps)
        return t

    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)


class TextMaskedDiffusionLoss:
    def __init__(self, config, model_pipe, grad_norm=False):
        self.noise = LogLinearNoise()
        self.sampling_eps = config.training.sampling_eps
        self.antithetic_sampling = config.training.antithetic_sampling
        self.importance_sampling = config.training.importance_sampling
        self.ignore_padding = config.training.ignore_padding
        self.mask_index = 32099
        self.neg_infinity = -1000000.0
        self.model_pipe = model_pipe
        self.grad_norm = grad_norm

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
        bs, seq_len = x.shape
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = xt != self.mask_index
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _forward_pass_diffusion(
        self,
        model,
        x0,
        image=None,
        attention_mask=None,
        return_dummy_output=True,
        disable_t5_grad=True,
        cond_mask=None,
        conditioning_embeds=None,
    ):
        t = self._sample_t(x0.shape[0], x0.device)
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])

        if cond_mask is not None:
            xt = x0 * (1 - cond_mask) + cond_mask * self.q_xt(x0, move_chance)
        else:
            xt = self.q_xt(x0, move_chance)

        text_hidden_states = prepare_text_inputs(
            self.model_pipe,
            xt,
            attention_mask=None,
            disable_grad=disable_t5_grad,
        )

        if conditioning_embeds is not None:
            text_hidden_states = torch.cat(
                [text_hidden_states, conditioning_embeds.to(text_hidden_states.dtype)],
                dim=1,
            )
        if image is not None:
            image = image.detach()

        timestep_tensor = torch.zeros(t.shape[0], device=t.device, dtype=text_hidden_states.dtype)

        if return_dummy_output:
            dummy_output, model_output = model(
                hidden_states=image,
                timestep=timestep_tensor,
                encoder_hidden_states=text_hidden_states,
                pooled_projections=None,
            )
        else:
            model_output = model(
                hidden_states=image,
                timestep=timestep_tensor,
                encoder_hidden_states=text_hidden_states,
                pooled_projections=None,
            )[1]
        seq_len = x0.shape[1]
        vocab_size = model_output.shape[-1]
        # model_output may have longer sequence due to appended conditioning; keep caption part
        model_output = model_output[..., :seq_len, :]
        x0 = x0.view(-1, seq_len)
        xt = xt.reshape(-1, seq_len)
        if cond_mask is None:
            cond_mask = torch.ones_like(xt, dtype=model_output.dtype, device=model_output.device)
        weight = cond_mask * (dsigma / torch.expm1(sigma))[:, None]
        x0_prob = torch.softmax(model_output, dim=-1)
        target_prob = x0_prob.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        loss = torch.sum(x0_prob * x0_prob, dim=-1) - 2 * target_prob
        loss = weight * loss
        return loss, model_output

    def compute_loss(
        self,
        model,
        input_tokens,
        image_condition,
        attention_mask,
        use_dummy_loss=True,
        label_mask=None,
        conditioning_embeds=None,
        **kwargs,
    ):
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
            )[0]

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
