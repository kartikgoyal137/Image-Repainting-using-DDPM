"""DDPM Gaussian Diffusion with RePaint inpainting schedule."""

import numpy as np
import torch as th
from collections import defaultdict


def get_named_beta_schedule(num_diffusion_timesteps):
    scale = 1000 / num_diffusion_timesteps
    return np.linspace(scale * 0.0001, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)


def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample, **kwargs):
    """Generate the RePaint resampling schedule (single-level jumps)."""
    jumps = {j: jump_n_sample - 1 for j in range(0, t_T - jump_length, jump_length)}
    t, ts = t_T, []
    while t >= 1:
        t -= 1
        ts.append(t)
        if t + 1 < t_T - 1:
            for _ in range(n_sample - 1):
                t += 1; ts.append(t)
                if t >= 0:
                    t -= 1; ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t += 1; ts.append(t)
    ts.append(-1)
    return ts


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        section_counts = [int(x) for x in section_counts.split(",")]
    if isinstance(section_counts, int):
        section_counts = [section_counts]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx, all_steps = 0, []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        for _ in range(section_count):
            all_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        start_idx += size
    return set(all_steps)


def _extract(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion:
    def __init__(self, betas, learn_sigma=False, rescale_timesteps=False, conf=None):
        self.rescale_timesteps = rescale_timesteps
        self.conf = conf
        self.learn_sigma = learn_sigma

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        ac_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = betas * (1.0 - ac_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(ac_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - ac_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        model_output, model_var_values = th.split(model_output, C, dim=1)

        if self.learn_sigma:
            min_log = _extract(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            log_variance = frac * max_log + (1 - frac) * min_log
        else:
            log_variance = _extract(np.log(np.append(self.posterior_variance[1],
                                   self.betas[1:])), t, x.shape)

        # Epsilon prediction (DDPM)
        pred_xstart = (
            _extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * model_output
        )
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        mean = (
            _extract(self.posterior_mean_coef1, t, x.shape) * pred_xstart
            + _extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        return mean, log_variance, pred_xstart

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None,
                 conf=None, pred_xstart=None, **kwargs):
        noise = th.randn_like(x)

        # RePaint: inject known region with noise at this timestep
        if conf.inpa_inj_sched_prev and pred_xstart is not None:
            gt = model_kwargs['gt']
            mask = model_kwargs.get('gt_keep_mask')
            ac = _extract(self.alphas_cumprod, t, x.shape)
            noised_gt = th.sqrt(ac) * gt + th.sqrt(1 - ac) * th.randn_like(x)
            x = mask * noised_gt + (1 - mask) * x

        mean, log_variance, pred_xstart = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean + nonzero_mask * th.exp(0.5 * log_variance) * noise

        return {"sample": sample, "pred_xstart": pred_xstart, "gt": model_kwargs.get('gt')}

    def undo(self, img, t):
        beta = _extract(self.betas, t, img.shape)
        return th.sqrt(1 - beta) * img + th.sqrt(beta) * th.randn_like(img)

    def p_sample_loop(self, model, shape, clip_denoised=True, model_kwargs=None,
                      device=None, progress=True, return_all=False, conf=None, **kwargs):
        if device is None:
            device = next(model.parameters()).device

        img = th.randn(*shape, device=device)
        pred_xstart = None
        out = None

        times = get_schedule_jump(**conf.schedule_jump_params)
        time_pairs = list(zip(times[:-1], times[1:]))
        if progress:
            from tqdm.auto import tqdm
            time_pairs = tqdm(time_pairs)

        for t_last, t_cur in time_pairs:
            t_batch = th.tensor([t_last] * shape[0], device=device)
            if t_cur < t_last:  # denoise step
                with th.no_grad():
                    out = self.p_sample(model, img, t_batch, clip_denoised=clip_denoised,
                                        model_kwargs=model_kwargs, conf=conf,
                                        pred_xstart=pred_xstart)
                    img = out["sample"]
                    pred_xstart = out["pred_xstart"]
            else:  # undo (re-noise) step
                t_shift = conf.get('inpa_inj_time_shift', 1)
                img = self.undo(img, t_batch + t_shift)

        return out if return_all else out["sample"]

    def _scale_timesteps(self, t):
        return t.float() * (1000.0 / self.num_timesteps) if self.rescale_timesteps else t


class SpacedDiffusion(GaussianDiffusion):
    """Diffusion with timestep respacing for faster inference."""
    def __init__(self, use_timesteps, **kwargs):
        betas = kwargs["betas"]
        original_alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        self.timestep_map = []
        new_betas = []
        last_ac = 1.0
        for i, ac in enumerate(original_alphas_cumprod):
            if i in set(use_timesteps):
                new_betas.append(1 - ac / last_ac)
                last_ac = ac
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap(model), *args, **kwargs)

    def _wrap(self, model):
        if isinstance(model, _W):
            return model
        return _W(model, self.timestep_map)

    def _scale_timesteps(self, t):
        return t


class _W:
    """Wraps model to remap timesteps for spaced diffusion."""
    def __init__(self, model, timestep_map):
        self.model = model
        self.timestep_map = timestep_map

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        return self.model(x, map_tensor[ts], **kwargs)
