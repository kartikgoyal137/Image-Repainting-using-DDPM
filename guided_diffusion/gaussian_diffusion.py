"""Gaussian Diffusion with SpacedDiffusion and RePaint inpainting."""

import enum
import numpy as np
import torch as th
from collections import defaultdict
from guided_diffusion.scheduler import get_schedule_jump


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    raise ValueError(f"Unknown beta schedule: {schedule_name}")


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


# --- space_timesteps (from respace.py) ---

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
        section_counts = [int(x) for x in section_counts.split(",")]
    if isinstance(section_counts, int):
        section_counts = [section_counts]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []

    if len(section_counts) == 1 and section_counts[0] > num_timesteps:
        return set(np.linspace(start=0, stop=num_timesteps, num=section_counts[0]))

    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


# --- GaussianDiffusion ---

class GaussianDiffusion:
    def __init__(self, *, betas, model_mean_type, model_var_type,
                 rescale_timesteps=False, conf=None):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.rescale_timesteps = rescale_timesteps
        self.conf = conf

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        beta = _extract_into_tensor(self.betas, t, img_after_model.shape)
        return th.sqrt(1 - beta) * img_after_model + th.sqrt(beta) * th.randn_like(img_after_model)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)

        if self.model_var_type == ModelVarType.LEARNED:
            model_log_variance = model_var_values
            model_variance = th.exp(model_log_variance)
        else:
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)

        if self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        else:
            raise NotImplementedError(self.model_mean_type)

        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None,
                 conf=None, pred_xstart=None, **kwargs):
        noise = th.randn_like(x)

        if conf.inpa_inj_sched_prev and pred_xstart is not None:
            gt_keep_mask = model_kwargs.get('gt_keep_mask')
            gt = model_kwargs['gt']
            alpha_cumprod = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

            gt_weight = th.sqrt(alpha_cumprod)
            noise_weight = th.sqrt(1 - alpha_cumprod)
            weighed_gt = gt_weight * gt + noise_weight * th.randn_like(x)

            x = gt_keep_mask * weighed_gt + (1 - gt_keep_mask) * x

        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised,
                                   model_kwargs=model_kwargs)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"],
                "gt": model_kwargs.get('gt')}

    def p_sample_loop(self, model, shape, clip_denoised=True, model_kwargs=None,
                      cond_fn=None, device=None, progress=True, return_all=False, conf=None,
                      **kwargs):
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, clip_denoised=clip_denoised,
            model_kwargs=model_kwargs, device=device,
            progress=progress, conf=conf
        ):
            final = sample

        if return_all:
            return final
        return final["sample"]

    def p_sample_loop_progressive(self, model, shape, clip_denoised=True,
                                  model_kwargs=None, device=None,
                                  progress=False, conf=None, **kwargs):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        image_after_step = th.randn(*shape, device=device)
        self.gt_noises = None
        pred_xstart = None
        sample_idxs = defaultdict(lambda: 0)

        if conf.schedule_jump_params:
            times = get_schedule_jump(**conf.schedule_jump_params)
            time_pairs = list(zip(times[:-1], times[1:]))
            if progress:
                from tqdm.auto import tqdm
                time_pairs = tqdm(time_pairs)

            for t_last, t_cur in time_pairs:
                t_last_t = th.tensor([t_last] * shape[0], device=device)

                if t_cur < t_last:  # reverse step
                    with th.no_grad():
                        out = self.p_sample(
                            model, image_after_step, t_last_t,
                            clip_denoised=clip_denoised,
                            model_kwargs=model_kwargs,
                            conf=conf, pred_xstart=pred_xstart)
                        image_after_step = out["sample"]
                        pred_xstart = out["pred_xstart"]
                        sample_idxs[t_cur] += 1
                        yield out
                else:  # forward (undo) step
                    t_shift = conf.get('inpa_inj_time_shift', 1)
                    image_after_step = self.undo(
                        image_after_step, image_after_step,
                        est_x_0=out['pred_xstart'], t=t_last_t + t_shift, debug=False)
                    pred_xstart = out["pred_xstart"]

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t


# --- SpacedDiffusion (from respace.py) ---

class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, conf=None, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.original_num_steps = len(kwargs["betas"])
        self.conf = conf

        base_diffusion = GaussianDiffusion(conf=conf, **kwargs)

        self.timestep_map = []
        new_betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        kwargs["betas"] = np.array(new_betas)
        super().__init__(conf=conf, **kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.rescale_timesteps,
                             self.original_num_steps)

    def _scale_timesteps(self, t):
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        return self.model(x, new_ts, **kwargs)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
