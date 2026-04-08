"""DDPM Gaussian Diffusion with RePaint inpainting schedule."""

import numpy as np
import torch as th


def get_named_beta_schedule(steps):
    s = 1000 / steps
    return np.linspace(s * 0.0001, s * 0.02, steps, dtype=np.float64)

def get_U(U_min=1, U_max=10, A=None, D=None, mode='A'):
    if mode == 'A':
        score = A
    elif mode == 'D':
        score = min(D / 50.0, 1.0)
    elif mode == 'AD':
        score = 0.5 * (A ** 0.5) + 0.5 * min(D / 50.0, 1.0)
    else:
        raise ValueError(f"Unknown dynamic_U mode: {mode}")
    return U_min + int((U_max - U_min) * score)

def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample, **kw):
    def resolve_u(j):
        if callable(jump_n_sample):
            return jump_n_sample(j)
        return jump_n_sample
    jumps = {j: resolve_u(j) - 1 for j in range(0, t_T - jump_length, jump_length)}
    t, ts = t_T, []
    while t >= 1:
        t -= 1; ts.append(t)
        if t + 1 < t_T - 1:
            for _ in range(n_sample - 1):
                t += 1; ts.append(t)
                if t >= 0: t -= 1; ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length): t += 1; ts.append(t)
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
    for i, sc in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        stride = 1 if sc <= 1 else (size - 1) / (sc - 1)
        for j in range(sc):
            all_steps.append(start_idx + round(j * stride))
        start_idx += size
    return set(all_steps)


def _ex(arr, t, shape):
    r = th.from_numpy(arr).to(device=t.device)[t].float()
    while len(r.shape) < len(shape): r = r[..., None]
    return r.expand(shape)


class GaussianDiffusion:
    def __init__(self, betas, conf=None, **kw):
        self.conf = conf
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = len(betas)
        alphas = 1.0 - betas
        ac = self.alphas_cumprod = np.cumprod(alphas, axis=0)
        ac_prev = np.append(1.0, ac[:-1])
        self.sqrt_recip_ac = np.sqrt(1.0 / ac)
        self.sqrt_recipm1_ac = np.sqrt(1.0 / ac - 1)
        pv = self.posterior_variance = betas * (1.0 - ac_prev) / (1.0 - ac)
        self.post_log_var = np.log(np.append(pv[1], pv[1:]))
        self.post_coef1 = betas * np.sqrt(ac_prev) / (1.0 - ac)
        self.post_coef2 = (1.0 - ac_prev) * np.sqrt(alphas) / (1.0 - ac)

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        B, C = x.shape[:2]
        out = model(x, self._scale_timesteps(t), **(model_kwargs or {}))
        eps, var_val = th.split(out, C, dim=1)
        # Learned range variance
        min_log = _ex(self.post_log_var, t, x.shape)
        max_log = _ex(np.log(self.betas), t, x.shape)
        log_var = ((var_val + 1) / 2) * max_log + ((1 - var_val) / 2 + 0.5) * min_log
        # Epsilon -> x0
        pred_x0 = _ex(self.sqrt_recip_ac, t, x.shape) * x - _ex(self.sqrt_recipm1_ac, t, x.shape) * eps
        if clip_denoised: pred_x0 = pred_x0.clamp(-1, 1)
        mean = _ex(self.post_coef1, t, x.shape) * pred_x0 + _ex(self.post_coef2, t, x.shape) * x
        return mean, log_var, pred_x0

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None,
                 conf=None, pred_xstart=None, **kw):
        noise = th.randn_like(x)
        if conf.inpa_inj_sched_prev and pred_xstart is not None:
            gt, mask = model_kwargs['gt'], model_kwargs.get('gt_keep_mask')
            ac = _ex(self.alphas_cumprod, t, x.shape)
            x = mask * (th.sqrt(ac) * gt + th.sqrt(1 - ac) * th.randn_like(x)) + (1 - mask) * x

        mean, log_var, pred_x0 = self.p_mean_variance(model, x, t, clip_denoised, model_kwargs)
        nz = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return {"sample": mean + nz * th.exp(0.5 * log_var) * noise,
                "pred_xstart": pred_x0, "gt": model_kwargs.get('gt')}

    def undo(self, img, t):
        b = _ex(self.betas, t, img.shape)
        return th.sqrt(1 - b) * img + th.sqrt(b) * th.randn_like(img)

    def p_sample_loop(self, model, shape, clip_denoised=True, model_kwargs=None,
                      device=None, progress=True, return_all=False, conf=None, **kw):
        device = device or next(model.parameters()).device
        img, pred_xstart, out = th.randn(*shape, device=device), None, None
        jump_params = dict(conf.schedule_jump_params)
        du = conf.pget('dynamic_U')
        if du:
            mode  = du.get('mode', 'A')
            U_min = du.get('U_min', 1)
            U_max = du.get('U_max', 10)
            t_T   = conf.schedule_jump_params['t_T']
            mask  = model_kwargs.get('gt_keep_mask')

            def compute_D(mask):
                import torch.nn.functional as F
                kernel = th.ones(1, 1, 3, 3, device=mask.device)
                m = (mask[:, 0:1] > 0.5).float()
                eroded = F.conv2d(m, kernel, padding=1)
                boundary = ((eroded < 9) & (m > 0.5)).float()
                if boundary.sum() == 0:
                    return 50.0
                coords = boundary[0, 0].nonzero(as_tuple=False).float()
                all_coords = th.stack(th.meshgrid(
                    th.arange(m.shape[-2], device=mask.device),
                    th.arange(m.shape[-1], device=mask.device), indexing='ij'
                ), dim=-1).reshape(-1, 2).float()
                dists = th.cdist(all_coords, coords).min(dim=1).values
                return dists.mean().item()

            D = compute_D(mask) if mode in ('D', 'AD') else None

            def u_fn(j):
                A = j / t_T
                return get_U(U_min=U_min, U_max=U_max, A=A, D=D, mode=mode)

            jump_params['jump_n_sample'] = u_fn

        times = get_schedule_jump(**jump_params)
        pairs = list(zip(times[:-1], times[1:]))
        if progress:
            from tqdm.auto import tqdm; pairs = tqdm(pairs)
        for t_last, t_cur in pairs:
            tb = th.tensor([t_last] * shape[0], device=device)
            if t_cur < t_last:
                with th.no_grad():
                    out = self.p_sample(model, img, tb, clip_denoised, model_kwargs, conf, pred_xstart)
                    img, pred_xstart = out["sample"], out["pred_xstart"]
            else:
                img = self.undo(img, tb + conf.get('inpa_inj_time_shift', 1))
        return out if return_all else out["sample"]

    def _scale_timesteps(self, t): return t


class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        betas = kwargs["betas"]
        ac = np.cumprod(1.0 - betas, axis=0)
        self.timestep_map, new_betas, last = [], [], 1.0
        for i, a in enumerate(ac):
            if i in set(use_timesteps):
                new_betas.append(1 - a / last); last = a
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *a, **kw):
        return super().p_mean_variance(self._wrap(model), *a, **kw)

    def _wrap(self, model):
        if isinstance(model, _W): return model
        return _W(model, self.timestep_map)

class _W:
    def __init__(self, m, tmap): self.m, self.tmap = m, tmap
    def __call__(self, x, ts, **kw):
        return self.m(x, th.tensor(self.tmap, device=ts.device, dtype=ts.dtype)[ts], **kw)
