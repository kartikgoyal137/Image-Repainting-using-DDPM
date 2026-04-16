"""DDPM Gaussian Diffusion with RePaint inpainting + Dynamic U resampling."""

import numpy as np
import torch as th
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Beta schedule
# ---------------------------------------------------------------------------

def get_named_beta_schedule(steps=1000):
    s = 1000 / steps
    return np.linspace(s * 0.0001, s * 0.02, steps, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dynamic U
# ---------------------------------------------------------------------------

def get_dynamic_U(j, t_T, U_min=1, U_max=10, mode='A', D=None):
    """
    Compute the number of resampling steps U for a given jump position j.

    Args:
        j:     Current jump position (timestep index in the schedule).
        t_T:   Total number of timesteps T (used to compute fractional progress A).
        U_min: Minimum resampling count.
        U_max: Maximum resampling count.
        mode:  One of:
                 'A'  — score based on timestep fraction (how early in denoising we are).
                 'D'  — score based on boundary distance of the mask.
                 'AD' — weighted combination of A and D.
        D:     Mean boundary distance (pixels). Required when mode is 'D' or 'AD'.
               If None, falls back to mode='A'.

    Returns:
        Integer U in [U_min, U_max].
    """
    A = j / t_T  # fractional progress: 1.0 near start, 0.0 near end

    if mode == 'A':
        score = A
    elif mode == 'D':
        if D is None:
            raise ValueError("D must be provided for mode='D'")
        score = min(D / 50.0, 1.0)
    elif mode == 'AD':
        if D is None:
            raise ValueError("D must be provided for mode='AD'")
        score = 0.5 * (A ** 0.5) + 0.5 * min(D / 50.0, 1.0)
    else:
        raise ValueError(f"Unknown dynamic_U mode: {mode}")

    return U_min + int((U_max - U_min) * score)


def compute_boundary_distance(mask):
    """
    Compute the mean distance (in pixels) from every pixel to the nearest
    mask boundary. Used as the D signal for dynamic_U mode='D' or 'AD'.

    Args:
        mask: Float tensor of shape (1, 1, H, W) or (1, C, H, W), values in [0, 1].
    Returns:
        Scalar float.
    """
    kernel = th.ones(1, 1, 3, 3, device=mask.device)
    m = (mask[:, 0:1] > 0.5).float()
    eroded = F.conv2d(m, kernel, padding=1)
    boundary = ((eroded < 9) & (m > 0.5)).float()

    if boundary.sum() == 0:
        return 50.0  # no boundary found — saturate D so score -> 1

    coords = boundary[0, 0].nonzero(as_tuple=False).float()
    all_coords = th.stack(th.meshgrid(
        th.arange(m.shape[-2], device=mask.device),
        th.arange(m.shape[-1], device=mask.device),
        indexing='ij'
    ), dim=-1).reshape(-1, 2).float()

    return th.cdist(all_coords, coords).min(dim=1).values.mean().item()


# ---------------------------------------------------------------------------
# RePaint schedule with dynamic U
# ---------------------------------------------------------------------------

def get_schedule_jump(t_T, jump_length, jump_n_sample):
    """
    Build the RePaint timestep sequence. At each jump position, the schedule
    steps backward (denoise) and then forward (re-noise) jump_n_sample times
    before continuing, enabling the known region to condition the unknown region.

    Args:
        t_T:           Total timesteps T.
        jump_length:   How many steps each jump covers (j in the paper).
        jump_n_sample: Fixed int U, or a callable (j) -> U for dynamic U.

    Returns:
        List of integer timesteps ending with -1 as a sentinel.
    """
    def resolve_u(j):
        return jump_n_sample(j) if callable(jump_n_sample) else jump_n_sample

    # Pre-compute how many extra resampling rounds remain at each jump position
    jumps = {j: resolve_u(j) - 1
             for j in range(0, t_T - jump_length, jump_length)}

    t, ts = t_T, []
    while t >= 1:
        t -= 1
        ts.append(t)

        # Standard n=1 harmonic resampling (always done once)
        if t + 1 < t_T - 1:
            t += 1; ts.append(t)
            t -= 1; ts.append(t)

        # Extra jump resampling rounds
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t += 1; ts.append(t)

    ts.append(-1)
    return ts


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def extract(arr, timesteps, shape):
    """Index a 1-D numpy array by timesteps and broadcast to tensor shape."""
    values = th.from_numpy(arr).to(timesteps.device)[timesteps].float()
    while values.ndim < len(shape):
        values = values[..., None]
    return values.expand(shape)


# ---------------------------------------------------------------------------
# Gaussian Diffusion — vanilla DDPM + RePaint inpainting + dynamic U
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """
    DDPM diffusion process with RePaint inpainting schedule.

    Expected model_kwargs:
        gt            — ground truth image tensor (B, C, H, W), values in [-1, 1]
        gt_keep_mask  — binary mask tensor (B, C, H, W), 1 = known, 0 = unknown
    """

    def __init__(self, betas):
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        ac = self.alphas_cumprod = np.cumprod(alphas, axis=0)
        ac_prev = np.append(1.0, ac[:-1])

        self.sqrt_recip_ac   = np.sqrt(1.0 / ac)
        self.sqrt_recipm1_ac = np.sqrt(1.0 / ac - 1)

        pv = self.posterior_variance = betas * (1.0 - ac_prev) / (1.0 - ac)
        self.post_log_var = np.log(np.append(pv[1], pv[1:]))
        self.post_coef1   = betas * np.sqrt(ac_prev) / (1.0 - ac)
        self.post_coef2   = (1.0 - ac_prev) * np.sqrt(alphas) / (1.0 - ac)

    # ------------------------------------------------------------------
    # Core denoising step
    # ------------------------------------------------------------------

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        """Run the model and return (posterior mean, log variance, pred x0)."""
        B, C = x.shape[:2]
        out = model(x, t, **(model_kwargs or {}))
        eps, var_val = th.split(out, C, dim=1)  # model predicts noise + variance

        min_log = extract(self.post_log_var,   t, x.shape)
        max_log = extract(np.log(self.betas),  t, x.shape)
        # Interpolate between min and max log variance
        log_var = ((var_val + 1) / 2) * max_log + ((1 - var_val) / 2 + 0.5) * min_log

        pred_x0 = (extract(self.sqrt_recip_ac, t, x.shape) * x
                   - extract(self.sqrt_recipm1_ac, t, x.shape) * eps)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)

        mean = (extract(self.post_coef1, t, x.shape) * pred_x0
                + extract(self.post_coef2, t, x.shape) * x)
        return mean, log_var, pred_x0

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, pred_xstart=None):
        """
        Single reverse diffusion step with RePaint known-pixel injection.

        Before each denoising step, the known pixels in x are replaced with
        properly noised versions of the ground truth. This lets the known region
        guide the inpainting through the diffusion process.
        """
        gt   = model_kwargs['gt']
        mask = model_kwargs['gt_keep_mask']

        # RePaint known-pixel injection at current noise level
        if pred_xstart is not None:
            ac = extract(self.alphas_cumprod, t, x.shape)
            noised_gt = th.sqrt(ac) * gt + th.sqrt(1 - ac) * th.randn_like(x)
            x = mask * noised_gt + (1 - mask) * x

        mean, log_var, pred_x0 = self.p_mean_variance(model, x, t, clip_denoised, model_kwargs)

        noise = th.randn_like(x)
        not_t0 = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        sample = mean + not_t0 * th.exp(0.5 * log_var) * noise

        return {"sample": sample, "pred_xstart": pred_x0, "gt": gt}

    def undo(self, img, t):
        """One forward noising step — used during jump-back resampling."""
        beta = extract(self.betas, t, img.shape)
        return th.sqrt(1 - beta) * img + th.sqrt(beta) * th.randn_like(img)

    # ------------------------------------------------------------------
    # Full sampling loop with RePaint schedule + dynamic U
    # ------------------------------------------------------------------

    def p_sample_loop(self, model, shape, model_kwargs=None, device=None,
                      clip_denoised=True, progress=True,
                      jump_length=10, jump_n_sample=10,
                      dynamic_U=None):
        """
        Run the full RePaint sampling loop.

        Args:
            model:         Denoising model.
            shape:         Output shape (B, C, H, W).
            model_kwargs:  Must contain 'gt' and 'gt_keep_mask'.
            device:        Torch device.
            clip_denoised: Clip predicted x0 to [-1, 1].
            progress:      Show tqdm progress bar.
            jump_length:   Number of steps per jump (j in paper). Default: 10.
            jump_n_sample: Fixed U (int) or callable (j) -> U. Default: 10.
            dynamic_U:     Optional dict to override jump_n_sample with dynamic U:
                             {
                               'mode':  'A' | 'D' | 'AD',
                               'U_min': int,   # default 1
                               'U_max': int,   # default 10
                             }
                           When provided, jump_n_sample is ignored.

        Returns:
            Dict with keys 'sample', 'pred_xstart', 'gt'.
        """
        device = device or next(model.parameters()).device
        img = th.randn(*shape, device=device)
        pred_xstart = None
        t_T = self.num_timesteps

        # Build the U function
        if dynamic_U is not None:
            mode  = dynamic_U.get('mode', 'A')
            U_min = dynamic_U.get('U_min', 1)
            U_max = dynamic_U.get('U_max', 10)
            D = (compute_boundary_distance(model_kwargs['gt_keep_mask'])
                 if mode in ('D', 'AD') else None)

            def u_fn(j):
                return get_dynamic_U(j, t_T, U_min=U_min, U_max=U_max, mode=mode, D=D)

            jump_n_sample = u_fn

        times = get_schedule_jump(t_T, jump_length, jump_n_sample)
        pairs = list(zip(times[:-1], times[1:]))

        if progress:
            from tqdm.auto import tqdm
            pairs = tqdm(pairs)

        out = None
        for t_last, t_cur in pairs:
            tb = th.tensor([t_last] * shape[0], device=device)
            with th.no_grad():
                if t_cur < t_last:
                    # Denoising step
                    out = self.p_sample(model, img, tb, clip_denoised, model_kwargs, pred_xstart)
                    img, pred_xstart = out["sample"], out["pred_xstart"]
                else:
                    # Jump-back: re-noise by one step
                    img = self.undo(img, tb + 1)

        return out