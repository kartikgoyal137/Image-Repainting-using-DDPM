"""Script utilities for creating models and diffusion objects."""

from . import gaussian_diffusion as gd
from .gaussian_diffusion import SpacedDiffusion, space_timesteps
from .unet import UNetModel

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        predict_xstart=False,
        rescale_timesteps=False,
    )


def create_model_and_diffusion(
    image_size, class_cond, learn_sigma, num_channels, num_res_blocks,
    channel_mult, num_heads, num_head_channels, num_heads_upsample,
    attention_resolutions, dropout, diffusion_steps, noise_schedule,
    timestep_respacing, predict_xstart, rescale_timesteps,
    use_checkpoint, use_scale_shift_norm, resblock_updown, use_fp16,
    use_new_attention_order, conf=None, **kwargs
):
    model = create_model(
        image_size, num_channels, num_res_blocks,
        channel_mult=channel_mult, learn_sigma=learn_sigma,
        class_cond=class_cond, use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads, num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm, dropout=dropout,
        resblock_updown=resblock_updown, use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order, conf=conf
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps, learn_sigma=learn_sigma,
        noise_schedule=noise_schedule, predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        timestep_respacing=timestep_respacing, conf=conf
    )
    return model, diffusion


def create_model(image_size, num_channels, num_res_blocks, channel_mult="",
                 learn_sigma=False, class_cond=False, use_checkpoint=False,
                 attention_resolutions="16", num_heads=1, num_head_channels=-1,
                 num_heads_upsample=-1, use_scale_shift_norm=False, dropout=0,
                 resblock_updown=False, use_fp16=False,
                 use_new_attention_order=False, conf=None):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    elif isinstance(channel_mult, tuple):
        pass
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = [image_size // int(res) for res in attention_resolutions.split(",")]

    return UNetModel(
        image_size=image_size, in_channels=3, model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks, attention_resolutions=tuple(attention_ds),
        dropout=dropout, channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint, use_fp16=use_fp16,
        num_heads=num_heads, num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order, conf=conf
    )


def create_gaussian_diffusion(*, steps=1000, learn_sigma=False,
                              noise_schedule="linear", predict_xstart=False,
                              rescale_timesteps=False, timestep_respacing="",
                              conf=None):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            gd.ModelVarType.FIXED_LARGE if not learn_sigma else gd.ModelVarType.LEARNED_RANGE
        ),
        rescale_timesteps=rescale_timesteps,
        conf=conf
    )


def select_args(args_dict, keys):
    return {k: args_dict[k] for k in keys if k in args_dict}
