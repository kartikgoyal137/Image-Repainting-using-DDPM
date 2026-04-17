from guided_diffusion.unet import UNetModel
from guided_diffusion.diffusion import (
    SpacedDiffusion, space_timesteps, get_named_beta_schedule,
)

NUM_CLASSES = 1000

def create_model(conf):
    ch_mult = conf.channel_mult
    if ch_mult == "":
        ch_mult = {512: (0.5,1,1,2,2,4,4), 256: (1,1,2,2,4,4),
                    128: (1,1,2,3,4), 64: (1,2,3,4)}[conf.image_size]
    elif not isinstance(ch_mult, tuple):
        ch_mult = tuple(int(x) for x in ch_mult.split(","))

    att_ds = [conf.image_size // int(r) for r in conf.attention_resolutions.split(",")]

    return UNetModel(
        image_size=conf.image_size, in_channels=3, model_channels=conf.num_channels,
        out_channels=6 if conf.learn_sigma else 3,
        num_res_blocks=conf.num_res_blocks, attention_resolutions=tuple(att_ds),
        dropout=conf.dropout or 0, channel_mult=ch_mult,
        num_classes=NUM_CLASSES if conf.class_cond else None,
        use_checkpoint=conf.use_checkpoint or False, use_fp16=conf.use_fp16 or False,
        num_heads=conf.num_heads or 1, num_head_channels=conf.num_head_channels or -1,
        num_heads_upsample=conf.num_heads_upsample or -1,
        use_scale_shift_norm=conf.use_scale_shift_norm or False,
        resblock_updown=conf.resblock_updown or False, conf=conf,
    )


def create_diffusion(conf):
    betas = get_named_beta_schedule(conf.diffusion_steps or 1000)
    ts_respace = conf.timestep_respacing or [conf.diffusion_steps or 1000]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(conf.diffusion_steps or 1000, ts_respace),
        betas=betas, learn_sigma=conf.learn_sigma or False,
        rescale_timesteps=conf.rescale_timesteps or False, conf=conf,
    )
