"""RePaint: Inpainting using Denoising Diffusion Probabilistic Models."""

import os
import argparse
import torch as th
import io

import blobfile as bf
import conf_mgt
from conf_mgt.conf_base import yamlread

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    select_args,
)


def dev(device):
    if device is None:
        if th.cuda.is_available():
            return th.device("cuda")
        return th.device("cpu")
    return th.device(device)


def load_state_dict(path, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def toU8(sample):
    if sample is None:
        return sample
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample.detach().cpu().numpy()


def main(conf: conf_mgt.Default_Conf):
    print("Start", conf['name'])

    device = dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")

    dset = 'eval'
    eval_name = conf.get_default_eval_name()
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}
        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=device)
            model_kwargs["y"] = classes

        result = diffusion.p_sample_loop(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        srs = toU8(result['sample'])
        gts = toU8(result['gt'])
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))
        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)
