"""Configuration management for RePaint."""

import os
import yaml
from PIL import Image
from collections import defaultdict
from os.path import expanduser


def yamlread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return yaml.safe_load(f.read())


def imwrite(path=None, img=None):
    Image.fromarray(img).save(path)


def to_file_ext(img_names, ext):
    return [name.rsplit('.', 1)[0] + '.' + ext for name in img_names]


def write_images(imgs, img_names, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for image_name, image in zip(img_names, imgs):
        out_path = os.path.join(dir_path, image_name)
        imwrite(img=image, path=out_path)


class NoneDict(defaultdict):
    def __init__(self):
        super().__init__(self.return_None)

    @staticmethod
    def return_None():
        return None

    def __getattr__(self, attr):
        return self.get(attr)


class Default_Conf(NoneDict):
    def __init__(self):
        pass

    def get_dataloader(self, dset='train', dsName=None, batch_size=None, return_dataset=False):
        if batch_size is None:
            batch_size = self.batch_size

        ds_conf = self['data'][dset][dsName].copy()

        if ds_conf.get('mask_loader', False):
            from guided_diffusion.image_datasets import load_data_inpa
            return load_data_inpa(**ds_conf, conf=self)
        else:
            raise NotImplementedError()

    def eval_imswrite(self, srs=None, img_names=None, dset=None, name=None,
                      ext='png', lrs=None, gts=None, gt_keep_masks=None, verify_same=True):
        img_names = to_file_ext(img_names, ext)

        if dset is None:
            dset = self.get_default_eval_name()

        if srs is not None:
            write_images(srs, img_names, expanduser(self['data'][dset][name]['paths']['srs']))

        if gt_keep_masks is not None:
            write_images(gt_keep_masks, img_names, expanduser(self['data'][dset][name]['paths']['gt_keep_masks']))

        gts_path = self['data'][dset][name]['paths'].get('gts')
        if gts is not None and gts_path:
            write_images(gts, img_names, expanduser(gts_path))

        if lrs is not None:
            write_images(lrs, img_names, expanduser(self['data'][dset][name]['paths']['lrs']))

    def get_default_eval_name(self):
        candidates = self['data']['eval'].keys()
        if len(candidates) != 1:
            raise RuntimeError(f"Need exactly one candidate for {self.name}: {candidates}")
        return list(candidates)[0]

    def pget(self, name, default=None):
        names = name.split('.') if '.' in name else [name]
        sub_dict = self
        for n in names:
            sub_dict = sub_dict.get(n, default)
            if sub_dict is None:
                return default
        return sub_dict
