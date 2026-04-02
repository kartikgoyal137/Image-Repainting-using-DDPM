"""Minimal UNet model for diffusion (loads pretrained guided-diffusion weights)."""

from abc import abstractmethod
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def norm(ch): return GroupNorm32(32, ch)
def conv(ci, co, k, **kw): return nn.Conv2d(ci, co, k, **kw)
def conv1d(ci, co, k): return nn.Conv1d(ci, co, k)

def zero_module(m):
    for p in m.parameters(): p.detach().zero_()
    return m

def timestep_embedding(t, dim):
    half = dim // 2
    f = th.exp(-math.log(10000) * th.arange(half, dtype=th.float32) / half).to(t.device)
    a = t[:, None].float() * f[None]
    e = th.cat([th.cos(a), th.sin(a)], dim=-1)
    return th.cat([e, th.zeros_like(e[:, :1])], dim=-1) if dim % 2 else e

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb): pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, ch, use_conv, out_channels=None, **kw):
        super().__init__()
        self.use_conv = use_conv
        if use_conv: self.conv = conv(ch, out_channels or ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x

class Downsample(nn.Module):
    def __init__(self, ch, use_conv, out_channels=None, **kw):
        super().__init__()
        oc = out_channels or ch
        self.op = conv(ch, oc, 3, stride=2, padding=1) if use_conv else nn.AvgPool2d(2, 2)
    def forward(self, x): return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(self, ch, emb_ch, dropout, out_channels=None, use_conv=False,
                 use_scale_shift_norm=False, up=False, down=False, **kw):
        super().__init__()
        oc = self.out_channels = out_channels or ch
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(norm(ch), nn.SiLU(), conv(ch, oc, 3, padding=1))
        self.updown = up or down
        if up:    self.h_upd, self.x_upd = Upsample(ch, False), Upsample(ch, False)
        elif down: self.h_upd, self.x_upd = Downsample(ch, False), Downsample(ch, False)
        else:     self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, 2*oc if use_scale_shift_norm else oc))
        self.out_layers = nn.Sequential(norm(oc), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv(oc, oc, 3, padding=1)))
        if oc == ch:      self.skip_connection = nn.Identity()
        elif use_conv:     self.skip_connection = conv(ch, oc, 3, padding=1)
        else:              self.skip_connection = conv(ch, oc, 1)

    def forward(self, x, emb):
        if self.updown:
            h = self.in_layers[-1](self.h_upd(self.in_layers[:-1](x)))
            x = self.x_upd(x)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            s, sh = th.chunk(emb_out, 2, dim=1)
            h = self.out_layers[1:]( self.out_layers[0](h) * (1 + s) + sh)
        else:
            h = self.out_layers(h + emb_out)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=1, num_head_channels=-1, **kw):
        super().__init__()
        self.num_heads = ch // num_head_channels if num_head_channels != -1 else num_heads
        self.norm = norm(ch)
        self.qkv = conv1d(ch, ch * 3, 1)
        self.proj_out = zero_module(conv1d(ch, ch, 1))

    def forward(self, x):
        b, c, *sp = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        bs, w, l = qkv.shape
        ch = w // (3 * self.num_heads)
        q, k, v = qkv.reshape(bs * self.num_heads, ch * 3, l).split(ch, dim=1)
        sc = 1 / math.sqrt(math.sqrt(ch))
        wt = th.softmax(th.einsum("bct,bcs->bts", q * sc, k * sc).float(), dim=-1).type(q.dtype)
        h = th.einsum("bts,bcs->bct", wt, v).reshape(bs, -1, l)
        return (x + self.proj_out(h)).reshape(b, c, *sp)

class UNetModel(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels,
                 num_res_blocks, attention_resolutions, dropout=0,
                 channel_mult=(1,2,4,8), conv_resample=True, num_classes=None,
                 use_fp16=False, num_heads=1, num_head_channels=-1,
                 num_heads_upsample=-1, use_scale_shift_norm=False,
                 resblock_updown=False, **kw):
        super().__init__()
        if num_heads_upsample == -1: num_heads_upsample = num_heads
        self.model_channels = model_channels
        self.num_classes = num_classes
        self.dtype = th.float16 if use_fp16 else th.float32
        ted = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, ted), nn.SiLU(), nn.Linear(ted, ted))
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, ted)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv(in_channels, ch, 3, padding=1))])
        ibc = [ch]; ds = 1
        RB = lambda c, oc, **k: ResBlock(c, ted, dropout, out_channels=oc, use_scale_shift_norm=use_scale_shift_norm, **k)
        AB = lambda c: AttentionBlock(c, num_heads=num_heads, num_head_channels=num_head_channels)
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                oc = int(mult * model_channels)
                layers = [RB(ch, oc)]
                ch = oc
                if ds in attention_resolutions: layers.append(AB(ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers)); ibc.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(
                    RB(ch, ch, down=True) if resblock_updown else Downsample(ch, conv_resample, out_channels=ch)))
                ibc.append(ch); ds *= 2

        self.middle_block = TimestepEmbedSequential(RB(ch, ch), AB(ch), RB(ch, ch))
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = ibc.pop(); oc = int(model_channels * mult)
                layers = [RB(ch + ich, oc)]
                ch = oc
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads_upsample, num_head_channels=num_head_channels))
                if level and i == num_res_blocks:
                    layers.append(RB(ch, ch, up=True) if resblock_updown else Upsample(ch, conv_resample, out_channels=ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(norm(ch), nn.SiLU(), zero_module(conv(input_ch, out_channels, 3, padding=1)))

    def convert_to_fp16(self):
        def f(l):
            if isinstance(l, (nn.Conv1d, nn.Conv2d)):
                l.weight.data = l.weight.data.half()
                if l.bias is not None: l.bias.data = l.bias.data.half()
        self.input_blocks.apply(f); self.middle_block.apply(f); self.output_blocks.apply(f)

    def forward(self, x, timesteps, y=None, gt=None, **kwargs):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)
        h = x.type(self.dtype)
        for m in self.input_blocks: h = m(h, emb); hs.append(h)
        h = self.middle_block(h, emb)
        for m in self.output_blocks: h = m(th.cat([h, hs.pop()], dim=1), emb)
        return self.out(h.type(x.dtype))
