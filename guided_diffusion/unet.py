
from abc import abstractmethod
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F



class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def norm(channels):
    return GroupNorm32(32, channels)

def conv2d(in_ch, out_ch, kernel, **kwargs):
    return nn.Conv2d(in_ch, out_ch, kernel, **kwargs)

def conv1d(in_ch, out_ch, kernel):
    return nn.Conv1d(in_ch, out_ch, kernel)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = th.exp(
        -math.log(10000) * th.arange(half, dtype=th.float32) / half
    ).to(timesteps.device)
    angles = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(angles), th.sin(angles)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x



class Upsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None, **kw):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv2d(channels, out_channels or channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None, **kw):
        super().__init__()
        out = out_channels or channels
        self.op = conv2d(channels, out, 3, stride=2, padding=1) if use_conv else nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.op(x)



class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 use_conv=False, use_scale_shift_norm=False, up=False, down=False, **kw):
        super().__init__()
        out_ch = self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            norm(channels), nn.SiLU(), conv2d(channels, out_ch, 3, padding=1)
        )

        self.updown = up or down
        if up:
            self.h_upd = self.x_upd = Upsample(channels, use_conv=False)
        elif down:
            self.h_upd = self.x_upd = Downsample(channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        emb_out_ch = 2 * out_ch if use_scale_shift_norm else out_ch
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, emb_out_ch))

        self.out_layers = nn.Sequential(
            norm(out_ch), nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(conv2d(out_ch, out_ch, 3, padding=1))
        )

        if out_ch == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv2d(channels, out_ch, 3, padding=1)
        else:
            self.skip_connection = conv2d(channels, out_ch, 1)

    def forward(self, x, emb):
        if self.updown:
            h = self.in_layers[:-1](x)   
            h = self.in_layers[-1](self.h_upd(h))  
            x = self.x_upd(x)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift  
            h = self.out_layers[1:](h)                        
        else:
            h = self.out_layers(h + emb_out)

        return self.skip_connection(x) + h



class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, **kw):
        super().__init__()
        self.num_heads = channels // num_head_channels if num_head_channels != -1 else num_heads
        self.norm = norm(channels)
        self.qkv = conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(conv1d(channels, channels, 1))

    def forward(self, x):
        B, C, *spatial = x.shape
        x_flat = x.reshape(B, C, -1)

        qkv = self.qkv(self.norm(x_flat))
        head_ch = C // self.num_heads
        q, k, v = qkv.reshape(B * self.num_heads, head_ch * 3, -1).split(head_ch, dim=1)

        scale = 1 / math.sqrt(math.sqrt(head_ch))
        attn = th.softmax(
            th.einsum("bct,bcs->bts", q * scale, k * scale).float(), dim=-1
        ).type(q.dtype)

        h = th.einsum("bts,bcs->bct", attn, v).reshape(B, C, -1)
        return (x_flat + self.proj_out(h)).reshape(B, C, *spatial)



class UNetModel(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels,
                 num_res_blocks, attention_resolutions, dropout=0,
                 channel_mult=(1, 2, 4, 8), conv_resample=True, num_classes=None,
                 use_fp16=False, num_heads=1, num_head_channels=-1,
                 num_heads_upsample=-1, use_scale_shift_norm=False,
                 resblock_updown=False, **kw):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.model_channels = model_channels
        self.num_classes = num_classes
        self.dtype = th.float16 if use_fp16 else th.float32

        emb_ch = model_channels * 4
        def make_res(in_ch, out_ch, **kwargs):
            return ResBlock(in_ch, emb_ch, dropout, out_channels=out_ch,
                            use_scale_shift_norm=use_scale_shift_norm, **kwargs)
        def make_attn(ch, heads):
            return AttentionBlock(ch, num_heads=heads, num_head_channels=num_head_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch)
        )
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, emb_ch)

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv2d(in_channels, ch, 3, padding=1))
        ])
        input_block_channels = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            out_ch = int(mult * model_channels)
            for _ in range(num_res_blocks):
                layers = [make_res(ch, out_ch)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(make_attn(ch, num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(ch)

            if level != len(channel_mult) - 1:
                down = make_res(ch, ch, down=True) if resblock_updown else Downsample(ch, conv_resample, out_channels=ch)
                self.input_blocks.append(TimestepEmbedSequential(down))
                input_block_channels.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            make_res(ch, ch), make_attn(ch, num_heads), make_res(ch, ch)
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = int(model_channels * mult)
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_channels.pop()
                layers = [make_res(ch + skip_ch, out_ch)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(make_attn(ch, num_heads_upsample))
                if level and i == num_res_blocks:
                    up = make_res(ch, ch, up=True) if resblock_updown else Upsample(ch, conv_resample, out_channels=ch)
                    layers.append(up)
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm(ch), nn.SiLU(),
            zero_module(conv2d(int(channel_mult[0] * model_channels), out_channels, 3, padding=1))
        )

    def convert_to_fp16(self):
        def cast(layer):
            if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
                layer.weight.data = layer.weight.data.half()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.half()
        self.input_blocks.apply(cast)
        self.middle_block.apply(cast)
        self.output_blocks.apply(cast)

    def forward(self, x, timesteps, y=None, gt=None, **kwargs):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        skips = []
        h = x.type(self.dtype)
        for block in self.input_blocks:
            h = block(h, emb)
            skips.append(h)

        h = self.middle_block(h, emb)

        for block in self.output_blocks:
            h = block(th.cat([h, skips.pop()], dim=1), emb)

        return self.out(h.type(x.dtype))