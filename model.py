import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(1, self.out_channels),
            # nn.GELU(),
            # nn.Conv2d(
            #     self.out_channels,
            #     self.out_channels,
            #     kernel_size=3,
            #     padding=1,
            #     bias=False
            # ),
            # nn.GroupNorm(1, self.out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.conv_block(x))
        else:
            return F.gelu(self.conv_block(x))


class DownResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(DownResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.in_channels, self.in_channels, residual=True),
            ConvBlock(self.in_channels, self.out_channels)
        )

        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_emb_dim,
                self.out_channels
            )
        )
        if self.in_channels == self.out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1
            )

    def forward(self, x, t_emb):
        x = self.maxpool_conv(x)
        emb_out = self.time_embedding(t_emb)

        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]

        return x + emb_out


class UpResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.conv_block = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels),
        )

        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t_emb):
        x = torch.cat([skip_x, x], dim=1)
        x = self.up(x)
        x = self.conv_block(x)

        emb_out = self.time_embedding(t_emb)

        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]

        return x + emb_out


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNetModel(nn.Module):
    def __init__(
            self,
            in_channels=3,
            model_channels=64,
            out_channels=3,
            num_res_blocks=2,
            channel_multiplier=(2, 2, 2)
    ):

        super(UNetModel, self).__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_multiplier

        self.time_embed_dim = model_channels * 4

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.time_embed_dim),
                nn.Linear(self.time_embed_dim, self.time_embed_dim),
                nn.ReLU()
            )

        self.input_block = ConvBlock(self.in_channels, self.model_channels)

        self.down_blocks = nn.ModuleList([])
        ch = model_channels

        for i, multiplier in enumerate(channel_multiplier):
            out = ch * multiplier
            layers = DownResBlock(
                in_channels=ch,
                out_channels=out,
                time_emb_dim=self.time_embed_dim,
                )
            ch = out

            self.down_blocks.append(layers)

        self.bot1 = ConvBlock(512, 1024)
        self.bot2 = ConvBlock(1024, 1024)
        self.bot3 = ConvBlock(1024, 512)

        self.up_blocks = nn.ModuleList([])

        for i, multiplier in enumerate(channel_multiplier[::-1]):
            out = ch//multiplier
            layers = UpResBlock(
                in_channels=ch*2,
                out_channels=out,
                time_emb_dim=self.time_embed_dim,
                )
            ch = out

            self.up_blocks.append(layers)

        self.output_block = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):

        t = self.time_mlp(t)
        x = self.input_block(x)

        # Unet
        residual_inputs = []
        for down in self.down_blocks:
            x = down(x, t)
            residual_inputs.append(x)

        x = self.bot1(x)
        x = self.bot2(x)
        x = self.bot3(x)

        for up in self.up_blocks:
            residual_x = residual_inputs.pop()
            x = up(x, residual_x, t)

        return self.output_block(x)

if __name__ == '__main__':
    net = UNetModel()
    x = torch.randn(3, 3, 256, 256)
    t = x.new_tensor([500] * x.shape[0])
    print(t)
    print(net(x, t).shape)
