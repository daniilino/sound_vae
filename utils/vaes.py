import torch.nn as nn
import torch.nn.functional as F
import torch

from typing import Tuple


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_size: Tuple[int, int]):
        super().__init__()
        self.out_size = out_size
        self.conv = nn.Conv2d(
            in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.interpolate(x, size=self.out_size)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x
    

class DecoderHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding='same'
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding='same'
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.LeakyReLU()
        self.final_act = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.final_act(x)
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding='same'):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()
    

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class VAE(nn.Module):
    def __init__(self, shape, z_dim):
        super().__init__()

        encoder_modules = []
        self.intermediate_hw = []

        self.encoder_dims = [1, 2, 4, 8, 16]
        self.decoder_dims = [256, 128, 64, 32, 16]

        # Build Encoder
        K, S, P = 3, 2, 1
        out_h, out_w = shape
        self.intermediate_hw.append((out_h, out_w))
        for i in range(len(self.encoder_dims) - 1):
            encoder_modules.append(
                EncoderBlock(
                    in_channels=self.encoder_dims[i],
                    out_channels=self.encoder_dims[i + 1],
                    kernel_size=K,
                    stride=S,
                    padding=P
                )
            )

            out_h = int(((out_h - K + (2 * P)) / S) + 1)
            out_w = int(((out_w - K + (2 * P)) / S) + 1)
            # (64 - 7 + 3) / 1 + 1
            self.intermediate_hw.append((out_h, out_w))

        self.encoder_layers = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Linear(
            self.encoder_dims[-1] * out_h * out_w, z_dim
        )  # [B, Z_dim]
        self.fc_var = nn.Linear(
            self.encoder_dims[-1] * out_h * out_w, z_dim
        )  # [B, Z_dim]

        # Build Decoder
        decoder_modules = []
        self.intermediate_hw.reverse()
        self.decoder_input = nn.Linear(z_dim, self.decoder_dims[0] * out_h * out_w)

        print(self.encoder_dims)
        print(self.decoder_dims)
        print(self.intermediate_hw)
        for i in range(len(self.decoder_dims) - 1):
            decoder_modules.append(
                DecoderBlock(
                    in_channels=self.decoder_dims[i],
                    out_channels=self.decoder_dims[i + 1],
                    out_size=self.intermediate_hw[i + 1],
                )
            )

        self.decoder_layers = nn.Sequential(*decoder_modules)
        self.final_layer = DecoderHead(self.decoder_dims[-1], 1)


    def encoder(self, x):
        h = self.encoder_layers(x)
        h = h.reshape(h.shape[0], -1)
        return self.fc_mu(h), self.fc_var(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample = g(x, eps)

    def decoder(self, z):
        h = self.decoder_input(z)
        H, W = self.intermediate_hw[0]
        C = self.decoder_dims[0]
        h = h.reshape(-1, C, H, W)
        h = self.decoder_layers(h)
        h = self.final_layer(h)
        return h

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
