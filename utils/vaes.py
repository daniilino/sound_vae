import torch.nn as nn
import torch.nn.functional as F
import torch

from typing import Tuple

class Upsampler(nn.Module):

    def __init__(self, in_channels, out_channels, out_size: Tuple[int, int]):
        super().__init__()
        self.out_HW = (out_size[0]*2, out_size[1]*2)
        self.conv = nn.Conv2d(in_channels, out_channels=out_channels,
                              kernel_size = 3, stride= 2, padding = 1)

    def forward(self, x):
        h = F.interpolate(x, size = self.out_HW)
        h = self.conv(h)
        return h

class VAE(nn.Module):
    def __init__(self, sample_x, hidden_dims, z_dim):
        super().__init__()

        modules = []
        self.intermediate_hw = []

        self.encoder_dims = [8, 16, 32, 64]
        self.decoder_dims = [256, 128, 64, 32, 16]

        # Build Encoder
        B, in_channels, out_h, out_w = sample_x.shape
        self.intermediate_hw.append((out_h, out_w))
        K, S, P = 3, 2, 1
        for h_dim in self.encoder_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size = K, stride= S, padding = P),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )


            out_h = int(((out_h - K + (2*P)) / S) + 1)
            out_w = int(((out_w - K + (2*P)) / S) + 1)
            # (64 - 7 + 3) / 1 + 1
            self.intermediate_hw.append((out_h, out_w))

            in_channels = h_dim

        self.encoder_layers = nn.Sequential(*modules)
        
        
        self.fc_mu = nn.Linear(self.encoder_dims[-1]*out_h*out_w, z_dim) # [B, Z_dim]
        self.fc_var = nn.Linear(self.encoder_dims[-1]*out_h*out_w, z_dim) # [B, Z_dim]

        # Build Decoder
        modules = []

        self.decoder_dims = [dim * 2 for dim in self.encoder_dims[::-1]]
        self.decoder_input = nn.Linear(z_dim, self.decoder_dims[0] * out_h * out_w)

        self.intermediate_hw.reverse()

        # ####### DECONV #################
        # for i in range(len(self.hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(self.hidden_dims[i],
        #                                self.hidden_dims[i + 1],
        #                                kernel_size=K,
        #                                stride = S,
        #                                padding = P,
        #                                output_padding = 1),
        #             nn.BatchNorm2d(self.hidden_dims[i + 1]),
        #             nn.LeakyReLU())
        #     )
        # #########################################
        ############## UPSAMPLING ###################
        for i in range(len(self.decoder_dims) - 1):
            modules.append(
                nn.Sequential(
                    Upsampler(in_channels = self.decoder_dims[i], out_channels = self.decoder_dims[i+1], out_size = self.intermediate_hw[i+1]),
                    nn.BatchNorm2d(self.decoder_dims[i+1]),
                    nn.LeakyReLU())
            )
        ##########################################

        self.decoder_layers = nn.Sequential(*modules)

        # ####### DECONV #################
        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(self.hidden_dims[-1],
        #                                        self.hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(self.hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(self.hidden_dims[-1], out_channels=1,
        #                               kernel_size= 7, padding= 3),
        #                     nn.BatchNorm2d(1),
        #                     )
        # #########################################
        
        ############## UPSAMPLING ###################
        self.final_layer = nn.Sequential(
                            Upsampler(in_channels = self.decoder_dims[-1], out_channels = self.decoder_dims[-1], out_size = self.intermediate_hw[-1]),
                            nn.BatchNorm2d(self.decoder_dims[-1]),
                            nn.LeakyReLU(),
                            Upsampler(in_channels = self.decoder_dims[-1], out_channels = 1, out_size = self.intermediate_hw[-1]),
                            nn.BatchNorm2d(1),
                            )
        ##########################################
        
    def encoder(self, x):
        h = self.encoder_layers(x)
        h = h.reshape(h.shape[0], -1)
        return self.fc_mu(h), self.fc_var(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample = g(x, eps)
        
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
