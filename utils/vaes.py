import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(self, sample_x, hidden_dims, z_dim):
        super().__init__()

        modules = []
        self.intermediate_hw = []

        if hidden_dims is None:
            self.hidden_dims = [8, 16, 32, 64]

        # Build Encoder
        B, in_channels, out_h, out_w = sample_x.shape
        self.intermediate_hw.append((out_h, out_w))
        K, S, P = 3, 2, 1
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size = K, stride= S, padding = P),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )


            out_h = int(((out_h - K + (2*P)) / S) + 1)
            out_w = int(((out_w - K + (2*P)) / S) + 1)
            self.intermediate_hw.append((out_h, out_w))

            in_channels = h_dim

        self.encoder_layers = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(self.hidden_dims[-1]*out_h*out_w, z_dim) # [B, Z_dim]
        self.fc_var = nn.Linear(self.hidden_dims[-1]*out_h*out_w, z_dim) # [B, Z_dim]

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(z_dim, self.hidden_dims[-1] * out_h * out_w)

        self.hidden_dims.reverse()


        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=K,
                                       stride = S,
                                       padding = P,
                                       output_padding = 1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder_layers = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(self.hidden_dims[-1], out_channels=1,
                                      kernel_size= 7, padding= 3),
                            nn.BatchNorm2d(1)
                            # nn.Softmax(dim=2)
                            )
        
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
        H, W = self.intermediate_hw[-1]
        C = self.hidden_dims[0]
        h = h.reshape(-1, C, H, W)
        h = self.decoder_layers(h)
        h = self.final_layer(h)
        return h
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
