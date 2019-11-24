from .base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.latent_dim = args.vae_latent_dim

        # Input dropout
        self.input_dropout = nn.Dropout(p=args.vae_dropout)

        # Construct a list of dimensions for the encoder and the decoder
        dims = [args.vae_hidden_dim] * 2 * args.vae_num_hidden
        dims = [args.num_items] + dims + [args.vae_latent_dim * 2]

        # Stack encoders and decoders
        encoder_modules, decoder_modules = [], []
        for i in range(len(dims)//2):
            encoder_modules.append(nn.Linear(dims[2*i], dims[2*i+1]))
            if i == 0:
                decoder_modules.append(nn.Linear(dims[-1]//2, dims[-2]))
            else:
                decoder_modules.append(nn.Linear(dims[-2*i-1], dims[-2*i-2]))
        self.encoder = nn.ModuleList(encoder_modules)
        self.decoder = nn.ModuleList(decoder_modules)

        # Initialize weights
        self.encoder.apply(self.weight_init)
        self.decoder.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()

    @classmethod
    def code(cls):
        return 'vae'

    def forward(self, x):
        x = F.normalize(x)
        x = self.input_dropout(x)
        
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i != len(self.encoder) - 1:
                x = torch.tanh(x)
        
        mu, logvar = x[:, :self.latent_dim], x[:, self.latent_dim:]

        if self.training:
            # since log(var) = log(sigma^2) = 2*log(sigma)
            sigma = torch.exp(0.5 * logvar)
            eps = torch.randn_like(sigma)
            x = mu + eps * sigma
        else:
            x = mu

        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i != len(self.decoder) - 1:
                x = torch.tanh(x)
                
        return x, mu, logvar

