import torch
import torch.nn.functional as F
from torch import nn

from config import config_dict
from ppm_tools import ppm_to_frequency


# ----- Encoder ----
class ConvBlock(nn.Module):
    def __init__(self, in_chann, out_chann):
        super().__init__()
        self.conv1 = self.make_conv_block(in_chann, out_chann, kernel_size=3, stride=1)
        self.conv2 = self.make_conv_block(out_chann, out_chann, kernel_size=4, stride=2)

        if in_chann == out_chann:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(in_chann, out_chann, kernel_size=1, padding=0)

    def make_conv_block(self, in_chann, out_chann, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv1d(in_chann, out_chann, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_chann),
            nn.GELU(),
        )

    def forward(self, x):
        residue = x
        x = self.conv1(x)
        residue = self.residual_layer(residue)
        x = x + residue
        x = self.conv2(x)  # reducing dimension
        return x


class Encoder(nn.Module):
    def __init__(self, in_chann=2, hidden_dim=config_dict['encoder_hidden_dim']):
        super().__init__()
        self.enc_net = nn.Sequential(
            ConvBlock(in_chann, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            nn.Flatten()
        )

    def forward(self, z):
        x = torch.view_as_real(z)
        x = x.permute(0, 2, 1)
        x = self.enc_net(x)
        return x


# ----- Decoder ----
class Decoder(nn.Module):
    def __init__(self, n_heads=config_dict['decoder_n_heads'], device='cpu'):
        super().__init__()
        self.n_heads = n_heads  # number of decoder heads
        self.in_proj = nn.Linear(8*64, 4*n_heads)
        self.f1 = ppm_to_frequency(config_dict['p1'], config_dict['trn_freq'])
        self.f2 = ppm_to_frequency(config_dict['p2'], config_dict['trn_freq'])
        self.trn_freq = config_dict['trn_freq']
        t = self.gen_time_points(length=config_dict['T'], t_step=0.00036, device=device)  # shape: (T, 1)
        self.register_buffer('t', t)

    def vec_gauss(self, t, params):
        a, f, ph, d = params.transpose(2, 0)
        sig = a * torch.exp(2 * torch.pi * f * t * 1j) * torch.exp(ph * 1j) * torch.exp(-1 * (d**2) * (t**2))
        return sig.transpose(2, 0)

    def vec_lorntz(self, t, params):
        a, f, ph, d = params.transpose(2, 0)
        sig = a * torch.exp(2 * torch.pi * f * t * 1j) * torch.exp(ph * 1j) * torch.exp(-1 * d * t)
        return sig.transpose(2, 0)

    def gen_time_points(self, length=512, t_step=0.00036, device='cpu'):
        t = torch.arange(0, length) * t_step
        return t[:, None, None].to(device)

    def limit_params(self, latents):
        latents_copy = latents.clone()
        latents_copy[:, :, 0, :] = F.softplus(latents[:, :, 0, :])
        latents_copy[:, :, 1, :] = torch.clamp(latents[:, :, 1, :], self.f2, self.f1)
        latents_copy[:, :, 2, :] = torch.clamp(latents[:, :, 2, :], -torch.pi, torch.pi)
        latents_copy[:, :, 3, :] = torch.clamp(latents[:, :, 3, :], 0, 2*self.trn_freq)
        return latents_copy

    def forward(self, latents, verbose=False):
        B, _ = latents.shape
        latents = self.in_proj(latents)
        latents = latents.view([B, 1, 4, self.n_heads])  # latents shape: (B, 1, 4, n_heads)
        # amir2
        latents = self.limit_params(latents)  # constraints on parameters

        if verbose:  # if verbose=True, print the parameters. [for debug purposes!]
            print(latents.shape)
            print(latents)

        # amir2
        line_shapes_g = self.vec_gauss(self.t, latents[..., :self.n_heads//2]).sum(dim=0)
        line_shapes_l = self.vec_lorntz(self.t, latents[..., self.n_heads//2:]).sum(dim=0)
        return line_shapes_g + line_shapes_l  # output shape: (B, T) | dtype: torch.complex64


# ----- AutoEncoder -----
class AutoEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder(device=device).to(device)

        self.apply(self._init_weights)

    def forward(self, z, verbose=False):
        latents = self.encoder(z)
        z_rec = self.decoder(latents, verbose=verbose)
        return z_rec

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def configure_optimizer(self, weight_decay=1e-4, learning_rate=1e-3):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},  # no weight decay for biases
        ]

        optimizer = torch.optim.AdamW(optim_groups, weight_decay=1e-2, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        return optimizer


if __name__ == '__main__':
    encoder = Encoder()
    size = (10, 512)
    z = torch.rand(size=size, dtype=torch.complex64)
    latents = encoder(z)
    assert tuple(latents.shape) == size

    decoder = Decoder()
    z_rec = decoder(latents, verbose=False)
    assert tuple(z_rec.shape) == size

    autoencoder = AutoEncoder()
    z_rec = autoencoder(z)
    assert tuple(z_rec.shape) == size
