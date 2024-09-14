import math
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn

from config import config_dict
from constants import TRAIN_SAMPLE_FILENAME, TRAIN_SAMPLE_URL
from data_utils import create_loaders, download_data
from model import AutoEncoder
from ppm_tools import ppm_to_point_index
from utils import (filter_low_power, normalize, plot_dual_freq, plot_timefreq,
                   save_checkpoint, load_checkpoint)
from datetime import datetime


# params
T = config_dict['T']
batch_size = config_dict['batch_size']
p1 = config_dict['p1']
p2 = config_dict['p2']
t_step = config_dict['t_step']
trn_freq = config_dict['trn_freq']
training_epochs = config_dict['training_epochs']
training_report_step = config_dict['training_report_step']

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

wr_freq_range = slice(ppm_to_point_index(p2, T, t_step, trn_freq), ppm_to_point_index(p1, T, t_step, trn_freq))  # water removal frequency range.
plt_freq_range = slice(ppm_to_point_index(7, T, t_step, trn_freq), ppm_to_point_index(1, T, t_step, trn_freq))  # water removal frequency range.

# load and preprocess
z = download_data(TRAIN_SAMPLE_FILENAME, TRAIN_SAMPLE_URL)
z_normalized = normalize(z[:T])
# plot_timefreq(z_normalized.T)
z_filtered = filter_low_power(z_normalized)

# split data and create dataloaders
z = torch.tensor(z_filtered).to(device)
train_dataloader, val_dataloader, test_dataloader = create_loaders(z, batch_size=batch_size)

# create the model
autoencoder = AutoEncoder().to(device)
params = sum(p.numel() for p in autoencoder.parameters())
print(f'number of parameters: {params}')


# scheduler
def get_lr(step, max_lr=1e-2, min_lr=1e-4, warmup_epochs=5, max_epochs=100):
    if step < warmup_epochs:
        return max_lr * (step + 1) / warmup_epochs

    if step > max_epochs:
        return min_lr

    decay_ratio = (step - warmup_epochs) / (max_epochs - warmup_epochs)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

opt = autoencoder.configure_optimizer()
autoencoder, optimizer, start_epoch = load_checkpoint(model=autoencoder, optimizer=opt, device=device)

def validate():
    with torch.no_grad():
        autoencoder.eval()
        for z_val in val_dataloader:
            z_val_rec = autoencoder(z_val)

            # freq
            z_val_f = torch.fft.fftshift(torch.fft.fft(z_val, dim=-1), dim=-1)
            z_val_f_real = torch.view_as_real(z_val_f)
            z_val_rec_f = torch.fft.fftshift(torch.fft.fft(z_val_rec, dim=-1), dim=-1)
            z_val_rec_f_real = torch.view_as_real(z_val_rec_f)

            # loss and plot
            val_loss = F.mse_loss(z_val_rec_f_real[:, wr_freq_range, :], z_val_f_real[:, wr_freq_range, :])
            print(f"validation loss: {val_loss:.4f}")
            print('validation plots')
            plot_dual_freq(z_val_f, z_val_rec_f, plt_freq_range)
        autoencoder.train()


# training loop
cur_step = 0
accum_loss = 0

autoencoder.train()
mse_loss = nn.MSELoss()

print(f'number of lineshape is {autoencoder.decoder.n_heads}')
for epoch in tqdm(range(start_epoch, training_epochs)):
    t1 = time.time()
    for z in train_dataloader:
        z = z.to(device)
        autoencoder.zero_grad()
        # time domain
        z_rec = autoencoder(z, verbose=False)
        # freq domain
        z_f = torch.view_as_real(torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1))
        z_rec_f = torch.view_as_real(torch.fft.fftshift(torch.fft.fft(z_rec, dim=-1), dim=-1))

        # loss
        loss = mse_loss(z_rec_f[:, wr_freq_range, :], z_f[:, wr_freq_range, :])
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0) # stop great gradient shocks of high loss
        lr = get_lr(epoch, max_lr=1e-2, min_lr=3e-5, warmup_epochs=5, max_epochs=training_epochs)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        opt.step()

        accum_loss += loss.detach()
        cur_step += 1
    t2 = time.time()
    # ----- reports ----
    with torch.no_grad():
        if epoch % training_report_step == 0:
            print(30 * '-')
            print(f'epoch {epoch:>3} | loss: {accum_loss.item()/cur_step:.4f} | epoch time: {t2-t1:.2f}s | lr: {lr:.3e}')

            accum_loss = 0
            cur_step = 0

            ## visualize training
            # plot_dual_freq(z_f, z_rec_f)

            # --- validate ---
            validate()

            # save checkpoint
            # -----------------
            save_checkpoint(autoencoder, opt, config_dict, epoch=epoch)
    # break
