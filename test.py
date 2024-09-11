import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import config_dict
from constants import (subject_data_urls, subject_file_names, wr_data_urls,
                       wr_file_names)
from data_utils import load_data
from model import AutoEncoder
from ppm_tools import ppm_to_point_index
from utils import load_checkpoint, normalize, plot_dual_freq, compare_plot, plot_timefreq

# params
T = config_dict['T']
batch_size = config_dict['batch_size']
p1 = config_dict['p1']
p2 = config_dict['p2']
t_step = config_dict['t_step']
trn_freq = config_dict['trn_freq']

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

# freq ranges
wr_freq_range = slice(ppm_to_point_index(p2, T, t_step, trn_freq), ppm_to_point_index(p1, T, t_step, trn_freq))  # water removal frequency range.
plt_freq_range = slice(ppm_to_point_index(7, T, t_step, trn_freq), ppm_to_point_index(1, T, t_step, trn_freq))  # water removal frequency range.

# model
autoencoder = AutoEncoder()
opt = autoencoder.configure_optimizer()
autoencoder, optimizer, start_epoch = load_checkpoint(model=autoencoder, optimizer=opt, device=device)

# load test data
test_data = load_data(subject_file_names, subject_data_urls)
z_test = test_data[:, :T].T

# load wr data
test_wr_data = load_data(wr_file_names, wr_data_urls)
z_wr_test = test_wr_data[:, :T].T

# normalize
z_test_normalized, r_base = normalize(z_test[:T], return_base_values=True)
z_wr_test_normalized = normalize(z_wr_test[:T], r_base=r_base)


# to tensor
z_test = torch.tensor(z_test_normalized, dtype=torch.complex64)
z_wr_test = torch.tensor(z_wr_test_normalized, dtype=torch.complex64)

# dataloader
dataloader = DataLoader(z_test.to(device), batch_size=1000)

z_res_list = []
loss_accum = 0
with torch.no_grad():
    autoencoder.eval()
    t1 = time.time()
    for z_test in dataloader:
        z_test_rec = autoencoder(z_test, verbose=False)
        z_test_f = torch.fft.fftshift(torch.fft.fft(z_test, dim=-1), dim=-1)
        z_test_rec_f = torch.fft.fftshift(torch.fft.fft(z_test_rec, dim=-1), dim=-1)

        z_test_f_real = torch.view_as_real(z_test_f)
        z_test_rec_f_real = torch.view_as_real(z_test_rec_f)

        loss = F.mse_loss(z_test_rec_f_real[:, wr_freq_range, :], z_test_f_real[:, wr_freq_range, :])
        loss_accum += loss

        z_test_res_f = z_test_f - z_test_rec_f
        z_res_list.append(z_test_res_f)
    t2 = time.time()
    print(f"test_time: {t2 - t1}")
    # freq

    z_test_res_f = torch.cat(z_res_list, dim=0)
    print(z_test_res_f.shape)

    print(f'test loss: {loss:.4f}')

wr_dl = z_test_res_f.real.mean(axis=0)

z_wr_test_f = torch.fft.fftshift(torch.fft.fft(z_wr_test, dim=-1), dim=-1)
wr_svd = z_wr_test_f.real.mean(axis=0)

compare_plot(wr_dl, wr_svd, plt_freq_range)


