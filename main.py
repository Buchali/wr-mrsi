from test import test

import torch
from torch.utils.data import DataLoader

from config import config_dict
from constants import (TRAIN_SAMPLE_FILENAME, TRAIN_SAMPLE_URL,
                       subject_data_urls, subject_file_names, wr_data_urls,
                       wr_file_names)
from data_utils import create_datasets, download_data, load_data
from model import AutoEncoder
from ppm_tools import ppm_to_point_index
from train import train
from utils import compare_plot, filter_low_power, normalize

batch_size = config_dict['batch_size']
T = config_dict['T']
p1 = config_dict['p1']
p2 = config_dict['p2']
t_step = config_dict['t_step']
trn_freq = config_dict['trn_freq']

wr_freq_range = slice(ppm_to_point_index(p2, T, t_step, trn_freq), ppm_to_point_index(p1, T, t_step, trn_freq))  # water removal frequency range.
plt_freq_range = slice(ppm_to_point_index(7, T, t_step, trn_freq), ppm_to_point_index(1, T, t_step, trn_freq))  # water removal frequency range.

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

## train
# load and preprocess
z = download_data(TRAIN_SAMPLE_FILENAME, TRAIN_SAMPLE_URL)
z_normalized = normalize(z[:T])
# plot_timefreq(z_normalized.T)
z_filtered = filter_low_power(z_normalized)

# split data and create dataloaders
z = torch.tensor(z_filtered).to(device)
generator = torch.manual_seed(42)
train_dataset, val_dataset = create_datasets(z, ratio=[0.9, 0.1], generator=generator)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, generator=generator)
val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, generator=generator)

# create the model
autoencoder = AutoEncoder().to(device)
params = sum(p.numel() for p in autoencoder.parameters())
print(f'number of parameters: {params}')

autoencoder, opt = train(autoencoder, train_dataloader, val_dataloader, config_dict, device=device)

## test
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
test_dataloader = DataLoader(z_test.to(device), batch_size=1024)

z_test_res_f = test(autoencoder, test_dataloader, device=device)

# compare results
wr_dl = z_test_res_f.real.mean(axis=0)
z_wr_test_f = torch.fft.fftshift(torch.fft.fft(z_wr_test, dim=-1), dim=-1)
wr_svd = z_wr_test_f.real.mean(axis=0)

compare_plot(wr_dl, wr_svd, plt_freq_range)
