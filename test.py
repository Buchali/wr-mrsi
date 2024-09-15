import time

import torch
from torch.nn import functional as F

from config import config_dict
from ppm_tools import ppm_to_point_index


def test(autoencoder, test_dataloader, device='cpu'):
    # params
    T = config_dict['T']
    p1 = config_dict['p1']
    p2 = config_dict['p2']
    t_step = config_dict['t_step']
    trn_freq = config_dict['trn_freq']

    # freq ranges
    wr_freq_range = slice(ppm_to_point_index(p2, T, t_step, trn_freq), ppm_to_point_index(p1, T, t_step, trn_freq))  # water removal frequency range.
    # plt_freq_range = slice(ppm_to_point_index(7, T, t_step, trn_freq), ppm_to_point_index(1, T, t_step, trn_freq))  # water removal frequency range.

    z_res_list = []
    loss_accum = 0
    with torch.no_grad():
        autoencoder.eval()
        t1 = time.time()
        for z_test in test_dataloader:
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

        # concatenate
        z_test_res_f = torch.cat(z_res_list, dim=0)
        print(f"test_time: {t2 - t1}")
        print(f'test loss: {loss:.4f}')
        print(z_test_res_f.shape)
        return z_test_res_f
