from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import config_dict
from data_utils import (create_datasets, load_data, remaining_subjects,
                        select_random_subjects)
from metrics import calculate_rmse_reconstruct_err, calculate_wpsr
from model import AutoEncoder
from ppm_tools import ppm_to_point_index
from testing import test
from training import train
from utils import (compare_plot, filter_low_power, load_checkpoint, normalize,
                   plot_freq, save_checkpoint)

batch_size = config_dict['batch_size']
T = config_dict['T']
p1 = config_dict['p1']
p2 = config_dict['p2']
t_step = config_dict['t_step']
trn_freq = config_dict['trn_freq']
max_training_epochs = config_dict['max_training_epochs']

wr_freq_range = slice(ppm_to_point_index(p2, T, t_step, trn_freq), ppm_to_point_index(p1, T, t_step, trn_freq))  # water removal frequency range.
plt_freq_range = slice(ppm_to_point_index(7.0, T, t_step, trn_freq), ppm_to_point_index(1.0, T, t_step, trn_freq))  # water removal frequency range.
meta_freq_range = slice(ppm_to_point_index(4.0, T, t_step, trn_freq), ppm_to_point_index(1.0, T, t_step, trn_freq)) 

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")


def cross_subject_train_test(config_dict, test_sub_list=None):
    ## train test split
    if not test_sub_list:
        test_sub_list, train_sub_list = select_random_subjects(num_picks=2, num_subjects=12)
    else:
        train_sub_list = remaining_subjects(test_sub_list)
    print(' Test Subjects:', test_sub_list)
    train_data = load_data(train_sub_list)
    z_train = train_data[:, :T].T
    print('train data size:', z_train.shape)

    config_dict['test_sub_list'] = test_sub_list
    config_dict['train_sub_list'] = train_sub_list

    ## train
    # load and preprocess
    z_train_normalized = normalize(z_train)
    # plot_timefreq(z_normalized.T)
    # z_train_filtered = filter_low_power(z_train_normalized)

    # split data and create dataloaders
    z_train = torch.tensor(z_train_normalized).to(device)
    generator = torch.manual_seed(42)
    train_dataset, val_dataset = create_datasets(z_train, ratio=[0.9, 0.1], generator=generator)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, generator=generator)

    # create the model
    autoencoder = AutoEncoder(config_dict=config_dict, device=device)
    params = sum(p.numel() for p in autoencoder.parameters())
    print(f'number of autoencoder parameters: {params}')

    autoencoder, opt = train(autoencoder, train_dataloader, val_dataloader, config_dict, device=device)
    # save checkpoint
    ex_subs = '_ex' + '-'.join(map(str, test_sub_list)) + f"_head{config_dict['decoder_n_heads']}" # excluded subjects
    save_checkpoint(autoencoder, opt, config_dict, epoch=max_training_epochs, checkpoint_dir='checkpoints/final', ex=ex_subs) # final checkpoint

    ## test
    # load test data
    test_data = load_data(test_sub_list)
    z_test = test_data[:, :T].T
    print('test data size:', z_test.shape)

    # load wr data
    # test_wr_data = load_data(wr_file_names, wr_data_urls)
    # z_wr_test = test_wr_data[:, :T].T

    # normalize
    z_test_normalized, r_base = normalize(z_test, return_base_values=True)
    # z_wr_test_normalized = normalize(z_wr_test, r_base=r_base)

    # to tensor
    z_test = torch.tensor(z_test_normalized, dtype=torch.complex64)
    # z_wr_test = torch.tensor(z_wr_test_normalized, dtype=torch.complex64)

    # dataloader
    test_dataloader = DataLoader(z_test.to(device), batch_size=1024)

    z_test_res_f = test(autoencoder, test_dataloader, device=device)

    # compare results
    wr_dl = z_test_res_f.real.mean(axis=0).cpu()
    plot_freq(wr_dl)

    # z_wr_test_f = torch.fft.fftshift(torch.fft.fft(z_wr_test, dim=-1), dim=-1)
    # wr_svd = z_wr_test_f.real.mean(axis=0)

    # compare_plot(wr_dl, wr_svd, plt_freq_range)


def calculate_metrics_on_final_checkpoints(save_result=True):
    ckpt_dir = Path('checkpoints/final')
    wpsr_list = []
    rr_list = []
    for checkpoint_path in ckpt_dir.iterdir():
        config = torch.load(checkpoint_path, weights_only=False)['config']
        autoencoder = AutoEncoder(config_dict=config, device=device)
        opt = autoencoder.configure_optimizer()
    
        autoencoder, opt, config = load_checkpoint(autoencoder, opt, checkpoint_path, device=device)
        autoencoder.eval()
        test_sub_list = config['test_sub_list']
        T = config['T']
        test_data = load_data(test_sub_list, data_dir='data')
        z_test = test_data[:, :T].T
        print('test sub list:', test_sub_list)

        # load wr data
        # test_wr_data = load_data(wr_file_names, wr_data_urls)
        # z_wr_test = test_wr_data[:, :T].T

        # normalize
        z_test_normalized, r_base = normalize(z_test, return_base_values=True)
        r_base = torch.tensor(r_base)[:, None].to(device)
        # z_wr_test_normalized = normalize(z_wr_test, r_base=r_base)

        # to tensor
        z_test = torch.tensor(z_test_normalized, dtype=torch.complex64).to(device)
        # z_wr_test = torch.tensor(z_wr_test_normalized, dtype=torch.complex64)

        # dataloader
        test_dataloader = DataLoader(z_test, batch_size=1024)

        z_test_res_f = test(autoencoder, test_dataloader, device=device)
        z_test_f = torch.fft.fftshift(torch.fft.fft(z_test, dim=-1), dim=-1)


        wpsr_score = calculate_wpsr(z_test_f, z_test_res_f, wr_freq_range)
        rr_error = calculate_rmse_reconstruct_err(r_base * z_test_f, r_base * z_test_res_f, meta_freq_range)

        wr_dl = z_test_res_f.real.mean(0).cpu()
        plot_freq(wr_dl)
    

        print(f'wpsr score of {checkpoint_path.stem}: {wpsr_score:.2f}')
        print(f'rr error of {checkpoint_path.stem}: {rr_error:.4f}')
        print('-' * 40)

        wpsr_list.append(wpsr_score.cpu().item())
        rr_list.append(rr_error.cpu().item())
        if save_result:
            z_test_res = torch.fft.ifft(torch.fft.ifftshift(r_base * z_test_res_f, dim=-1), dim=-1) # time domain
            file_name = 'TestSubs' + '_'.join(map(str, test_sub_list))
            out_path = Path('result') / file_name
            np.save(out_path, z_test_res.detach().cpu().numpy())

        # compare results
        # wr_dl = z_test_res_f.real.mean(axis=0).cpu()

    wpsr = np.array(wpsr_list)
    rr = np.array(rr_list)
    print(f'WSR mean: {wpsr.mean():.2f} std:{wpsr.std():.2f}')
    print(f'WRE mean: {rr.mean():.2f} std:{rr.std():.2f}')
    

if __name__ == '__main__':
    
    # import time
    # from config import config_dict


    # time_file_path = "result/train_test_processing_time.txt"
    # sub_groups = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    # for group in sub_groups:

    #     t_start = time.time()
    #     cross_subject_train_test(config_dict=config_dict, test_sub_list=group)
    #     t_end = time.time()
        
    #     with open(time_file_path, "a") as time_file:
    #         time_file.write(f"test group {group} processing time: {t_end - t_start:.2f} seconds. \n")

    #     print(f"Processing time saved as: {time_file_path}")

    calculate_metrics_on_final_checkpoints()
