import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ppm_tools import ppm_to_point_index
from utils import load_checkpoint, plot_dual_freq, save_checkpoint


def train(autoencoder, train_dataloader, val_dataloader, config_dict, use_checkpoint=False, device='cpu'):
    """
    Train the autoencoder model.

    Parameters:
    autoencoder (nn.Module): The autoencoder model.
    train_dataloader (torch.utils.data.DataLoader): The dataloader for the training dataset.
    val_dataloader (torch.utils.data.DataLoader): The dataloader for the validation dataset.
    config_dict (dict): The configuration dictionary.
    use_checkpoint (bool): Whether to use a checkpoint for training.
    device (str): The device to use for training.
    """
    # params
    T = config_dict['T']
    p1 = config_dict['p1']
    p2 = config_dict['p2']
    t_step = config_dict['t_step']
    trn_freq = config_dict['trn_freq']
    max_training_epochs = config_dict['max_training_epochs']
    training_report_epoch = config_dict['training_report_epoch']
    start_epoch = 0

    wr_freq_range = slice(ppm_to_point_index(p2, T, t_step, trn_freq), ppm_to_point_index(p1, T, t_step, trn_freq))  # water removal frequency range.
    plt_freq_range = slice(ppm_to_point_index(7, T, t_step, trn_freq), ppm_to_point_index(1, T, t_step, trn_freq))  # water removal frequency range.

    # scheduler
    def get_lr(step, max_lr=1e-2, min_lr=1e-4, warmup_epochs=5, max_epochs=100):
        """
        Cosine annealing learning rate scheduler.
        """
        if step < warmup_epochs:
            return max_lr * (step + 1) / warmup_epochs

        if step > max_epochs:
            return min_lr

        decay_ratio = (step - warmup_epochs) / (max_epochs - warmup_epochs)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    opt = autoencoder.configure_optimizer()
    if use_checkpoint:
        autoencoder, opt, config = load_checkpoint(model=autoencoder, optimizer=opt, checkpoint_path='checkpoints/training', device=device)
        start_epoch = config['cur_epoch']

    def validate():
        """
        Validate the autoencoder model on the validation dataset.
        """
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
    # mse_loss = nn.MSELoss()

    for epoch in tqdm(range(start_epoch, max_training_epochs)):
        t1 = time.time()
        for z in train_dataloader:
            z = z.to(device)
            autoencoder.zero_grad()
            # time domain
            z_rec = autoencoder(z, verbose=False)
            # freq domain
            z_f = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
            z_rec_f = torch.fft.fftshift(torch.fft.fft(z_rec, dim=-1), dim=-1)

            z_f_real = torch.view_as_real(z_f)
            z_rec_f_real = torch.view_as_real(z_rec_f)

            # loss
            loss = F.mse_loss(z_rec_f_real[:, wr_freq_range, :], z_f_real[:, wr_freq_range, :])

            loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0) # stop great gradient shocks of high loss
            lr = get_lr(epoch, max_lr=1e-2, min_lr=3e-5, warmup_epochs=5, max_epochs=max_training_epochs)
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            opt.step()

            accum_loss += loss.detach()
            cur_step += 1
        t2 = time.time()
        # ----- reports ----
        with torch.no_grad():
            if epoch % training_report_epoch == 0:

                print(30 * '-')
                print(f'epoch {epoch:>3} | loss: {accum_loss.item()/cur_step:.4f} | epoch time: {t2-t1:.2f}s | lr: {lr:.3e}')

                accum_loss = 0
                cur_step = 0

                # visualize training
                # plot_dual_freq(z_f, z_rec_f)

                # --- validate ---
                validate()

                # save checkpoint
                # -----------------
                save_checkpoint(autoencoder, opt, config_dict, epoch=epoch, checkpoint_dir='checkpoints/training')
        # break
    return autoencoder, opt
