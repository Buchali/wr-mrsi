import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import config_dict
from ppm_tools import point_to_ppm


def normalize(z,  return_base_values=False, r_base=None):
    # polar representation
    r = np.abs(z)
    ang = np.angle(z)

    if r_base is None:
        r_base = np.amax(r, axis=0)
    r_normalized = r / r_base
    z_normalized = r_normalized * np.exp(1j * ang)
    if return_base_values:
        return z_normalized.T, r_base
    return z_normalized.T

    ## if phase correction is needed:
    # reference_time_point = np.argmax(np.abs(z), axis=0).mean().astype(int) # based on maximum intensity
    # ang_base = np.angle(z_normalized[reference_time_point, :])
    # z_shifted = (z_normalized * np.exp(-1j * ang_base))
    # return z_shifted.T


def filter_low_power(z, order=6):
    powers = np.mean(abs(z), axis=-1)
    threshold = np.median(powers) / order
    true_indexes = np.where(powers >= threshold)[0]
    print(len(true_indexes))
    return z[true_indexes]


def plot_timefreq(z):
    N, T = z.shape
    # plot mean of signals in time and freq domains.
    fig, axes = plt.subplots(1, 2, figsize=(20, 3))
    ppm = point_to_ppm(T)

    # plot in time domain
    axes[0].plot(np.real(z.mean(axis=-1)), linewidth=2)

    # plot in freq domain
    z_mean_f = (np.real(np.fft.fftshift(np.fft.fft(z, axis=0), axes=0))).mean(axis=-1)  # [plt_freq_range]
    axes[1].plot(ppm, z_mean_f, linewidth=2)
    axes[1].invert_xaxis()
    axes[1].grid(True)

    axes[0].set_xlabel('Time Domain')
    axes[1].set_xlabel('Freq Domain')
    axes[0].set_ylabel('Real')

    plt.show()


def plot_dual_freq(z_f, z_rec_f, plt_freq_range):
    z_f = z_f.cpu()
    z_rec_f = z_rec_f.cpu()
    ppm = point_to_ppm(config_dict['T'])
    ppm = ppm[plt_freq_range]
    z_f = z_f[..., plt_freq_range]
    z_rec_f = z_rec_f[..., plt_freq_range]

    N, _ = z_f.shape

    fig, axes = plt.subplots(2, 1, figsize=(7, 5))
    # plot in Freq domain
    axes[0].plot(ppm, z_f.mean(axis=0).real, color='k', linestyle='dashed', linewidth=1.0, label='data')
    axes[0].plot(ppm, z_rec_f.mean(axis=0).real, color='r', label='rec')

    axes[0].invert_xaxis()
    axes[0].set_xlabel('Freq (ppm)')
    axes[0].legend()

    # Residual in Freq domain
    z_res_f = z_f - z_rec_f
    axes[1].plot(ppm, z_res_f.mean(axis=0).real, color='g', label='residual')

    axes[1].invert_xaxis()
    axes[1].set_xlabel('Freq (ppm)')
    axes[1].legend()

    plt.show()

def save_checkpoint(model, optimizer, config, epoch, checkpoint_path=Path("checkpoints")):
    config['cur_epoch'] = epoch
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    today_str = datetime.today().strftime('%Y-%m-%d')
    # Check if the directory exists
    if not checkpoint_path.is_dir():
        # Create the directory
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    filename = Path(checkpoint_path/f"checkpoint_epoch{epoch}_{today_str}.pth")
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, checkpoint_path=Path("checkpoints"), device='cpu'):
    if not checkpoint_path.is_dir() or not any(checkpoint_path.glob("*.pth")):
        return model, optimizer, 0
    # Load checkpoint

    def get_filename_with_highest_epoch(path):
        # List all files that match the pattern
        files = list(path.glob('checkpoint_epoch*_*.pth'))
        # Extract the epoch number and find the file with the highest epoch
        highest_epoch_file = max(
            files,
            key=lambda f: int(re.search(r'checkpoint_epoch(\d+)_', f.name).group(1))
        ) if files else None

        return highest_epoch_file

    print(get_filename_with_highest_epoch(checkpoint_path))
    checkpoint = torch.load(get_filename_with_highest_epoch(checkpoint_path), map_location=torch.device(device))
    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Restore configuration and epoch
    config = checkpoint['config']
    epoch = config.get('cur_epoch', 0)
    return model, optimizer, epoch


def compare_plot(wr_dl, wr_svd, plt_freq_range, org_data=None):
    ppm = point_to_ppm(config_dict['T'])
    fig, ax = plt.subplots(figsize=(10, 5))
    wr_dl = wr_dl.cpu()
    wr_svd = wr_svd.cpu()
    ax.plot(ppm[plt_freq_range], wr_svd[plt_freq_range], label='wr-SVD')
    ax.plot(ppm[plt_freq_range], wr_dl[plt_freq_range], label='wr-DL')
    if org_data is not None:
        ax.plot(ppm[plt_freq_range], org_data[plt_freq_range], label='org-data')

    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel('Freq (ppm)')
    ax.set_ylabel('Real')
    plt.show()
