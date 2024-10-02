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
    print("num filtered samples:", len(z) - len(true_indexes))
    return z[true_indexes]


def plot_freq(z_f):
    T = len(z_f)
    # plot mean of signals in time and freq domains.
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ppm = point_to_ppm(T)

    # plot in freq domain
    ax.plot(ppm, z_f, linewidth=1.5)
    ax.invert_xaxis()

    ax.set_xlabel('Freq (ppm)')
    ax.set_ylabel('Intencity (real)')

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

def save_checkpoint(model, optimizer, config, epoch, checkpoint_dir='checkpoints/training', ex=''):
    # Check if the directory exists
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        # Create the directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config['cur_epoch'] = epoch
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    today_str = datetime.today().strftime('%Y%m%d')
    
    checkpoint_path = checkpoint_dir / f"ckpt_ep{epoch}_{today_str}{ex}.pth"
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path='checkpoints/training', device='cpu'):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_dir = Path(checkpoint_path)
        if not any(checkpoint_dir.glob("*.pth")):
            raise ValueError(f'No checkpoints available in {checkpoint_dir} path!')
        # Load checkpoint

        def get_filename_with_highest_epoch(path):
            # List all files that match the pattern
            files = list(path.glob('ckpt_ep*_*.pth'))
            # Extract the epoch number and find the file with the highest epoch
            highest_epoch_file = max(
                files,
                key=lambda f: int(re.search(r'ckpt_ep(\d+)_', f.name).group(1))
            ) if files else None

            return highest_epoch_file
        checkpoint_path = get_filename_with_highest_epoch(checkpoint_dir)

    elif not checkpoint_path.is_file() or checkpoint_path.suffix != '.pth':
        raise ValueError(f'Could Not Find {checkpoint_path}!')
        
    print(f'loading checkpoin: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Restore configuration and epoch
    config = checkpoint['config']
    return model, optimizer, config


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
    ax.set_ylabel('Intensity (real)')
    plt.show()
