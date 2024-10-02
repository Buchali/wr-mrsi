import os
import time
import numpy as np

import hlsvdpropy as hlsvdpro
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


def watrem( data, dt, n, f):
    npts = len(data)
    dwell = dt
    nsv_sought = n
    result = hlsvdpro.hlsvd(data, nsv_sought, dwell)
    nsv_found, singvals, freq, damp, ampl, phas = result
    # while np.isclose(singvals,0).any():
    #      nsv_sought -= np.isclose(singvals,0).sum()
    #      # if nsv_sought<0:
    #      #     nsv_sought -= 1
    #      result = hlsvdpro.hlsvd(data, nsv_sought, dwell)
    #      nsv_found, singvals, freq, damp, ampl, phas = result
    #     # fid = hlsvdpro.create_hlsvd_fids(result, npts, dwell, sum_results=True, convert=False)
    #     # plt.plot(np.fft.fftshift(np.fft.fft(data))[900:1500])
    #     # plt.plot(np.fft.fftshift(np.fft.fft(fid))[900:1500])
    #     # plt.show()
    # idx = np.where((np.abs(result[2]) < f) & (np.abs(result[3])>0.005))
    min_band = f[0]
    max_band = f[1]
    idx = []
    for min_, max_ in zip(min_band,max_band):
        idx.append(np.where(((result[2]) > min_) & ((result[2]) < max_)))
    idx = np.unique(np.concatenate(idx, 1))
    result_ = (len(idx), result[1][idx], result[2][idx], result[3][idx], result[4][idx], result[5][idx])
    fid = hlsvdpro.create_hlsvd_fids(result_, npts, dwell, sum_results=True, convert=False)
    # plt.plot(np.linspace(-(1/(2*dt)), (1/(2*dt)), len(data)), np.fft.fftshift(np.fft.fft(data)))
    # plt.plot(np.linspace(-(1/(2*dt)), (1/(2*dt)), len(data)),np.fft.fftshift(np.fft.fft(fid)))
    # plt.show()
    return data - fid


def watrem_batch(dataset, dt, n, f):
    dataset_ = np.zeros_like(dataset)
    for idx in tqdm(range(len(dataset[0])), desc="Water Removal Progress"):
        dataset_[:,idx] = watrem(dataset[:,idx], dt, n, f)
    return dataset_


def process_data(data_path):
    start_time = time.time()
    data_path = Path(data_path)

    # Load data and mask
    data = np.load(data_path).T

    # Set parameters
    dt = 0.36e-3
    tsf = 298.08

    # Perform water removal
    dataset_hsvd = watrem_batch(data, dt, 30, ([-tsf * 1.3], [tsf * 0.7]))

    # Save the processed data with "_wr" prefix
    output_path = data_path.parent / (data_path.stem + "_wr.npy")
    np.save(output_path, dataset_hsvd.T)

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Processed data saved as: {output_path}")
    print(f"Processing time: {processing_time:.2f} seconds")

    # Save processing time to a file
    time_file_path = data_path.parent / "all_processing_time.txt"
    with open(time_file_path, "a") as time_file:
        time_file.write(f"{data_path.stem} processing time: {processing_time:.2f} seconds. \n")

    print(f"Processing time saved as: {time_file_path}")


if __name__ == '__main__':
    rng = range(1, 13)
    # Iterate through folders HC01_M01 to HC12_M01
    data_dir = Path('data')
    for sub_num in rng:
        print(f'subject {sub_num} is being processed...')
        sub_file = Path(f"HC{sub_num:02d}_M01.npy")
        sub_path = data_dir / sub_file
        process_data(sub_path)
    