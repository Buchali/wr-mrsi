import torch


def calculate_wpsr(original_water_peak, residual_water_peak, wr_range):
    """
    Calculates the Water Peak Suppression Ratio (WPSR) for water peak removal.
    """
    # Water Peak Suppression Ratio (WPSR)
    power_original = torch.sum(torch.real(original_water_peak[:, wr_range]) ** 2, dim=-1)
    power_residual = torch.sum(torch.real(residual_water_peak[:, wr_range]) ** 2, dim=-1)
    return (10 * torch.log10(power_original / power_residual)).mean()


def calculate_rmse_reconstruct_err(original_data, wr_data, meta_range):
    """
    Calculates the Root Mean Square Error (RMSE) for the reconstruction error.
    """
    # Mask out the water peak region

    # Calculate error outside the water peak region
    error = torch.real(original_data[:, meta_range] - wr_data[:, meta_range])
    mse = torch.sum(error ** 2, dim=-1)  # Compute MSE for each sample (outside water peak region)
    rmse = torch.sqrt(mse).mean()  # Average RMSE across all samples
    return rmse * 1e-6


def r2_score(y_true, y_pred):
    """
    Calculates the R² score for the reconstruction error.
    """
    # Extract the real part of the complex signals
    y_true_real = y_true.real
    y_pred_real = y_pred.real

    # Compute the residual sum of squares (RSS)
    ss_res = torch.sum((y_true_real - y_pred_real) ** 2, dim=1)

    # Compute the total sum of squares (TSS)
    ss_tot = torch.sum((y_true_real - torch.mean(y_true_real, dim=1, keepdim=True)) ** 2, dim=1)

    # Compute R² for each signal and take the mean over the batch
    r2 = 1 - ss_res / ss_tot
    r2_mean = torch.mean(r2)  # Return the mean R² across the batch

    return r2_mean


def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) for the reconstruction error.
    """
    # Extract the real part of the complex signals
    y_true_real = y_true.real
    y_pred_real = y_pred.real

    # Compute the absolute difference
    numerator = torch.abs(y_true_real - y_pred_real)

    # Compute the sum of the absolute values of the true and predicted signals
    denominator = (torch.abs(y_true_real) + torch.abs(y_pred_real)) / 2
    # To avoid division by zero, add a small epsilon to the denominator
    epsilon = 1e-8
    smape = torch.mean(200 * numerator / (denominator + epsilon), dim=1)  # Compute SMAPE for each signal
    smape_mean = torch.mean(smape)  # Return the mean SMAPE across the batch
    
    return smape_mean
