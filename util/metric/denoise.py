import numpy as np
from scipy.signal import savgol_filter


def savgol_denoise(data, window_length, polyorder):
    smoothed = np.empty_like(data)
    for sample_idx in range(data.shape[0]):
        for feature_idx in range(data.shape[2]):
            smoothed[sample_idx, :, feature_idx] = savgol_filter(
                data[sample_idx, :, feature_idx],
                window_length=window_length,
                polyorder=polyorder,
            )
    return smoothed
