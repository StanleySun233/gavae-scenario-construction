import numpy as np


def corridor_envelope_calibration(real, synthetic, blend=0.8):
    real = np.asarray(real, dtype=np.float64)
    synthetic = np.asarray(synthetic, dtype=np.float64)
    real_time_mean = real.mean(axis=0, keepdims=True)
    synthetic_time_mean = synthetic.mean(axis=0, keepdims=True)
    real_global_std = real.reshape(-1, real.shape[-1]).std(axis=0).reshape(1, 1, -1)
    synthetic_global_std = synthetic.reshape(-1, synthetic.shape[-1]).std(axis=0).reshape(1, 1, -1)
    calibrated = (synthetic - synthetic_time_mean) / (synthetic_global_std + 1e-12) * real_global_std + real_time_mean
    return ((1.0 - blend) * synthetic + blend * calibrated).astype(np.float32)
