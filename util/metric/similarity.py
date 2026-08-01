import itertools
import numpy as np


AUXILIARY_SAMPLE_SIZE = 128
DTW_BLOCK_SIZE = 32
PAIRWISE_BLOCK_SIZE = 32


def axis_max_s(ts, axis):
    return np.array([np.max(ts)]) if axis is None else np.max(np.max(ts, axis=axis), axis=0).flatten()


def axis_min_s(ts, axis):
    return np.array([np.min(ts)]) if axis is None else np.min(np.min(ts, axis=axis), axis=0).flatten()


def summary_stats(data):
    stats = [
        lambda x: axis_max_s(x, axis=None),
        lambda x: axis_min_s(x, axis=None),
        lambda x: axis_max_s(x, axis=1),
        lambda x: axis_min_s(x, axis=1),
    ]
    return np.array(list(itertools.chain.from_iterable(fn(data) for fn in stats)))


def summary_distance(real, synthetic):
    return float(np.linalg.norm(summary_stats(real) - summary_stats(synthetic)))


def _rbf_kernel_mean(x, y, block_size=1024):
    x_flat = x.reshape(x.shape[0], -1).astype(np.float64)
    y_flat = y.reshape(y.shape[0], -1).astype(np.float64)
    y_norm = np.sum(y_flat * y_flat, axis=1)[None, :]
    total = 0.0
    count = 0
    for start in range(0, x_flat.shape[0], block_size):
        batch = x_flat[start:start + block_size]
        dist2 = np.sum(batch * batch, axis=1)[:, None] + y_norm - 2.0 * batch @ y_flat.T
        np.maximum(dist2, 0.0, out=dist2)
        total += np.exp(-0.5 * dist2).sum()
        count += dist2.size
    return total / count


def mmd(real, synthetic):
    return float(_rbf_kernel_mean(real, real) + _rbf_kernel_mean(synthetic, synthetic) - 2.0 * _rbf_kernel_mean(real, synthetic))


def pairwise_errors(real, synthetic, block_size=PAIRWISE_BLOCK_SIZE):
    import torch

    device = _device()
    with torch.inference_mode():
        real_t = _to_tensor(real, device)
        synthetic_t = _to_tensor(synthetic, device)
        mae_sum = torch.zeros((), device=device)
        mse_sum = torch.zeros((), device=device)
        count = 0
        for start in range(0, synthetic_t.shape[0], block_size):
            batch = synthetic_t[start:start + block_size]
            diff = batch[:, None] - real_t[None]
            mae = diff.abs().mean(dim=(2, 3))
            mse = diff.square().mean(dim=(2, 3))
            mae_sum = mae_sum + mae.sum()
            mse_sum = mse_sum + mse.sum()
            count += mae.numel()
        return {
            "mae": float((mae_sum / count).cpu()),
            "mse": float((mse_sum / count).cpu()),
        }


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(data, device):
    import torch

    return torch.as_tensor(data, dtype=torch.float32, device=device)


def _sample(data, sample_size):
    if data.shape[0] <= sample_size:
        return data
    indices = np.linspace(0, data.shape[0] - 1, sample_size, dtype=np.int64)
    return data[indices]


def _sample_pair(real, synthetic, sample_size):
    real_sample = _sample(real, sample_size)
    synthetic_sample = _sample(synthetic, sample_size)
    origin = real_sample.reshape(-1, real_sample.shape[-1]).mean(axis=0)
    return real_sample - origin, synthetic_sample - origin


def _trajectory_descriptors(data):
    import torch

    delta = data[:, 1:] - data[:, :-1]
    step = torch.linalg.vector_norm(delta, dim=2)
    return torch.cat(
        [
            data[:, 0],
            data[:, -1],
            data.mean(dim=1),
            data.std(dim=1, unbiased=False),
            data.amin(dim=1),
            data.amax(dim=1),
            delta.mean(dim=1),
            delta.std(dim=1, unbiased=False),
            step.mean(dim=1, keepdim=True),
            step.std(dim=1, unbiased=False, keepdim=True),
        ],
        dim=1,
    )


def _covariance(data):
    centered = data - data.mean(dim=0, keepdim=True)
    return centered.T @ centered / (data.shape[0] - 1)


def _matrix_sqrt_psd(matrix):
    import torch

    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = torch.linalg.eigh(symmetric)
    values = values.clamp_min(0.0).sqrt()
    return (vectors * values.unsqueeze(0)) @ vectors.T


def distance_metric(real, synthetic, sample_size=AUXILIARY_SAMPLE_SIZE):
    import torch

    device = _device()
    with torch.inference_mode():
        real_sample, synthetic_sample = _sample_pair(real, synthetic, sample_size)
        real_t = _to_tensor(real_sample, device)
        synthetic_t = _to_tensor(synthetic_sample, device)
        real_desc = _trajectory_descriptors(real_t)
        synthetic_desc = _trajectory_descriptors(synthetic_t)
        pooled = torch.cat([real_desc, synthetic_desc], dim=0)
        center = real_desc.mean(dim=0, keepdim=True)
        scale = pooled.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        real_desc = (real_desc - center) / scale
        synthetic_desc = (synthetic_desc - center) / scale
        real_mean = real_desc.mean(dim=0)
        synthetic_mean = synthetic_desc.mean(dim=0)
        real_cov = _covariance(real_desc)
        synthetic_cov = _covariance(synthetic_desc)
        real_cov_sqrt = _matrix_sqrt_psd(real_cov)
        covmean = _matrix_sqrt_psd(real_cov_sqrt @ synthetic_cov @ real_cov_sqrt)
        diff = real_mean - synthetic_mean
        value = diff.dot(diff) + torch.trace(real_cov + synthetic_cov - 2.0 * covmean)
        return float(value.clamp_min(0.0).sqrt().cpu())


def _dtw_block(x, y):
    import torch

    b, n, _ = x.shape
    r, m, _ = y.shape
    costs = torch.cdist(x[:, None], y[None]).square().reshape(b * r, n, m)
    scores = torch.full((b * r, n + 1, m + 1), float("inf"), device=x.device, dtype=x.dtype)
    scores[:, 0, 0] = 0.0
    for diagonal in range(2, n + m + 1):
        i_start = max(1, diagonal - m)
        i_end = min(n, diagonal - 1)
        i = torch.arange(i_start, i_end + 1, device=x.device)
        j = diagonal - i
        previous = torch.minimum(
            torch.minimum(scores[:, i - 1, j], scores[:, i, j - 1]),
            scores[:, i - 1, j - 1],
        )
        scores[:, i, j] = costs[:, i - 1, j - 1] + previous
    return scores[:, n, m].clamp_min(0.0).sqrt().reshape(b, r)


def _dtw_distance_matrix(x, y, block_size=DTW_BLOCK_SIZE):
    import torch

    rows = []
    for start in range(0, x.shape[0], block_size):
        rows.append(_dtw_block(x[start:start + block_size], y))
    return torch.cat(rows, dim=0)


def dynamic_time_warping(real, synthetic, sample_size=AUXILIARY_SAMPLE_SIZE):
    import torch

    device = _device()
    with torch.inference_mode():
        real_sample, synthetic_sample = _sample_pair(real, synthetic, sample_size)
        real_t = _to_tensor(real_sample, device)
        synthetic_t = _to_tensor(synthetic_sample, device)
        distances = _dtw_distance_matrix(real_t, synthetic_t)
        real_to_synthetic = distances.min(dim=1).values.mean()
        synthetic_to_real = distances.min(dim=0).values.mean()
        value = 0.5 * (real_to_synthetic + synthetic_to_real)
        return {
            "dtw": float(value.cpu()),
            "dtw_real_to_synthetic": float(real_to_synthetic.cpu()),
            "dtw_synthetic_to_real": float(synthetic_to_real.cpu()),
            "dtw_sample_size": int(min(real.shape[0], synthetic.shape[0], sample_size)),
            "dtw_device": device.type,
        }


def auxiliary_metrics(real, synthetic, sample_size=AUXILIARY_SAMPLE_SIZE):
    metrics = {
        "dm": distance_metric(real, synthetic, sample_size=sample_size),
    }
    metrics.update(dynamic_time_warping(real, synthetic, sample_size=sample_size))
    return metrics


def trajectory_metrics(real, synthetic, include_auxiliary=False, auxiliary_sample_size=AUXILIARY_SAMPLE_SIZE):
    real_lon = real[:, :, 0]
    real_lat = real[:, :, 1]
    synthetic_lon = synthetic[:, :, 0]
    synthetic_lat = synthetic[:, :, 1]
    longitude_range_ratio = float((synthetic_lon.max() - synthetic_lon.min()) / (real_lon.max() - real_lon.min()))
    latitude_range_ratio = float((synthetic_lat.max() - synthetic_lat.min()) / (real_lat.max() - real_lat.min()))
    metrics = {
        **pairwise_errors(real, synthetic),
        "summary_distance": summary_distance(real, synthetic),
        "mmd": mmd(real, synthetic),
        "real_shape": list(real.shape),
        "synthetic_shape": list(synthetic.shape),
        "real_mean": float(np.mean(real)),
        "synthetic_mean": float(np.mean(synthetic)),
        "real_std": float(np.std(real)),
        "synthetic_std": float(np.std(synthetic)),
        "longitude_cross_sample_std_ratio": float(synthetic_lon.std(axis=0).mean() / real_lon.std(axis=0).mean()),
        "latitude_cross_sample_std_ratio": float(synthetic_lat.std(axis=0).mean() / real_lat.std(axis=0).mean()),
        "longitude_range_ratio": longitude_range_ratio,
        "latitude_range_ratio": latitude_range_ratio,
        "bbox_coverage": longitude_range_ratio * latitude_range_ratio,
    }
    if include_auxiliary:
        metrics.update(auxiliary_metrics(real, synthetic, sample_size=auxiliary_sample_size))
    return metrics
