import argparse
import json
import time

import numpy as np
from tqdm import tqdm

import config
from model.gavae import (
    COVERAGE_LOSS_WEIGHT,
    EDGE_RECON_WEIGHT,
    KL_WEIGHT,
    LOWER_RANGE_LOSS_WEIGHT,
    LOWER_RECON_WEIGHT,
    MOTION_LOSS_WEIGHT,
    RANGE_LOSS_WEIGHT,
)
from util.dataset.routes import get_route
from util.dataset.scaler import TSFeatureWiseScaler
from util.metric.denoise import savgol_denoise
from util.metric.envelope import corridor_envelope_calibration
from util.metric.similarity import trajectory_metrics


def route_name(route_id):
    return {"1": "route1", "2": "route2"}[route_id]


def train_dir(route):
    return config.TRAIN_RUN_ROOT / route.name


def generate_dir(route):
    return config.GENERATE_RUN_ROOT


def prepare_data(route):
    raw = np.load(route.train_path).astype(np.float32)
    scaler = TSFeatureWiseScaler()
    scaled = scaler.fit_transform(raw).astype(np.float32)
    return raw, scaled, scaler


def configure_torch(torch):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def autocast_enabled():
    return config.PRECISION == "bfloat16"


def should_report_epoch(epoch, total_epochs):
    return epoch % 10 == 0 or epoch == total_epochs


def build_model(route, device):
    from model.gavae import build_vae

    return build_vae(route.seq_len, route.feature_dim, route.latent_dim, dropout_rate=config.DROPOUT_RATE, device=device)


def sample_latent(route, model, scaled):
    import torch

    device = next(model.parameters()).device
    data = torch.from_numpy(scaled).to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled()):
        z_mean, z_log_var = model.encode(data)
    indices = torch.randint(0, data.shape[0], (config.SAMPLES,), device=device)
    epsilon = torch.randn((config.SAMPLES, route.latent_dim), device=device)
    return z_mean.index_select(0, indices).float() + torch.exp(0.5 * z_log_var.index_select(0, indices).float()) * epsilon


def decode(model, latent):
    import torch

    device = next(model.parameters()).device
    outputs = []
    with torch.inference_mode():
        for start in range(0, latent.shape[0], 128):
            chunk = latent[start:start + 128].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled()):
                outputs.append(model.decode(chunk).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def generate_arrays(route, model, scaler, scaled):
    latent = sample_latent(route, model, scaled)
    generated = scaler.inverse_transform(decode(model, latent).copy())
    return generated, latent.float().cpu().numpy()


def metric_payload(route, synthetic_path):
    real = np.load(route.train_path).astype(np.float64)
    synthetic = np.load(synthetic_path).astype(np.float64)
    metrics = trajectory_metrics(real, synthetic, include_auxiliary=True)
    metrics["route"] = route.name
    metrics["synthetic"] = str(synthetic_path)
    return metrics


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def format_metric_lines(metrics):
    keys = [
        "summary_distance",
        "mmd",
        "dm",
        "dtw",
        "bbox_coverage",
        "longitude_cross_sample_std_ratio",
        "latitude_cross_sample_std_ratio",
    ]
    lines = []
    for key in keys:
        if key in metrics:
            lines.append(f"{key.replace('_', ' ')}: {metrics[key]:.6f}")
    return lines


def format_scaled_test_metric_lines(metrics):
    scaled = [
        ("MAE", "mae", 1e-3),
        ("MSE", "mse", 1e-6),
        ("SD", "summary_distance", 1e-3),
        ("MMD", "mmd", 1e-5),
        ("DTW", "dtw", 1e-3),
    ]
    lines = ["Scaled values: MAE 10^-3, MSE 10^-6, SD 10^-3, MMD 10^-5, DTW 10^-3"]
    for label, key, scale in scaled:
        if key in metrics:
            lines.append(f"{label}: {metrics[key] / scale:.6f}")
    return lines


def plot_generation(route, real, generated, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(80, real.shape[0], generated.shape[0])
    real_indices = np.linspace(0, real.shape[0] - 1, count, dtype=np.int64)
    generated_indices = np.linspace(0, generated.shape[0] - 1, count, dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), dpi=220, constrained_layout=True)
    panels = [
        (axes[0], real[real_indices], "Real"),
        (axes[1], generated[generated_indices], "Generated"),
    ]
    for axis, data, title in panels:
        for trajectory in data:
            axis.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.7, alpha=0.35)
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.ticklabel_format(useOffset=False)
        axis.grid(True, linewidth=0.3, alpha=0.35)
    fig.suptitle(route.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_train(args):
    import torch
    from model.gavae import vae_losses

    route = get_route(route_name(args.route))
    output_dir = train_dir(route)
    output_dir.mkdir(parents=True, exist_ok=True)
    real, scaled, scaler = prepare_data(route)
    configure_torch(torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(route, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, eps=1e-7)
    data = torch.from_numpy(scaled).to(device)
    order = torch.arange(data.shape[0], device=device)
    history_path = output_dir / "history.jsonl"
    best_loss = float("inf")
    best_epoch = None
    best_row = None
    best_state = None
    start_time = time.time()

    progress = tqdm(range(1, config.EPOCHS + 1), desc=f"train {route.name}", unit="epoch")
    for epoch in progress:
        model.train()
        totals, recons, kls, motions, coverages = [], [], [], [], []
        for start in range(0, data.shape[0], config.BATCH_SIZE):
            x = data.index_select(0, order[start:start + config.BATCH_SIZE])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled()):
                reconstruction, z_mean, z_log_var = model(x)
                total, recon, kl, motion, coverage = vae_losses(
                    x,
                    reconstruction,
                    z_mean,
                    z_log_var,
                    motion_loss_weight=MOTION_LOSS_WEIGHT,
                    kl_weight=KL_WEIGHT,
                    lower_recon_weight=LOWER_RECON_WEIGHT,
                    edge_recon_weight=EDGE_RECON_WEIGHT,
                    coverage_loss_weight=COVERAGE_LOSS_WEIGHT,
                    range_loss_weight=RANGE_LOSS_WEIGHT,
                    lower_range_loss_weight=LOWER_RANGE_LOSS_WEIGHT,
                )
                loss = total.sum()
            loss.backward()
            optimizer.step()
            totals.append(total.detach())
            recons.append(recon.detach())
            kls.append(kl.detach().expand(x.shape[0]))
            motions.append(motion.detach().expand(x.shape[0]))
            coverages.append(coverage.detach().expand(x.shape[0]))
        row = {
            "epoch": epoch,
            "elapsed_sec": time.time() - start_time,
            "total_loss": float(torch.cat(totals).mean().cpu()),
            "reconstruction_loss": float(torch.cat(recons).mean().cpu()),
            "kl_loss": float(torch.cat(kls).mean().cpu()),
            "motion_loss": float(torch.cat(motions).mean().cpu()),
            "coverage_loss": float(torch.cat(coverages).mean().cpu()),
        }
        history_path.open("a").write(json.dumps(row, sort_keys=True) + "\n")
        if should_report_epoch(epoch, config.EPOCHS):
            progress.set_postfix(
                total=f"{row['total_loss']:.4f}",
                recon=f"{row['reconstruction_loss']:.4f}",
                kl=f"{row['kl_loss']:.4f}",
                motion=f"{row['motion_loss']:.4f}",
                coverage=f"{row['coverage_loss']:.4f}",
            )
        if row["total_loss"] < best_loss:
            best_loss = row["total_loss"]
            best_epoch = epoch
            best_row = row
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "route": route.name,
            "dropout_rate": config.DROPOUT_RATE,
            "precision": config.PRECISION,
            "selected_epoch": best_epoch,
            "selection_row": best_row,
        },
        checkpoint_path,
    )
    generated, latent = generate_arrays(route, model, scaler, scaled)
    generated = corridor_envelope_calibration(real, generated, blend=config.CORRIDOR_CALIBRATION_BLEND)
    denoised = savgol_denoise(generated, route.denoise_window, route.denoise_polyorder)
    np.save(output_dir / "generated.npy", generated)
    np.save(output_dir / "denoised.npy", denoised)
    np.save(output_dir / "latent.npy", latent)
    metrics = metric_payload(route, output_dir / "denoised.npy")
    write_json(output_dir / "metrics.json", metrics)
    summary = {
        "route": route.name,
        "checkpoint": str(checkpoint_path),
        "generated": str(output_dir / "generated.npy"),
        "denoised": str(output_dir / "denoised.npy"),
        "metrics": str(output_dir / "metrics.json"),
        "selected_epoch": best_epoch,
    }
    write_json(output_dir / "summary.json", summary)
    tqdm.write(f"{route.name} training complete")
    tqdm.write(f"selected epoch: {best_epoch}")
    for line in format_metric_lines(metrics):
        tqdm.write(line)


def run_test(args):
    route = get_route(route_name(args.route))
    source = train_dir(route) / "denoised.npy"
    output = config.TEST_RUN_ROOT / f"{route.name}_metrics.json"
    with tqdm(total=2, desc=f"test {route.name}", unit="step") as progress:
        progress.set_postfix_str("evaluating")
        payload = metric_payload(route, source)
        progress.update()
        progress.set_postfix_str("saving")
        write_json(output, payload)
        progress.update()
    tqdm.write(f"{route.name} test complete")
    for line in format_scaled_test_metric_lines(payload):
        tqdm.write(line)


def run_generate(args):
    import torch

    route = get_route(route_name(args.route))
    with tqdm(total=5, desc=f"generate {route.name}", unit="step") as progress:
        progress.set_postfix_str("loading data")
        real, scaled, scaler = prepare_data(route)
        progress.update()
        progress.set_postfix_str("loading model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(train_dir(route) / "checkpoint.pt", map_location=device)
        model = build_model(route, device)
        model.load_state_dict(state["state_dict"])
        model.eval()
        progress.update()
        progress.set_postfix_str("generating")
        generated, latent = generate_arrays(route, model, scaler, scaled)
        generated = corridor_envelope_calibration(real, generated, blend=config.CORRIDOR_CALIBRATION_BLEND)
        denoised = savgol_denoise(generated, route.denoise_window, route.denoise_polyorder)
        progress.update()
        progress.set_postfix_str("plotting")
        output_dir = generate_dir(route)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / f"{route.name}_trajectories.png"
        plot_generation(route, real, denoised, figure_path)
        progress.update()
        progress.set_postfix_str("saving")
        np.save(output_dir / f"{route.name}_generated.npy", generated)
        np.save(output_dir / f"{route.name}_denoised.npy", denoised)
        np.save(output_dir / f"{route.name}_latent.npy", latent)
        progress.update()
    tqdm.write(f"{route.name} generation complete")
    tqdm.write(f"generated samples: {generated.shape[0]}")
    tqdm.write(f"figure: {figure_path.name}")


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, func in (("train", run_train), ("test", run_test), ("generate", run_generate)):
        command = subparsers.add_parser(name)
        command.add_argument("--route", choices=["1", "2"], required=True)
        command.set_defaults(func=func)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
