from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class RouteConfig:
    name: str
    seq_len: int
    feature_dim: int
    latent_dim: int
    train_path: Path
    denoise_window: int
    denoise_polyorder: int


ROUTES = {
    "route1": RouteConfig(
        name="route1",
        seq_len=91,
        feature_dim=2,
        latent_dim=100,
        train_path=config.DATASET_ROOT / "route1/train.npy",
        denoise_window=21,
        denoise_polyorder=3,
    ),
    "route2": RouteConfig(
        name="route2",
        seq_len=61,
        feature_dim=2,
        latent_dim=100,
        train_path=config.DATASET_ROOT / "route2/train.npy",
        denoise_window=11,
        denoise_polyorder=3,
    ),
}


def get_route(name: str) -> RouteConfig:
    return ROUTES[name]
