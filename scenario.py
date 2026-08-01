import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, LinearLocator

import config


DT_SECONDS = 10.0
EARTH_NM_PER_DEG = 60.0
SCENARIOS = ("crossing", "head-on", "overtaking")
MIN_DISTANCE_NM = 0.05
PROXIMITY_NM = 0.50
TCPA_WINDOW_SECONDS = 600.0
DCPA_LIMIT_NM = 0.50
EARLY_STEPS = int(round(config.SCENARIO_EARLY_SECONDS / DT_SECONDS))
AFTER_STEPS = int(round(config.SCENARIO_AFTER_SECONDS / DT_SECONDS))
CASES_PER_SCENARIO = 50
SCENARIO_IMAGE_ROOT = Path("scenarios")
SCENARIO_DIRECTORIES = {
    "crossing": "crossing",
    "head-on": "headon",
    "overtaking": "overtaking",
}
SCENARIO_TITLES = {
    "crossing": "Crossing",
    "head-on": "Head-on",
    "overtaking": "Overtaking",
}


def default_generated_paths():
    return {
        "route1": config.GENERATE_RUN_ROOT / "route1_denoised.npy",
        "route2": config.GENERATE_RUN_ROOT / "route2_denoised.npy",
    }


def to_nm(own, target):
    both = np.concatenate([own.reshape(-1, 2), target.reshape(-1, 2)], axis=0)
    origin = both.mean(axis=0)
    lat0 = math.radians(float(origin[1]))

    def project(data):
        out = np.empty_like(data, dtype=np.float64)
        out[..., 0] = (data[..., 0] - origin[0]) * EARTH_NM_PER_DEG * math.cos(lat0)
        out[..., 1] = (data[..., 1] - origin[1]) * EARTH_NM_PER_DEG
        return out

    return project(own), project(target)


def heading(velocity):
    angle = np.degrees(np.arctan2(velocity[..., 0], velocity[..., 1]))
    return (angle + 360.0) % 360.0


def angle_diff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def classify(angle):
    out = np.full(angle.shape, "Crossing", dtype=object)
    out[angle >= 150.0] = "Head-on"
    out[angle <= 30.0] = "Overtaking"
    return out


def triangular_left(angle, full_until, zero_at):
    return np.clip((zero_at - angle) / (zero_at - full_until), 0.0, 1.0)


def triangular_right(angle, zero_at, full_from):
    return np.clip((angle - zero_at) / (full_from - zero_at), 0.0, 1.0)


def trapezoid_middle(angle, left_zero, left_full, right_full, right_zero):
    left = np.clip((angle - left_zero) / (left_full - left_zero), 0.0, 1.0)
    right = np.clip((right_zero - angle) / (right_zero - right_full), 0.0, 1.0)
    return np.minimum(left, right)


def fuzzy_membership(angle):
    overtaking = triangular_left(angle, 22.5, 67.5)
    crossing = trapezoid_middle(angle, 22.5, 67.5, 112.5, 157.5)
    head_on = triangular_right(angle, 112.5, 157.5)
    raw = np.stack([crossing, head_on, overtaking], axis=-1)
    total = np.maximum(raw.sum(axis=-1, keepdims=True), 1e-12)
    return raw / total


def build_scenario_inputs(route1, route2, scenario):
    if scenario == "crossing":
        return route1, route2, np.arange(-300.0, 301.0, 30.0)
    if scenario == "head-on":
        return route1, route2, np.arange(-300.0, 301.0, 30.0)
    if scenario == "overtaking":
        return route1, route1, np.arange(-300.0, 301.0, 30.0)
    raise ValueError(f"Unsupported scenario: {scenario}")


def roi_mask(points, lower, upper):
    return ((points[..., 0] >= lower[0]) & (points[..., 0] <= upper[0]) & (points[..., 1] >= lower[1]) & (points[..., 1] <= upper[1]))


def scenario_angle_mask(scenario, angle):
    if scenario == "crossing":
        return (angle >= 60.0) & (angle <= 120.0)
    if scenario == "head-on":
        return angle >= 150.0
    if scenario == "overtaking":
        return angle <= 30.0
    raise ValueError(f"Unsupported scenario: {scenario}")


def scenario_rows(own_data, target_data, samples, offsets, batch_size, scenario_name):
    own_data = own_data[:samples]
    target_data = target_data[:samples]
    own_nm, target_nm = to_nm(own_data, target_data)
    own_v = np.gradient(own_nm, DT_SECONDS, axis=1)
    target_v_all = np.gradient(target_nm, DT_SECONDS, axis=1)
    rows = []
    n1 = own_nm.shape[0]
    n2 = target_nm.shape[0]
    t1 = own_nm.shape[1]
    t2 = target_nm.shape[1]
    lower = np.maximum(own_nm.reshape(-1, 2).min(axis=0), target_nm.reshape(-1, 2).min(axis=0))
    upper = np.minimum(own_nm.reshape(-1, 2).max(axis=0), target_nm.reshape(-1, 2).max(axis=0))
    has_roi = np.all(lower < upper)

    for offset in offsets:
        shift = int(round(offset / DT_SECONDS))
        start1 = max(0, shift)
        start2 = max(0, -shift)
        overlap = min(t1 - start1, t2 - start2)
        if overlap <= 1:
            continue
        own = own_nm[:, start1:start1 + overlap]
        target_all = target_nm[:, start2:start2 + overlap]
        own_v_window = own_v[:, start1:start1 + overlap]
        target_v_window_all = target_v_all[:, start2:start2 + overlap]
        times = (np.arange(start1, start1 + overlap) * DT_SECONDS).astype(np.float64)

        for j0 in range(0, n2, batch_size):
            j1 = min(j0 + batch_size, n2)
            target = target_all[j0:j1]
            target_v = target_v_window_all[j0:j1]
            relative_position = target[None, :, :, :] - own[:, None, :, :]
            relative_velocity = target_v[None, :, :, :] - own_v_window[:, None, :, :]
            dist = np.linalg.norm(relative_position, axis=3)
            if scenario_name == "overtaking":
                for i in range(n1):
                    target_index = i - j0
                    if 0 <= target_index < j1 - j0:
                        dist[i, target_index, :] = np.inf
            speed_sq = np.sum(relative_velocity * relative_velocity, axis=3)
            closing = -np.sum(relative_position * relative_velocity, axis=3)
            tcpa_all = np.full_like(dist, np.inf, dtype=np.float64)
            valid_speed = speed_sq > 1e-12
            tcpa_all[valid_speed] = closing[valid_speed] / speed_sq[valid_speed]
            dcpa_all = np.full_like(dist, np.inf, dtype=np.float64)
            dcpa_all[valid_speed] = np.linalg.norm(relative_position[valid_speed] + relative_velocity[valid_speed] * tcpa_all[valid_speed][:, None], axis=1)
            if has_roi:
                in_roi = roi_mask(own, lower, upper)[:, None, :] & roi_mask(target, lower, upper)[None, :, :]
            else:
                in_roi = np.ones(dist.shape, dtype=bool)
            motion_ok = (
                in_roi
                & (dist <= PROXIMITY_NM)
                & (tcpa_all > 0.0)
                & (tcpa_all <= TCPA_WINDOW_SECONDS)
                & (dcpa_all <= DCPA_LIMIT_NM)
            )
            k = dist.argmin(axis=2)
            best_k = np.where(motion_ok, dcpa_all, np.inf).argmin(axis=2)
            i_idx = np.arange(n1)[:, None]
            j_idx = np.arange(j1 - j0)[None, :]
            min_distance = dist[i_idx, j_idx, k]
            dcpa = dcpa_all[i_idx, j_idx, best_k]
            tcpa = tcpa_all[i_idx, j_idx, best_k]
            own_vec = own_v_window[i_idx, k]
            target_vec = target_v[j_idx, k]
            own_head = heading(own_vec)
            target_head = heading(target_vec)
            encounter_angle = angle_diff(own_head, target_head)
            encounter_type = classify(encounter_angle)
            fuzzy = fuzzy_membership(encounter_angle)
            encounter_time = times[k]
            own_speed = np.linalg.norm(own_vec, axis=2)
            target_speed = np.linalg.norm(target_vec, axis=2)
            valid_window = (k >= EARLY_STEPS) & (k + AFTER_STEPS < overlap)
            pair_ok = (min_distance <= MIN_DISTANCE_NM) & motion_ok.any(axis=2) & scenario_angle_mask(scenario_name, encounter_angle) & valid_window
            for i in range(n1):
                for local_j in range(j1 - j0):
                    if not pair_ok[i, local_j]:
                        continue
                    cpa_index = int(k[i, local_j])
                    own_cpa_index = int(start1 + cpa_index)
                    target_cpa_index = int(start2 + cpa_index)
                    scenario_start_index = cpa_index - EARLY_STEPS
                    scenario_end_index = cpa_index + AFTER_STEPS
                    rows.append(
                        {
                            "scenario": scenario_name,
                            "own_id": int(i),
                            "target_id": int(j0 + local_j),
                            "offset_s": float(offset),
                            "t_star_s": float(encounter_time[i, local_j]),
                            "t_early_s": float(config.SCENARIO_EARLY_SECONDS),
                            "t_after_s": float(config.SCENARIO_AFTER_SECONDS),
                            "cpa_index": cpa_index,
                            "own_cpa_index": own_cpa_index,
                            "target_cpa_index": target_cpa_index,
                            "scenario_start_index": scenario_start_index,
                            "scenario_end_index": scenario_end_index,
                            "sigma_i": int(i),
                            "sigma_j": int(j0 + local_j),
                            "min_distance_nm": float(min_distance[i, local_j]),
                            "dcpa_nm": float(dcpa[i, local_j]),
                            "tcpa_s": float(tcpa[i, local_j]),
                            "encounter_time_s": float(encounter_time[i, local_j]),
                            "encounter_angle_deg": float(encounter_angle[i, local_j]),
                            "encounter_type": str(encounter_type[i, local_j]),
                            "crossing_mu": float(fuzzy[i, local_j, 0]),
                            "head_on_mu": float(fuzzy[i, local_j, 1]),
                            "overtaking_mu": float(fuzzy[i, local_j, 2]),
                            "own_speed_nm_s": float(own_speed[i, local_j]),
                            "target_speed_nm_s": float(target_speed[i, local_j]),
                        }
                    )
    return rows


def summarize(rows):
    return {"retained_scenarios": len(rows)}


def select_cases(rows, own_data, target_data):
    scenario_name = rows[0]["scenario"] if rows else "scenario"
    angle_targets = {
        "crossing": 90.0,
        "head-on": 170.0,
        "overtaking": 15.0,
    }
    scored = []
    for row in rows:
        shift = int(round(float(row["offset_s"]) / DT_SECONDS))
        start1 = max(0, shift)
        start2 = max(0, -shift)
        overlap = min(own_data.shape[1] - start1, target_data.shape[1] - start2)
        cpa_index = int(row["cpa_index"])
        pre_steps = cpa_index - EARLY_STEPS
        post_steps = overlap - (cpa_index + AFTER_STEPS + 1)
        if pre_steps < EARLY_STEPS or post_steps < AFTER_STEPS:
            continue
        cpa_ratio = cpa_index / max(overlap - 1, 1)
        target_angle = angle_targets[row["scenario"]]
        score = (
            abs(float(row["encounter_angle_deg"]) - target_angle),
            abs(float(row["min_distance_nm"]) - 0.025),
            abs(cpa_ratio - 0.5),
            abs(float(row["dcpa_nm"]) - 0.04),
        )
        scored.append((*score, row))

    selected = []
    seen_pairs = set()
    for *_, row in sorted(scored, key=lambda item: item[:-1]):
        pair = (int(row["own_id"]), int(row["target_id"]))
        if pair in seen_pairs:
            continue
        selected.append(row)
        seen_pairs.add(pair)
        if len(selected) == CASES_PER_SCENARIO:
            break
    if len(selected) != CASES_PER_SCENARIO:
        raise RuntimeError(f"{scenario_name} produced {len(selected)} of {CASES_PER_SCENARIO} required scenarios")
    return selected


def source_data(route1, route2, scenario):
    if scenario == "crossing":
        return route1, route2
    if scenario == "head-on":
        return route1, route2
    if scenario == "overtaking":
        return route1, route1
    raise ValueError(f"Unsupported scenario: {scenario}")


def aligned_trajectories(route1, route2, row):
    own_all, target_all = source_data(route1, route2, row["scenario"])
    shift = int(round(float(row["offset_s"]) / DT_SECONDS))
    start1 = max(0, shift)
    start2 = max(0, -shift)
    overlap = min(own_all.shape[1] - start1, target_all.shape[1] - start2)
    own_id = int(row["own_id"])
    target_id = int(row["target_id"])
    own = own_all[own_id, start1:start1 + overlap]
    target = target_all[target_id, start2:start2 + overlap]
    return own, target, int(row["cpa_index"])


def scenario_clip(route1, route2, row):
    own, target, cpa_index = aligned_trajectories(route1, route2, row)
    pre_end = cpa_index - EARLY_STEPS
    post_start = cpa_index + AFTER_STEPS + 1
    return {
        "own_pre": own[:pre_end],
        "own_encounter": own[pre_end:post_start],
        "own_post": own[post_start:],
        "target_pre": target[:pre_end],
        "target_encounter": target[pre_end:post_start],
        "target_post": target[post_start:],
        "sigma": np.array([row["own_id"], row["target_id"], float(row["t_star_s"]), config.SCENARIO_EARLY_SECONDS, config.SCENARIO_AFTER_SECONDS], dtype=np.float64),
    }


def write_scenario_clips(root, route1, route2, cases):
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for path in clips_dir.glob("*.npz"):
        path.unlink()
    directories = {}
    for scenario, name in SCENARIO_DIRECTORIES.items():
        directory = clips_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.npz"):
            path.unlink()
        directories[scenario] = directory

    counters = {scenario: 0 for scenario in SCENARIOS}
    enriched = []
    for row in cases:
        scenario = row["scenario"]
        index = counters[scenario]
        clip_path = directories[scenario] / f"{index:03d}.npz"
        clip = scenario_clip(route1, route2, row)
        np.savez_compressed(clip_path, **clip)
        item = dict(row)
        item["clip_path"] = str(clip_path)
        item["sigma"] = {
            "i": int(row["sigma_i"]),
            "j": int(row["sigma_j"]),
            "t_star_s": float(row["t_star_s"]),
            "t_early_s": float(row["t_early_s"]),
            "t_after_s": float(row["t_after_s"]),
        }
        enriched.append(item)
        counters[scenario] += 1
    return enriched


def plot_scenario_image(path, own, target, cpa_index, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    first = cpa_index - EARLY_STEPS
    second = cpa_index + AFTER_STEPS
    for trajectory, color in ((own, "#1565c0"), (target, "#c43c39")):
        segments = (
            (0, first + 1, "-", 2.0),
            (first, second + 1, (0, (0.45, 1.55)), 1.7),
            (second, len(trajectory), (0, (5, 2)), 2.2),
        )
        for start, end, linestyle, linewidth in segments:
            ax.plot(
                trajectory[start:end, 0],
                trajectory[start:end, 1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                dash_capstyle="round",
            )
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color=color, s=28, marker="o", zorder=4)
        ax.scatter(trajectory[cpa_index, 0], trajectory[cpa_index, 1], color=color, s=42, marker="x", linewidths=2.0, zorder=5)

    ax.plot(
        [own[cpa_index, 0], target[cpa_index, 0]],
        [own[cpa_index, 1], target[cpa_index, 1]],
        color="#374151",
        linewidth=0.8,
        alpha=0.8,
    )
    points = np.concatenate([own, target], axis=0)
    lon_pad = max(float(np.ptp(points[:, 0])) * 0.08, 0.0005)
    lat_pad = max(float(np.ptp(points[:, 1])) * 0.08, 0.0005)
    ax.set_xlim(float(points[:, 0].min()) - lon_pad, float(points[:, 0].max()) + lon_pad)
    ax.set_ylim(float(points[:, 1].min()) - lat_pad, float(points[:, 1].max()) + lat_pad)
    ax.set_aspect(1.0 / max(math.cos(math.radians(float(points[:, 1].mean()))), 0.1))
    ax.set_facecolor("#e8f3f6")
    ax.grid(True, color="#aab7c4", linewidth=0.6, alpha=0.7, linestyle="--")
    ax.xaxis.set_major_locator(LinearLocator(4))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    handles = [
        Line2D([0], [0], color="#4b5563", linewidth=2.0, linestyle="-", label="Pre-encounter"),
        Line2D([0], [0], color="#4b5563", linewidth=1.7, linestyle=(0, (0.45, 1.55)), label="Encounter segment"),
        Line2D([0], [0], color="#4b5563", linewidth=2.2, linestyle=(0, (5, 2)), label="Post-encounter"),
        Line2D([0], [0], color="#374151", marker="x", markersize=7, linestyle="none", label="CPA"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=True, fancybox=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_scenario_images(root, route1, route2, cases):
    root.mkdir(parents=True, exist_ok=True)
    for path in root.glob("*.png"):
        path.unlink()
    directories = {}
    for scenario, name in SCENARIO_DIRECTORIES.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.png"):
            path.unlink()
        directories[scenario] = directory

    counters = {scenario: 0 for scenario in SCENARIOS}
    enriched = []
    for row in cases:
        scenario = row["scenario"]
        index = counters[scenario]
        image_path = directories[scenario] / f"{index:03d}.png"
        own, target, cpa_index = aligned_trajectories(route1, route2, row)
        plot_scenario_image(
            image_path,
            own,
            target,
            cpa_index,
            f"{SCENARIO_TITLES[scenario]} scenario {index:03d}",
        )
        item = dict(row)
        item["image_path"] = str(image_path)
        enriched.append(item)
        counters[scenario] += 1
    return enriched


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario", "own_id", "target_id", "offset_s", "t_star_s", "t_early_s", "t_after_s", "cpa_index", "own_cpa_index", "target_cpa_index", "scenario_start_index", "scenario_end_index", "sigma_i", "sigma_j", "min_distance_nm", "dcpa_nm", "tcpa_s", "encounter_time_s", "encounter_angle_deg", "encounter_type", "crossing_mu", "head_on_mu", "overtaking_mu", "own_speed_nm_s", "target_speed_nm_s", "clip_path", "image_path"]
    frame = pd.DataFrame(rows)
    for name in fieldnames:
        if name not in frame:
            frame[name] = pd.Series(dtype="object")
    frame.to_csv(path, index=False, columns=fieldnames)


def print_summary(summaries, cases):
    print("scenario construction complete")
    for name, summary in summaries.items():
        print(f"{name}: {summary['retained_scenarios']} retained scenarios")
    print(f"selected scenarios: {len(cases)}")


def construct_scenarios(route1, route2):
    summaries = {}
    cases = []
    for name in SCENARIOS:
        own, target, offsets = build_scenario_inputs(route1, route2, name)
        rows = scenario_rows(own, target, config.SAMPLES, offsets, config.SCENARIO_BATCH_SIZE, name)
        summaries[name] = summarize(rows)
        cases.extend(select_cases(rows, own, target))
    return summaries, sorted(cases, key=lambda row: (row["scenario"], row["dcpa_nm"]))


def build_parser():
    return argparse.ArgumentParser()


def main():
    build_parser().parse_args()
    paths = default_generated_paths()
    route1 = np.load(paths["route1"]).astype(np.float64)
    route2 = np.load(paths["route2"]).astype(np.float64)
    summaries, cases = construct_scenarios(route1, route2)
    config.SCENARIO_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    cases = write_scenario_clips(config.SCENARIO_RUN_ROOT, route1, route2, cases)
    cases = write_scenario_images(SCENARIO_IMAGE_ROOT, route1, route2, cases)
    result = {
        "route1_path": str(paths["route1"]),
        "route2_path": str(paths["route2"]),
        "samples_per_scenario": config.SAMPLES,
        "time_step_seconds": DT_SECONDS,
        "scenario_representation": "sigma_ij = (i, j, t_star, t_early, t_after), with own/target pre-encounter, encounter, and post-encounter clips stored by encounter type under data/runs/scenario/clips.",
        "matching_conditions": {
            "minimum_observed_distance_nm": MIN_DISTANCE_NM,
            "instantaneous_distance_nm": PROXIMITY_NM,
            "tcpa_window_seconds": TCPA_WINDOW_SECONDS,
            "dcpa_nm": DCPA_LIMIT_NM,
            "complete_window_seconds": {
                "t_early": config.SCENARIO_EARLY_SECONDS,
                "t_after": config.SCENARIO_AFTER_SECONDS,
            },
            "angle_rules": {
                "crossing": "60 <= encounter angle <= 120 degrees",
                "head-on": "encounter angle >= 150 degrees",
                "overtaking": "encounter angle <= 30 degrees",
            },
        },
        "summaries": summaries,
        "selected_scenarios": cases,
    }
    (config.SCENARIO_RUN_ROOT / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_csv(config.SCENARIO_RUN_ROOT / "cases.csv", cases)
    print_summary(summaries, cases)


if __name__ == "__main__":
    main()
