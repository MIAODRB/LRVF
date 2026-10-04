#!/usr/bin/env python
"""Create the Nature-style Figure 4 robustness analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.transforms import Bbox
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix

from prepare_figure4_data import load_event_quality


FIG4 = Path(__file__).resolve().parent
OUT_DEFAULT = OUTPUT_DIR / "figure4"
METADATA_CACHE = DATA_DIR / "figure4" / "figure4_outputs" / "nac_metadata_cache.npz"

MM_PER_INCH = 25.4
FIG_WIDTH_MM = 180.0
FIG_HEIGHT_MM = 125.0
BOOTSTRAP_REPLICATES = 800
RANDOM_SEED = 20260714

MID_LAT_MIN = 30.0
MID_LAT_MAX = 60.0
HIGH_LAT_MIN = 68.0
HIGH_LAT_MAX = 76.0
REVERSAL_LOW = 63.48970201322201
REVERSAL_HIGH = 66.66960727698721
REVERSAL_MID = 65.07965464510461
POLAR_LIMIT = 80.0

INCIDENCE_SPLIT = 52.18
RESOLUTION_SPLIT = 0.652
SCORE_SPLIT = 0.5713289

CORE_INCIDENCE_MIN = 35.0
CORE_INCIDENCE_MAX = 75.0
CORE_RESOLUTION_MAX = 1.0

BLACK = "#252525"
DARK_GREY = "#4A4A4A"
MID_GREY = "#7A7A7A"
LIGHT_GREY = "#E9E9E9"
GRID_GREY = "#D8D8D8"
BLUE = "#3B6FB6"
ORANGE = "#D68132"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    mask: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    selected_values = values[mask].astype(np.float64)
    selected_clusters = clusters[mask]
    if len(selected_values) == 0:
        return {
            "effect": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "n_events": 0,
            "n_images": 0,
        }
    unique_clusters, inverse = np.unique(selected_clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=selected_values)
    counts = np.bincount(inverse).astype(np.float64)
    effect = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    chunk = 80
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        weights = rng.poisson(1.0, size=(stop - start, len(unique_clusters))).astype(np.float32)
        estimates[start:stop] = (weights @ sums) / (weights @ counts)
    ci_low, ci_high = np.nanpercentile(estimates, [2.5, 97.5])
    return {
        "effect": effect,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_events": int(len(selected_values)),
        "n_images": int(len(unique_clusters)),
    }


def smoothed_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = gaussian_filter1d(numerator, sigma=1.0, axis=-1, mode="nearest")
    den = gaussian_filter1d(denominator, sigma=1.0, axis=-1, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def curve_statistics(
    values: np.ndarray,
    clusters: np.ndarray,
    abs_lat: np.ndarray,
    mask: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    lat_edges = np.arange(0.0, 92.0, 2.0)
    bin_index = np.searchsorted(lat_edges, abs_lat, side="right") - 1
    keep = mask & (bin_index >= 0) & (bin_index < len(lat_edges) - 1)
    active_clusters, row_index = np.unique(clusters[keep], return_inverse=True)
    col_index = bin_index[keep]
    selected_values = values[keep].astype(np.float64)
    shape = (len(active_clusters), len(lat_edges) - 1)
    sums = coo_matrix((selected_values, (row_index, col_index)), shape=shape).tocsr()
    counts = coo_matrix((np.ones_like(selected_values), (row_index, col_index)), shape=shape).tocsr()
    event_count = np.asarray(counts.sum(axis=0)).ravel()
    image_count = np.asarray((counts > 0).sum(axis=0)).ravel()
    mean = smoothed_ratio(np.asarray(sums.sum(axis=0)).ravel(), event_count)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_bootstrap, shape[1]), dtype=np.float32)
    chunk = 40
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        weights = rng.poisson(1.0, size=(stop - start, shape[0])).astype(np.float32)
        boot_sum = np.asarray((sums.T @ weights.T).T)
        boot_count = np.asarray((counts.T @ weights.T).T)
        boot[start:stop] = smoothed_ratio(boot_sum, boot_count)
    ci_low = np.full(shape[1], np.nan, dtype=np.float64)
    ci_high = np.full(shape[1], np.nan, dtype=np.float64)
    finite_columns = np.any(np.isfinite(boot), axis=0)
    if np.any(finite_columns):
        ci_low[finite_columns], ci_high[finite_columns] = np.nanpercentile(
            boot[:, finite_columns], [2.5, 97.5], axis=0
        )
    valid = (event_count >= 500) & (image_count >= 10)
    centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    first_failure = np.flatnonzero((centers >= 70.0) & ~valid)
    if len(first_failure):
        valid[first_failure[0] :] = False
    return {
        "centers": centers,
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "event_count": event_count,
        "image_count": image_count,
        "valid": valid,
    }


def persistent_crossing(stats: dict[str, np.ndarray]) -> float | None:
    x = stats["centers"]
    y = stats["mean"]
    valid = stats["valid"]
    reference = np.nanmedian(y[(x >= 30.0) & (x < 60.0) & valid])
    for idx in range(1, len(x)):
        if x[idx] < 55.0 or not (valid[idx - 1] and valid[idx]):
            continue
        if y[idx - 1] * y[idx] <= 0.0:
            future = y[idx : min(idx + 3, len(y))]
            future_valid = valid[idx : min(idx + 3, len(y))]
            if future_valid.sum() >= 2 and np.nanmedian(future[future_valid]) * reference < 0.0:
                return float(x[idx - 1] - y[idx - 1] * (x[idx] - x[idx - 1]) / (y[idx] - y[idx - 1]))
    return None


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.tick_params(direction="out", color=BLACK)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.0, fontweight="bold", ha="left", va="top")


def build_masks(event: dict[str, np.ndarray], meta: dict[str, np.ndarray]) -> tuple[dict, np.ndarray]:
    code = event["nac_code"]
    incidence = meta["incidence"][code]
    resolution = meta["resolution"][code]
    frame = meta["frame"][code]
    samples = meta["line_samples"][code]
    score = event["score"]
    matched = meta["matched"][code]
    all_mask = matched & np.isfinite(incidence) & np.isfinite(resolution)
    core = (
        all_mask
        & (incidence >= CORE_INCIDENCE_MIN)
        & (incidence <= CORE_INCIDENCE_MAX)
        & (resolution <= CORE_RESOLUTION_MAX)
        & (samples == 5064)
        & (score >= SCORE_SPLIT)
    )
    strata = {
        "Incidence angle": [
            (f"Lower (≤{INCIDENCE_SPLIT:.1f}°)", all_mask & (incidence <= INCIDENCE_SPLIT)),
            (f"Higher (>{INCIDENCE_SPLIT:.1f}°)", all_mask & (incidence > INCIDENCE_SPLIT)),
        ],
        "Pixel scale": [
            (f"Finer (≤{RESOLUTION_SPLIT:.3f} m px$^{{-1}}$)", all_mask & (resolution <= RESOLUTION_SPLIT)),
            (f"Coarser (>{RESOLUTION_SPLIT:.3f} m px$^{{-1}}$)", all_mask & (resolution > RESOLUTION_SPLIT)),
        ],
        "NAC frame": [
            ("NAC-L", all_mask & (frame == b"LEFT")),
            ("NAC-R", all_mask & (frame == b"RIGHT")),
        ],
        "Spatial summing": [
            ("Non-summed (5064 samples)", all_mask & (samples == 5064)),
            ("2× summed (2532 samples)", all_mask & (samples == 2532)),
        ],
        "Detection confidence": [
            (f"Lower (<{SCORE_SPLIT:.3f})", all_mask & (score < SCORE_SPLIT)),
            (f"Higher (≥{SCORE_SPLIT:.3f})", all_mask & (score >= SCORE_SPLIT)),
        ],
    }
    return strata, core


def compute_statistics(
    event: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    n_bootstrap: int,
) -> tuple[list[dict], dict, dict, list[dict], np.ndarray]:
    values = event["poleward_component"]
    clusters = event["nac_code"]
    abs_lat = np.abs(event["latitude"])
    mid = (abs_lat >= MID_LAT_MIN) & (abs_lat < MID_LAT_MAX)
    high = (abs_lat >= HIGH_LAT_MIN) & (abs_lat < HIGH_LAT_MAX)
    north = event["latitude"] >= 0.0
    south = ~north
    strata, core = build_masks(event, meta)
    all_mask = meta["matched"][clusters]

    forest: list[dict] = []
    overall = cluster_bootstrap_mean(values, clusters, all_mask & mid, n_bootstrap, RANDOM_SEED)
    forest.append({"group": "Overall", "level": "All deduplicated data", **overall})
    seed = RANDOM_SEED + 10
    for group, levels in strata.items():
        for level, mask in levels:
            stats = cluster_bootstrap_mean(values, clusters, mask & mid, n_bootstrap, seed)
            forest.append({"group": group, "level": level, **stats})
            seed += 1

    all_curve = curve_statistics(values, clusters, abs_lat, all_mask, n_bootstrap, RANDOM_SEED + 100)
    core_curve = curve_statistics(values, clusters, abs_lat, core, n_bootstrap, RANDOM_SEED + 101)

    high_rows: list[dict] = []
    for dataset, dataset_mask, hollow, seed_base in (
        ("All data", all_mask, False, RANDOM_SEED + 200),
        ("Core subset", core, True, RANDOM_SEED + 210),
    ):
        for hemisphere, hemi_mask, marker, color, offset in (
            ("North", north, "o", BLUE, 0),
            ("South", south, "^", ORANGE, 1),
        ):
            stats = cluster_bootstrap_mean(values, clusters, dataset_mask & high & hemi_mask, n_bootstrap, seed_base + offset)
            high_rows.append(
                {
                    "dataset": dataset,
                    "hemisphere": hemisphere,
                    "marker": marker,
                    "color": color,
                    "hollow": hollow,
                    **stats,
                }
            )
    return forest, all_curve, core_curve, high_rows, core


def draw_figure(
    forest: list[dict],
    all_curve: dict,
    core_curve: dict,
    high_rows: list[dict],
) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
    fig = plt.figure(figsize=(FIG_WIDTH_MM / MM_PER_INCH, FIG_HEIGHT_MM / MM_PER_INCH))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.66, 0.34],
        height_ratios=[0.57, 0.43],
        left=0.075,
        right=0.975,
        bottom=0.105,
        top=0.955,
        wspace=0.37,
        hspace=0.46,
    )
    top = grid[0, :].subgridspec(1, 3, width_ratios=[0.40, 0.40, 0.20], wspace=0.02)
    ax_labels = fig.add_subplot(top[0, 0])
    ax_effect = fig.add_subplot(top[0, 1])
    ax_counts = fig.add_subplot(top[0, 2])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    positions = [0.20]
    current = 1.75
    n_groups = (len(forest) - 1) // 2
    for _ in range(n_groups):
        positions.extend([current, current + 0.72])
        current += 1.78
    y_max = positions[-1] + 0.65
    for ax in (ax_labels, ax_effect, ax_counts):
        ax.set_ylim(y_max, -1.05)

    ax_labels.axis("off")
    ax_counts.axis("off")
    ax_labels.text(0.00, -0.72, "Condition", fontsize=6.4, color=MID_GREY, va="center")
    ax_labels.text(0.48, -0.72, "Level", fontsize=6.4, color=MID_GREY, va="center")
    ax_counts.text(0.08, -0.72, "Events", fontsize=6.4, color=MID_GREY, va="center", ha="right")
    ax_counts.text(0.95, -0.72, "NAC images", fontsize=6.4, color=MID_GREY, va="center", ha="right")

    ci_min = min(row["ci_low"] for row in forest)
    ci_max = max(row["ci_high"] for row in forest)
    span = max(ci_max - ci_min, 0.02)
    x_min = min(0.0, ci_min) - 0.12 * span
    x_max = max(0.0, ci_max) + 0.12 * span
    ax_effect.set_xlim(x_min, x_max)
    ax_effect.axvline(0.0, color=MID_GREY, lw=0.8, zorder=0)
    ax_effect.axvspan(forest[0]["ci_low"], forest[0]["ci_high"], color="#BDBDBD", alpha=0.20, lw=0, zorder=0)
    ax_effect.xaxis.set_major_locator(MaxNLocator(5))
    ax_effect.grid(axis="x", color=GRID_GREY, lw=0.45, zorder=0)
    ax_effect.set_xlabel(r"Mid-latitude effect, $E_{\mathrm{mid}}$")
    ax_effect.set_yticks([])
    style_axis(ax_effect)
    ax_effect.spines["left"].set_visible(False)

    group_first_position: dict[str, float] = {}
    for idx, (row, y) in enumerate(zip(forest, positions)):
        if row["group"] == "Overall":
            ax_labels.text(0.00, y, row["level"], ha="left", va="center", fontweight="bold")
            marker, color, face = "D", BLACK, BLACK
        else:
            if row["group"] not in group_first_position:
                group_first_position[row["group"]] = y
                ax_labels.text(0.00, y, row["group"], ha="left", va="center", color=DARK_GREY)
            ax_labels.text(0.48, y, row["level"], ha="left", va="center")
            marker, color, face = "o", BLUE, BLUE
        ax_effect.errorbar(
            row["effect"],
            y,
            xerr=[[row["effect"] - row["ci_low"]], [row["ci_high"] - row["effect"]]],
            fmt=marker,
            color=color,
            mfc=face,
            mec=color,
            ms=4.0 if idx == 0 else 3.6,
            mew=0.8,
            elinewidth=0.9,
            capsize=1.7,
            zorder=3,
        )
        ax_counts.text(0.08, y, f"{row['n_events']:,}", ha="right", va="center")
        ax_counts.text(0.95, y, f"{row['n_images']:,}", ha="right", va="center")

    separator_y = [(positions[0] + positions[1]) * 0.5] + [
        (positions[2 + 2 * idx] + positions[3 + 2 * idx]) * 0.5 for idx in range(n_groups - 1)
    ]
    for y in separator_y:
        for ax in (ax_labels, ax_effect, ax_counts):
            ax.axhline(y, color="#E4E4E4", lw=0.55, clip_on=False)
    panel_label(ax_labels, "a", x=-0.06, y=1.10)

    # Panel b: all events versus frozen core subset.
    ax_b.axhline(0.0, color=MID_GREY, lw=0.8, zorder=1)
    ax_b.axvspan(0.0, 2.0, color=LIGHT_GREY, alpha=0.55, lw=0, zorder=0)
    ax_b.axvspan(REVERSAL_LOW, REVERSAL_HIGH, color="#BDBDBD", alpha=0.22, lw=0, zorder=0)
    ax_b.axvline(REVERSAL_MID, color="#666666", lw=0.8, ls=(0, (2.5, 2.0)), zorder=1)
    ax_b.axvspan(POLAR_LIMIT, 90.0, color=LIGHT_GREY, alpha=0.72, lw=0, zorder=0)
    for stats, color, label, linestyle in (
        (all_curve, DARK_GREY, "All deduplicated data", "-"),
        (core_curve, BLUE, "High-quality core subset", (0, (4.0, 2.0))),
    ):
        valid = stats["valid"]
        mean = np.where(valid, stats["mean"], np.nan)
        lo = np.where(valid, stats["ci_low"], np.nan)
        hi = np.where(valid, stats["ci_high"], np.nan)
        ax_b.fill_between(stats["centers"], lo, hi, color=color, alpha=0.15, linewidth=0, zorder=2)
        ax_b.plot(stats["centers"], mean, color=color, ls=linestyle, lw=1.2, label=label, zorder=3)
    finite_limits = []
    for stats in (all_curve, core_curve):
        valid = stats["valid"]
        finite_limits.extend(stats["ci_low"][valid].tolist())
        finite_limits.extend(stats["ci_high"][valid].tolist())
    y_abs = max(0.08, max(abs(float(v)) for v in finite_limits) * 1.20)
    ax_b.set_xlim(0.0, 90.0)
    ax_b.set_ylim(-y_abs, y_abs)
    ax_b.xaxis.set_major_locator(MultipleLocator(15.0))
    ax_b.yaxis.set_major_locator(MaxNLocator(6))
    ax_b.grid(axis="y", color=GRID_GREY, lw=0.45, zorder=0)
    ax_b.set_xlabel("Absolute latitude, |latitude| (°)")
    ax_b.set_ylabel("Mean poleward component, P")
    ax_b.legend(loc="lower left", frameon=False, handlelength=2.5, borderaxespad=0.3)
    ax_b.annotate(
        "Frozen reversal interval",
        xy=(REVERSAL_MID, 0.0),
        xytext=(60.0, y_abs * 0.76),
        fontsize=6.0,
        ha="right",
        va="top",
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7),
    )
    style_axis(ax_b)
    panel_label(ax_b, "b", x=-0.10, y=1.09)

    # Panel c: frozen post-reversal band, separated by hemisphere and data subset.
    y_positions = [3.1, 2.25, 0.85, 0.0]
    values = [row["effect"] for row in high_rows]
    lows = [row["ci_low"] for row in high_rows]
    highs = [row["ci_high"] for row in high_rows]
    x_span = max(highs) - min(lows)
    c_min = min(0.0, min(lows)) - 0.12 * x_span
    c_max = max(0.0, max(highs)) + 0.35 * x_span
    zero_fraction = (0.0 - c_min) / (c_max - c_min)
    count_x = min(0.84, zero_fraction + 0.025)
    ax_c.axvline(0.0, color=MID_GREY, lw=0.8, zorder=0)
    ax_c.axhline(1.55, color="#E1E1E1", lw=0.6, zorder=0)
    for row, y in zip(high_rows, y_positions):
        face = "white" if row["hollow"] else row["color"]
        ax_c.errorbar(
            row["effect"],
            y,
            xerr=[[row["effect"] - row["ci_low"]], [row["ci_high"] - row["effect"]]],
            fmt=row["marker"],
            color=row["color"],
            mfc=face,
            mec=row["color"],
            ms=4.2,
            mew=1.0,
            elinewidth=0.95,
            capsize=1.8,
            zorder=3,
        )
        ax_c.text(
            count_x,
            y,
            f"NAC n = {row['n_images']:,}",
            transform=ax_c.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=5.2,
            color=MID_GREY,
        )
    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels(["All · North", "All · South", "Core · North", "Core · South"])
    ax_c.set_xlim(c_min, c_max)
    ax_c.set_ylim(-0.65, 3.75)
    ax_c.xaxis.set_major_locator(MaxNLocator(5))
    ax_c.grid(axis="x", color=GRID_GREY, lw=0.45, zorder=0)
    ax_c.set_xlabel(r"High-latitude effect, $E_{\mathrm{high}}$")
    ax_c.text(0.00, 1.02, "← equatorward / reversed", transform=ax_c.transAxes, ha="left", va="bottom", fontsize=5.7)
    ax_c.text(0.95, 1.02, "poleward →", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=5.7)
    style_axis(ax_c)
    panel_label(ax_c, "c", x=-0.22, y=1.10)

    return fig, {"a": [ax_labels, ax_effect, ax_counts], "b": [ax_b], "c": [ax_c]}


def save_panel_crops(fig: plt.Figure, panels: dict[str, list[plt.Axes]], output_dir: Path) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, axes in panels.items():
        boxes = [ax.get_tightbbox(renderer) for ax in axes]
        bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).expanded(1.04, 1.08)
        fig.savefig(output_dir / f"figure4_panel_{label}.pdf", bbox_inches=bbox, pad_inches=0.01)
        fig.savefig(output_dir / f"figure4_panel_{label}.png", dpi=600, bbox_inches=bbox, pad_inches=0.01)


def write_outputs(
    output_dir: Path,
    forest: list[dict],
    all_curve: dict,
    core_curve: dict,
    high_rows: list[dict],
    core_mask: np.ndarray,
    event: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    n_bootstrap: int,
) -> dict:
    with (output_dir / "figure4_panel_a_forest_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["group", "level", "effect", "ci_low", "ci_high", "n_events", "n_images"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in forest)

    with (output_dir / "figure4_panel_b_curve_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "abs_latitude_deg",
                "all_mean_P",
                "all_ci_low",
                "all_ci_high",
                "all_events",
                "all_images",
                "core_mean_P",
                "core_ci_low",
                "core_ci_high",
                "core_events",
                "core_images",
            ]
        )
        for idx, center in enumerate(all_curve["centers"]):
            writer.writerow(
                [
                    center,
                    all_curve["mean"][idx],
                    all_curve["ci_low"][idx],
                    all_curve["ci_high"][idx],
                    int(all_curve["event_count"][idx]),
                    int(all_curve["image_count"][idx]),
                    core_curve["mean"][idx],
                    core_curve["ci_low"][idx],
                    core_curve["ci_high"][idx],
                    int(core_curve["event_count"][idx]),
                    int(core_curve["image_count"][idx]),
                ]
            )

    with (output_dir / "figure4_panel_c_high_latitude_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "hemisphere", "effect", "ci_low", "ci_high", "n_events", "n_images"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in high_rows)

    code = event["nac_code"]
    summary = {
        "events_total": int(len(code)),
        "nac_images_total": int(len(event["nac_ids"])),
        "metadata_match_events": int(meta["matched"][code].sum()),
        "metadata_match_unique_images": int(meta["matched"].sum()),
        "core_events": int(core_mask.sum()),
        "core_images": int(np.unique(code[core_mask]).size),
        "core_fraction_events": float(core_mask.mean()),
        "mid_latitude_band_deg": [MID_LAT_MIN, MID_LAT_MAX],
        "high_latitude_band_deg": [HIGH_LAT_MIN, HIGH_LAT_MAX],
        "frozen_reversal_interval_deg": [REVERSAL_LOW, REVERSAL_HIGH],
        "all_curve_crossing_deg": persistent_crossing(all_curve),
        "core_curve_crossing_deg": persistent_crossing(core_curve),
        "bootstrap_replicates": n_bootstrap,
        "bootstrap_cluster": "nac_id",
        "core_rules": {
            "incidence_deg": [CORE_INCIDENCE_MIN, CORE_INCIDENCE_MAX],
            "resolution_m_per_pixel_max": CORE_RESOLUTION_MAX,
            "line_samples": 5064,
            "score_min": SCORE_SPLIT,
        },
    }
    (output_dir / "figure4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    event = load_event_quality()
    with np.load(METADATA_CACHE) as cached:
        meta = {key: cached[key] for key in cached.files}
    forest, all_curve, core_curve, high_rows, core = compute_statistics(event, meta, args.bootstrap)
    fig, panels = draw_figure(forest, all_curve, core_curve, high_rows)
    base = args.output_dir / "Figure4_observation_condition_robustness"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    save_panel_crops(fig, panels, args.output_dir)
    plt.close(fig)
    summary = write_outputs(
        args.output_dir,
        forest,
        all_curve,
        core_curve,
        high_rows,
        core,
        event,
        meta,
        args.bootstrap,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
