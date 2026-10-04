#!/usr/bin/env python
"""Render the publication-ready Figure 5 from prepared terrain statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.transforms import Bbox


FIG5 = Path(__file__).resolve().parent
OUT = OUTPUT_DIR / "figure5"
INPUT_DEFAULT = DATA_DIR / "figure5" / "figure5_outputs"
MM_PER_INCH = 25.4
FIG_WIDTH_MM = 180.0
FIG_HEIGHT_MM = 125.0

BLUE = "#386CB0"
ORANGE = "#D9822B"
DARK = "#202020"
MID_GREY = "#747474"
LIGHT_GREY = "#E7E7E7"
GRID_GREY = "#DADADA"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DEFAULT)
    return parser.parse_args()


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)
    ax.tick_params(direction="out", color=DARK)


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=DARK,
    )


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def hillshade(slope_deg: np.ndarray, aspect_deg: np.ndarray) -> np.ndarray:
    altitude = np.deg2rad(38.0)
    azimuth = np.deg2rad(315.0)
    slope = np.deg2rad(slope_deg)
    aspect = np.deg2rad(aspect_deg)
    illumination = (
        np.sin(altitude) * np.cos(slope)
        + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    )
    finite = np.isfinite(illumination)
    lo, hi = np.nanpercentile(illumination[finite], [1.0, 99.0])
    scaled = np.clip((illumination - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
    return 0.31 + 0.62 * scaled


def crop_to_full_valid_square(
    x: np.ndarray,
    y: np.ndarray,
    elevation: np.ndarray,
) -> tuple[slice, slice]:
    """Return the largest vertically centred square containing no DTM nodata."""
    valid = np.isfinite(elevation)
    n_rows, n_cols = valid.shape
    minimum_side = int(0.70 * min(n_rows, n_cols))
    for side in range(min(n_rows, n_cols), minimum_side - 1, -1):
        row_start = (n_rows - side) // 2
        row_stop = row_start + side
        window = valid[row_start:row_stop]
        if not np.all(window.any(axis=1)):
            continue
        first = np.argmax(window, axis=1)
        last = n_cols - 1 - np.argmax(window[:, ::-1], axis=1)
        left = int(first.max())
        right = int(last.min())
        if right - left + 1 < side:
            continue
        col_start = left + (right - left + 1 - side) // 2
        col_stop = col_start + side
        if np.all(valid[row_start:row_stop, col_start:col_stop]):
            return slice(row_start, row_stop), slice(col_start, col_stop)
    raise ValueError("Could not find a fully valid square inside the Panel a DTM crop")


def draw_panel_a(ax: plt.Axes, terrain: dict[str, np.ndarray], summary: dict) -> None:
    row_slice, col_slice = crop_to_full_valid_square(
        terrain["x"], terrain["y"], terrain["elevation"]
    )
    x = terrain["x"][col_slice]
    y = terrain["y"][row_slice]
    elevation = terrain["elevation"][row_slice, col_slice]
    slope = terrain["slope"][row_slice, col_slice]
    aspect = terrain["aspect"][row_slice, col_slice]
    shade = hillshade(slope, aspect)
    cmap = LinearSegmentedColormap.from_list("neutral_hillshade", ["#383838", "#F4F4F1"])
    extent = [float(x[0]), float(x[-1]), float(y[-1]), float(y[0])]
    image = ax.imshow(
        shade,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        rasterized=True,
        zorder=0,
    )
    image.cmap.set_bad(WHITE)

    finite = np.isfinite(elevation)
    contour_low = np.ceil(np.nanpercentile(elevation[finite], 2.0) / 100.0) * 100.0
    contour_high = np.floor(np.nanpercentile(elevation[finite], 98.0) / 100.0) * 100.0
    levels = np.arange(contour_low, contour_high + 1.0, 100.0)
    ax.contour(
        x,
        y,
        elevation,
        levels=levels,
        colors="#555555",
        linewidths=0.28,
        alpha=0.30,
        zorder=1,
    )

    # Low-interference downslope reference field on a fixed 1.2 km lattice.
    stride = max(1, int(round(1200.0 / np.median(np.diff(x)))))
    grid_x, grid_y = np.meshgrid(x[::stride], y[::stride])
    grid_aspect = aspect[::stride, ::stride]
    grid_slope = slope[::stride, ::stride]
    valid_field = np.isfinite(grid_aspect) & np.isfinite(grid_slope) & (grid_slope >= 5.0)
    field_length = 260.0
    field_u = field_length * np.sin(np.deg2rad(grid_aspect))
    field_v = field_length * np.cos(np.deg2rad(grid_aspect))
    ax.quiver(
        grid_x[valid_field],
        grid_y[valid_field],
        field_u[valid_field],
        field_v[valid_field],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=ORANGE,
        alpha=0.48,
        width=0.0024,
        headwidth=3.8,
        headlength=4.4,
        headaxislength=4.0,
        zorder=2,
    )

    arrow_keep = (
        (terrain["arrow_x"] >= x[0])
        & (terrain["arrow_x"] <= x[-1])
        & (terrain["arrow_y"] <= y[0])
        & (terrain["arrow_y"] >= y[-1])
    )
    arrow_x = terrain["arrow_x"][arrow_keep]
    arrow_y = terrain["arrow_y"][arrow_keep]
    arrow_bearing = terrain["arrow_bearing"][arrow_keep]
    arrow_length = 410.0
    u = arrow_length * np.sin(np.deg2rad(arrow_bearing))
    v = arrow_length * np.cos(np.deg2rad(arrow_bearing))
    ax.quiver(
        arrow_x,
        arrow_y,
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=BLUE,
        alpha=0.92,
        width=0.0034,
        headwidth=4.2,
        headlength=5.1,
        headaxislength=4.5,
        zorder=4,
    )

    x_min, x_max = extent[0], extent[1]
    y_min, y_max = extent[2], extent[3]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#777777")

    # Scale bar and north arrow are placed in locally quiet corners.
    bar_length = 2000.0
    bar_x0 = x_min + 0.07 * (x_max - x_min)
    bar_y = y_min + 0.065 * (y_max - y_min)
    ax.plot([bar_x0, bar_x0 + bar_length], [bar_y, bar_y], color=DARK, lw=1.4, zorder=8)
    ax.plot([bar_x0, bar_x0], [bar_y - 70.0, bar_y + 70.0], color=DARK, lw=0.9, zorder=8)
    ax.plot(
        [bar_x0 + bar_length, bar_x0 + bar_length],
        [bar_y - 70.0, bar_y + 70.0],
        color=DARK,
        lw=0.9,
        zorder=8,
    )
    ax.text(bar_x0 + bar_length / 2.0, bar_y + 150.0, "2 km", ha="center", va="bottom", fontsize=6.3)

    north_x = x_max - 0.075 * (x_max - x_min)
    north_y0 = y_min + 0.085 * (y_max - y_min)
    ax.annotate(
        "",
        xy=(north_x, north_y0 + 720.0),
        xytext=(north_x, north_y0),
        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=0.9, mutation_scale=8.0),
        zorder=8,
    )
    ax.text(north_x, north_y0 + 820.0, "N", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    handles = [
        Line2D([0], [0], color=BLUE, lw=1.5, marker=">", markersize=4.5, markevery=[1], label="Rockfall vector"),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            alpha=0.70,
            lw=1.2,
            marker=">",
            markersize=4.0,
            markevery=[1],
            label="Steepest descent",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        facecolor=WHITE,
        edgecolor="none",
        framealpha=0.82,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.35,
    )
    ax.text(
        1.00,
        1.012,
        "Chaplygin NE rim  |  5 m px$^{-1}$ NAC DTM",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=DARK,
        clip_on=False,
    )
    panel_label(ax, "a", x=-0.055, y=1.035)


def draw_panel_b(ax: plt.Axes, distribution: dict[str, np.ndarray], summary: dict) -> None:
    x = distribution["angle_centers_deg"]
    observed = distribution["observed_density"]
    null_mean = distribution["null_mean"]
    null_lo = distribution["null_ci_low"]
    null_hi = distribution["null_ci_high"]
    ax.fill_between(x, null_lo, null_hi, color="#BEBEBE", alpha=0.34, lw=0, label="Null 95% envelope")
    ax.plot(x, null_mean, color=ORANGE, lw=1.0, ls=(0, (3.0, 2.0)), label="Cluster-rotation null")
    ax.fill_between(x, 0.0, observed, color=BLUE, alpha=0.11, lw=0)
    ax.plot(x, observed, color=BLUE, lw=1.45, label="Observed")
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel(r"Rockfall–downslope angle, $\delta$ (°)")
    ax.set_ylabel(r"Probability density (deg$^{-1}$)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))
    ax.grid(axis="y", color=GRID_GREY, lw=0.45, zorder=0)
    style_axis(ax)
    ax.legend(loc="upper right", frameon=False, handlelength=2.0, borderaxespad=0.15, labelspacing=0.25)

    stats = summary["downslope_alignment"]
    annotation = (
        rf"$D$ = {stats['D']:.2f} [{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]"
        + "\n"
        + rf"median $\delta$ = {stats['median_delta_deg']:.1f}°"
        + "\n"
        + rf"$\delta \leq 30$°: {100.0 * stats['fraction_delta_le_30']:.1f}%"
        + "\n"
        + f"n = {summary['core_events']:,}; NAC = {summary['core_nac_images']}"
    )
    ax.text(
        0.97,
        0.56,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        linespacing=1.25,
        color=DARK,
    )
    # Keep the panel letter inside the figure's top safe area so that raster
    # and vector exports cannot clip its upper edge.
    panel_label(ax, "b", x=-0.27, y=1.045)


def draw_panel_c(ax: plt.Axes, rows: list[dict]) -> None:
    by_width = {int(row["half_width_deg"]): row for row in rows}
    ordered = [by_width[60], by_width[45], by_width[75]]
    labels = ["Primary (±60°)", "Narrow (±45°)", "Broad (±75°)"]
    y = np.array([2.0, 1.0, 0.0])
    ax.axvline(1.0, color=MID_GREY, lw=0.8, zorder=0)
    ax.axvspan(0.5, 1.0, color="#F2F2F2", alpha=0.6, lw=0, zorder=0)
    for idx, (row, yy) in enumerate(zip(ordered, y)):
        estimate = float(row["estimate"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        primary = idx == 0
        ax.errorbar(
            estimate,
            yy,
            xerr=[[estimate - lo], [hi - estimate]],
            fmt="o",
            color=BLUE if primary else MID_GREY,
            mfc=BLUE if primary else WHITE,
            mec=BLUE if primary else MID_GREY,
            ms=4.6 if primary else 4.0,
            mew=0.9,
            elinewidth=1.0,
            capsize=1.8,
            zorder=3,
        )
        ax.text(
            3.75,
            yy,
            f"n = {int(row['n_events']):,}",
            transform=ax.transData,
            ha="center",
            va="center",
            fontsize=5.6,
            color=MID_GREY,
            bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.94, pad=0.6),
            zorder=5,
        )
        if primary:
            ax.text(
                estimate,
                yy + 0.28,
                f"{estimate:.2f} [{lo:.2f}, {hi:.2f}]",
                ha="center",
                va="bottom",
                fontsize=5.7,
                color=DARK,
            )
    ax.set_xscale("log")
    ax.set_xlim(0.50, 4.15)
    ax.set_ylim(-0.62, 2.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}" if 0.49 <= value <= 4.2 else ""))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Adjusted conditional odds ratio\n(pole-facing / equator-facing)")
    ax.grid(axis="x", color=GRID_GREY, lw=0.45, zorder=0)
    style_axis(ax)
    ax.text(0.00, 1.02, "equator-facing ←", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8)
    ax.text(1.00, 1.02, "→ pole-facing", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8)
    panel_label(ax, "c", x=-0.18, y=1.10)


def draw_figure(
    terrain: dict[str, np.ndarray],
    distribution: dict[str, np.ndarray],
    model_rows: list[dict],
    summary: dict,
) -> tuple[plt.Figure, dict[str, list[plt.Axes]]]:
    fig = plt.figure(figsize=(FIG_WIDTH_MM / MM_PER_INCH, FIG_HEIGHT_MM / MM_PER_INCH))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.655, 0.345],
        height_ratios=[0.56, 0.44],
        left=0.045,
        right=0.977,
        bottom=0.095,
        top=0.965,
        wspace=0.30,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    draw_panel_a(ax_a, terrain, summary)
    draw_panel_b(ax_b, distribution, summary)
    draw_panel_c(ax_c, model_rows)
    return fig, {"a": [ax_a], "b": [ax_b], "c": [ax_c]}


def save_panel_crops(fig: plt.Figure, panels: dict[str, list[plt.Axes]], output_dir: Path) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, axes in panels.items():
        boxes = [axis.get_tightbbox(renderer) for axis in axes]
        bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).expanded(1.05, 1.08)
        fig.savefig(output_dir / f"figure5_panel_{label}.pdf", bbox_inches=bbox, pad_inches=0.01)
        fig.savefig(output_dir / f"figure5_panel_{label}.png", dpi=600, bbox_inches=bbox, pad_inches=0.01)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    with np.load(args.input_dir / "figure5_panel_a_terrain.npz") as cached:
        terrain = {key: cached[key] for key in cached.files}
    with np.load(args.input_dir / "figure5_panel_b_distribution.npz") as cached:
        distribution = {key: cached[key] for key in cached.files}
    model_rows = load_csv(args.input_dir / "figure5_panel_c_conditional_odds.csv")
    summary = json.loads((args.input_dir / "figure5_summary.json").read_text(encoding="utf-8"))

    fig, panels = draw_figure(terrain, distribution, model_rows, summary)
    base = args.output_dir / "Figure5_local_downslope_transport"
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    save_panel_crops(fig, panels, args.output_dir)
    plt.close(fig)
    print(json.dumps({"outputs": str(base), "size_mm": [FIG_WIDTH_MM, FIG_HEIGHT_MM]}, indent=2))


if __name__ == "__main__":
    main()
