#!/usr/bin/env python
"""Prepare terrain-matched event statistics and compact plot caches for Figure 5.

The workflow uses the frozen Chaplygin NAC DTM selected before examining
downslope-alignment results.  Selection criteria were DTM pixel scale, LOLA
registration quality, absence of reported line/sample jitter, and the number
of catalogued rockfalls falling inside the DTM footprint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates


ROOT = DATA_DIR
FIG5 = Path(__file__).resolve().parent
OUT = OUTPUT_DIR / "figure5"
SHP_PATH = ROOT / "GLOBAL_rockfall_directions_dedup.shp"
DBF_PATH = ROOT / "GLOBAL_rockfall_directions_dedup.dbf"
TAB_PATH = DATA_DIR / "CUMINDEX.TAB"
META_CACHE = ROOT / "figure4" / "figure4_outputs" / "nac_metadata_cache.npz"
DTM_PATH = DATA_DIR / "figure5" / "data" / "NAC_DTM_CHAPLYGIN.TIF"
DTM_LABEL = DATA_DIR / "figure5" / "data" / "NAC_DTM_CHAPLYGIN.LBL"

MOON_RADIUS_M = 1_737_400.0
DTM_LON0_DEG = 180.0
DTM_STANDARD_PARALLEL_DEG = -4.0
DTM_SCALE_M = 5.0
DTM_X0_M = -864_445.0
DTM_Y0_M = -101_860.0
RECORD_BYTES = 902
MAIN_SCALE_M = 25.0
SENSITIVITY_SCALES_M = (15.0, 25.0, 50.0)
MIN_SLOPE_DEG = 5.0
MIN_SCORE = 0.50
MIN_MARKER_SPAN_M = 10.0
MAX_RESOLUTION_M = 1.20
INCIDENCE_RANGE_DEG = (35.0, 75.0)
CONTROL_STEP_PX = 10
RANDOM_SEED = 20260714
N_BOOTSTRAP = 3000
N_NULL = 1500
ANGLE_BIN_WIDTH_DEG = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-metadata", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--null-replicates", type=int, default=N_NULL)
    return parser.parse_args()


def clean_float(value: str, missing: tuple[float, ...] = ()) -> float:
    try:
        result = float(value.strip())
    except ValueError:
        return float("nan")
    if any(np.isclose(result, sentinel) for sentinel in missing):
        return float("nan")
    return result


def read_events() -> dict[str, np.ndarray]:
    shp_dtype = np.dtype(
        [
            ("record_number", ">i4"),
            ("content_length", ">i4"),
            ("shape_type", "<i4"),
            ("bbox", "<f8", (4,)),
            ("n_parts", "<i4"),
            ("n_points", "<i4"),
            ("part0", "<i4"),
            ("points", "<f8", (4,)),
        ]
    )
    n_events = (SHP_PATH.stat().st_size - 100) // shp_dtype.itemsize
    shp = np.memmap(SHP_PATH, mode="r", dtype=shp_dtype, offset=100, shape=(n_events,))
    if not (
        np.all(shp["shape_type"] == 3)
        and np.all(shp["n_parts"] == 1)
        and np.all(shp["n_points"] == 2)
    ):
        raise ValueError("Figure 5 expects fixed two-point PolyLine events")
    points = np.asarray(shp["points"])
    lon1, lat1, lon2, lat2 = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlon = np.deg2rad(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    bearing = (
        np.rad2deg(
            np.arctan2(
                np.sin(dlon) * np.cos(phi2),
                np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon),
            )
        )
        + 360.0
    ) % 360.0
    hav = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2.0) ** 2
    # Endpoint span of the stored direction marker; this is not physical runout length.
    marker_span_m = 2.0 * MOON_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))

    dbf_dtype = np.dtype(
        [("deleted", "S1"), ("nac_id", "S80"), ("score", "S24"), ("lat_zone", "S80")]
    )
    dbf = np.memmap(DBF_PATH, mode="r", dtype=dbf_dtype, offset=129, shape=(n_events,))
    if np.any(dbf["deleted"] == b"*"):
        raise ValueError("Deleted DBF records are not supported")
    score = np.array([float(value) for value in dbf["score"]], dtype=np.float32)
    nac_ids, nac_code = np.unique(np.char.strip(dbf["nac_id"].astype("S13")), return_inverse=True)
    return {
        "lon1": (lon1 % 360.0).astype(np.float64),
        "lat1": lat1.astype(np.float64),
        "lon2": (lon2 % 360.0).astype(np.float64),
        "lat2": lat2.astype(np.float64),
        "bearing": bearing.astype(np.float32),
        "marker_span_m": marker_span_m.astype(np.float32),
        "score": score,
        "nac_ids": nac_ids,
        "nac_code": nac_code.astype(np.int32),
    }


def lonlat_to_xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wrapped = ((np.asarray(lon) - DTM_LON0_DEG + 180.0) % 360.0) - 180.0
    x = (
        MOON_RADIUS_M
        * math.cos(math.radians(DTM_STANDARD_PARALLEL_DEG))
        * np.deg2rad(wrapped)
    )
    y = MOON_RADIUS_M * np.deg2rad(np.asarray(lat))
    return x, y


def xy_to_rowcol(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col = (np.asarray(x) - DTM_X0_M) / DTM_SCALE_M - 0.5
    row = (DTM_Y0_M - np.asarray(y)) / DTM_SCALE_M - 0.5
    return row, col


def load_dtm() -> tuple[np.ndarray, np.ndarray]:
    if not DTM_PATH.exists() or not DTM_LABEL.exists():
        raise FileNotFoundError("Chaplygin DTM and detached label are required under DATA_DIR/figure5/data")
    image = Image.open(DTM_PATH)
    if image.mode != "F" or image.size != (2438, 9791):
        raise ValueError(f"Unexpected DTM geometry: mode={image.mode}, size={image.size}")
    elevation = np.asarray(image, dtype=np.float32)
    nodata = float(image.tag_v2[42113])
    valid = np.isfinite(elevation) & (elevation > nodata / 2.0)
    return elevation, valid


def smooth_terrain(
    elevation: np.ndarray,
    valid: np.ndarray,
    analysis_scale_m: float,
) -> dict[str, np.ndarray]:
    sigma = analysis_scale_m / DTM_SCALE_M
    weights = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
    weighted = gaussian_filter(np.where(valid, elevation, 0.0).astype(np.float32), sigma=sigma, mode="nearest")
    smoothed = weighted / np.maximum(weights, 1.0e-6)
    stable = valid & (weights >= 0.98)
    smoothed[~stable] = np.nan
    grad_y, grad_x = np.gradient(smoothed, -DTM_SCALE_M, DTM_SCALE_M)
    slope = np.rad2deg(np.arctan(np.hypot(grad_x, grad_y))).astype(np.float32)
    aspect = ((np.rad2deg(np.arctan2(-grad_x, -grad_y)) + 360.0) % 360.0).astype(np.float32)
    dgy_dy, _ = np.gradient(grad_y, -DTM_SCALE_M, DTM_SCALE_M)
    _, dgx_dx = np.gradient(grad_x, -DTM_SCALE_M, DTM_SCALE_M)
    curvature = (dgx_dx + dgy_dy).astype(np.float32)
    return {
        "smoothed": smoothed.astype(np.float32),
        "slope": slope,
        "aspect": aspect,
        "curvature": curvature,
        "stable": stable,
    }


def sample_grid(values: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return map_coordinates(values, np.vstack([row, col]), order=1, mode="constant", cval=np.nan)


def scan_cumindex(target_ids: np.ndarray, rebuild: bool = False) -> dict[str, np.ndarray]:
    cache_path = OUT / "figure5_nac_footprints.npz"
    if cache_path.exists() and not rebuild:
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}

    target = np.char.strip(target_ids.astype("S13"))
    order = np.argsort(target)
    target_sorted = target[order]
    n = len(target)
    arrays: dict[str, np.ndarray] = {
        "product_id": target,
        "matched": np.zeros(n, dtype=bool),
        "resolution": np.full(n, np.nan, dtype=np.float32),
        "incidence": np.full(n, np.nan, dtype=np.float32),
        "emission": np.full(n, np.nan, dtype=np.float32),
        "phase": np.full(n, np.nan, dtype=np.float32),
        "north_azimuth": np.full(n, np.nan, dtype=np.float32),
        "subsolar_azimuth": np.full(n, np.nan, dtype=np.float32),
        "line_samples": np.full(n, -1, dtype=np.int32),
        "image_lines": np.full(n, -1, dtype=np.int32),
        "corner_lat": np.full((n, 4), np.nan, dtype=np.float64),
        "corner_lon": np.full((n, 4), np.nan, dtype=np.float64),
    }
    n_records = TAB_PATH.stat().st_size // RECORD_BYTES
    if n_records * RECORD_BYTES != TAB_PATH.stat().st_size:
        raise ValueError("CUMINDEX.TAB does not contain complete 902-byte records")

    with TAB_PATH.open("rb") as handle:
        block_records = 50_000
        for block_start in range(0, n_records, block_records):
            count = min(block_records, n_records - block_start)
            raw = handle.read(count * RECORD_BYTES)
            records = np.frombuffer(raw, dtype=np.uint8).reshape(count, RECORD_BYTES)
            keys = np.ascontiguousarray(records[:, 122:135]).view("S13").ravel()
            keys = np.char.strip(keys)
            positions = np.searchsorted(target_sorted, keys)
            candidate = np.flatnonzero(positions < n)
            if len(candidate):
                pos = positions[candidate]
                exact = target_sorted[pos] == keys[candidate]
                candidate, pos = candidate[exact], pos[exact]
            for local_row, sorted_position in zip(candidate, pos):
                code = int(order[int(sorted_position)])
                if arrays["matched"][code]:
                    continue
                start = int(local_row) * RECORD_BYTES
                line = raw[start : start + RECORD_BYTES].decode("ascii")
                row = next(csv.reader([line]))
                if len(row) != 83:
                    raise ValueError(f"Expected 83 CUMINDEX fields, found {len(row)}")
                arrays["matched"][code] = True
                arrays["image_lines"][code] = int(row[52].strip())
                arrays["line_samples"][code] = int(row[53].strip())
                arrays["resolution"][code] = clean_float(row[57], (0.0,))
                arrays["emission"][code] = clean_float(row[58], (99.99,))
                arrays["incidence"][code] = clean_float(row[59], (999.99,))
                arrays["phase"][code] = clean_float(row[60], (999.99,))
                arrays["north_azimuth"][code] = clean_float(row[61], (999.99,))
                arrays["subsolar_azimuth"][code] = clean_float(row[62], (999.99,))
                arrays["corner_lat"][code] = [
                    clean_float(row[71], (999.99,)),
                    clean_float(row[73], (999.99,)),
                    clean_float(row[75], (999.99,)),
                    clean_float(row[77], (999.99,)),
                ]
                arrays["corner_lon"][code] = [
                    clean_float(row[72], (999.99,)),
                    clean_float(row[74], (999.99,)),
                    clean_float(row[76], (999.99,)),
                    clean_float(row[78], (999.99,)),
                ]
            if arrays["matched"].all():
                break
    if not arrays["matched"].all():
        missing = [value.decode("ascii") for value in target[~arrays["matched"]]]
        raise ValueError(f"Missing NAC metadata for: {missing}")
    np.savez_compressed(cache_path, **arrays)
    return arrays


def cluster_bootstrap_effect(
    delta_deg: np.ndarray,
    nac_code: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, np.ndarray]:
    unique, inverse = np.unique(nac_code, return_inverse=True)
    values = np.cos(np.deg2rad(delta_deg))
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(seed)
    weights = rng.poisson(1.0, size=(n_bootstrap, len(unique))).astype(np.float32)
    denom = weights @ counts
    boot = (weights @ sums) / np.where(denom > 0.0, denom, np.nan)
    effect = float(values.mean())
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return effect, float(lo), float(hi), boot.astype(np.float32)


def build_null_distribution(
    bearing: np.ndarray,
    aspect: np.ndarray,
    nac_code: np.ndarray,
    n_null: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.arange(0.0, 180.0 + ANGLE_BIN_WIDTH_DEG, ANGLE_BIN_WIDTH_DEG)
    centers = 0.5 * (edges[:-1] + edges[1:])
    observed_delta = np.abs(((bearing - aspect + 180.0) % 360.0) - 180.0)
    observed, _ = np.histogram(observed_delta, bins=edges)
    observed = gaussian_filter1d(observed.astype(float), sigma=0.80, mode="nearest")
    observed = observed / (observed.sum() * ANGLE_BIN_WIDTH_DEG)

    unique, inverse = np.unique(nac_code, return_inverse=True)
    rng = np.random.default_rng(seed)
    null = np.empty((n_null, len(centers)), dtype=np.float32)
    for replicate in range(n_null):
        rotation = rng.uniform(0.0, 360.0, size=len(unique))
        rotated = (bearing + rotation[inverse]) % 360.0
        delta = np.abs(((rotated - aspect + 180.0) % 360.0) - 180.0)
        hist, _ = np.histogram(delta, bins=edges)
        hist = gaussian_filter1d(hist.astype(float), sigma=0.80, mode="nearest")
        null[replicate] = hist / (hist.sum() * ANGLE_BIN_WIDTH_DEG)
    mean = np.mean(null, axis=0)
    lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
    return centers, observed, mean.astype(np.float32), np.vstack([lo, hi]).astype(np.float32)


def terrain_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    index = np.searchsorted(edges, values, side="right") - 1
    index[(index < 0) | (index >= len(edges) - 1)] = -1
    return index


def orientation_class(aspect: np.ndarray, latitude: np.ndarray | float, half_width: float) -> np.ndarray:
    pole_bearing = np.where(np.asarray(latitude) >= 0.0, 0.0, 180.0)
    component = np.cos(np.deg2rad(aspect - pole_bearing))
    threshold = math.cos(math.radians(half_width))
    result = np.zeros(np.broadcast_shapes(np.shape(aspect), np.shape(pole_bearing)), dtype=np.int8)
    result[component >= threshold] = 1
    result[component <= -threshold] = -1
    return result


def conditional_or(
    yp: np.ndarray,
    ye: np.ndarray,
    ap: np.ndarray,
    ae: np.ndarray,
    cluster: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, np.ndarray]:
    yp, ye, ap, ae = [np.asarray(value, dtype=float) for value in (yp, ye, ap, ae)]
    total = yp + ye

    def newton(weights: np.ndarray | None = None, start: float = 0.0) -> float:
        beta = float(start)
        if weights is None:
            weights = np.ones_like(total)
        for _ in range(30):
            eb = math.exp(float(np.clip(beta, -20.0, 20.0)))
            probability = eb * ap / (eb * ap + ae)
            score = np.sum(weights * (yp - total * probability))
            information = np.sum(weights * total * probability * (1.0 - probability))
            if information <= 1.0e-10:
                break
            step = score / information
            beta += float(np.clip(step, -2.0, 2.0))
            if abs(step) < 1.0e-9:
                break
        return beta

    beta = newton()
    unique, inverse = np.unique(cluster, return_inverse=True)
    rng = np.random.default_rng(seed)
    boot = np.full(n_bootstrap, np.nan, dtype=np.float32)
    for replicate in range(n_bootstrap):
        cluster_weight = rng.poisson(1.0, size=len(unique)).astype(float)
        weights = cluster_weight[inverse]
        if np.sum(weights * total) > 0:
            boot[replicate] = newton(weights, beta)
    lo, hi = np.nanpercentile(np.exp(boot), [2.5, 97.5])
    return float(math.exp(beta)), float(lo), float(hi), boot


def build_conditional_model(
    half_width: float,
    core: np.ndarray,
    event: dict[str, np.ndarray],
    local_nac_code: np.ndarray,
    event_slope: np.ndarray,
    event_elevation: np.ndarray,
    event_curvature: np.ndarray,
    event_aspect: np.ndarray,
    metadata: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    slope_edges: np.ndarray,
    elevation_edges: np.ndarray,
    curvature_edges: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    event_orientation = orientation_class(event_aspect, event["lat1"], half_width)
    control_orientation = orientation_class(control["aspect"], control["latitude"], half_width)
    es = terrain_bin(event_slope, slope_edges)
    ee = terrain_bin(event_elevation, elevation_edges)
    ec = terrain_bin(event_curvature, curvature_edges)
    cs = terrain_bin(control["slope"], slope_edges)
    ce = terrain_bin(control["elevation"], elevation_edges)
    cc = terrain_bin(control["curvature"], curvature_edges)

    rows: dict[tuple[int, int, int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for code in range(len(metadata["product_id"])):
        polygon_x, polygon_y = lonlat_to_xy(metadata["corner_lon"][code], metadata["corner_lat"][code])
        polygon = np.column_stack([polygon_x, polygon_y])
        finite_polygon = np.all(np.isfinite(polygon), axis=1)
        if finite_polygon.sum() < 3:
            continue
        polygon = polygon[finite_polygon]
        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)
        candidate = (
            (control["x"] >= xmin)
            & (control["x"] <= xmax)
            & (control["y"] >= ymin)
            & (control["y"] <= ymax)
        )
        control_idx = np.flatnonzero(candidate)
        if len(control_idx):
            inside = MplPath(polygon, closed=True).contains_points(
                np.column_stack([control["x"][control_idx], control["y"][control_idx]]), radius=1.0e-9
            )
            control_idx = control_idx[inside]
        for idx in control_idx:
            orient = int(control_orientation[idx])
            if orient == 0 or min(cs[idx], ce[idx], cc[idx]) < 0:
                continue
            key = (code, int(cs[idx]), int(ce[idx]), int(cc[idx]))
            if orient > 0:
                rows[key][2] += 1.0
            else:
                rows[key][3] += 1.0

        event_idx = np.flatnonzero(core & (local_nac_code == code))
        for idx in event_idx:
            orient = int(event_orientation[idx])
            if orient == 0 or min(es[idx], ee[idx], ec[idx]) < 0:
                continue
            key = (code, int(es[idx]), int(ee[idx]), int(ec[idx]))
            if orient > 0:
                rows[key][0] += 1.0
            else:
                rows[key][1] += 1.0

    retained = [
        (key, values)
        for key, values in rows.items()
        if values[2] > 0.0 and values[3] > 0.0 and (values[0] + values[1]) > 0.0
    ]
    if not retained:
        raise ValueError("No conditional terrain strata retained")
    cluster = np.array([key[0] for key, _ in retained], dtype=np.int32)
    values = np.asarray([value for _, value in retained], dtype=float)
    estimate, lo, hi, boot = conditional_or(
        values[:, 0], values[:, 1], values[:, 2], values[:, 3], cluster, n_bootstrap, seed
    )
    return {
        "half_width_deg": float(half_width),
        "estimate": estimate,
        "ci_low": lo,
        "ci_high": hi,
        "n_events": int(np.sum(values[:, :2])),
        "n_strata": int(len(values)),
        "n_images": int(len(np.unique(cluster))),
        "bootstrap_log_or": boot,
    }


def select_panel_a_window(y: np.ndarray, eligible: np.ndarray, height_m: float = 12_000.0) -> tuple[float, float]:
    values = np.sort(y[eligible])
    best_count, best_low = -1, float(values.min())
    stop = 0
    for start, low in enumerate(values):
        while stop < len(values) and values[stop] <= low + height_m:
            stop += 1
        if stop - start > best_count:
            best_count, best_low = stop - start, float(low)
    return best_low, best_low + height_m


def spatial_thin(x: np.ndarray, y: np.ndarray, score: np.ndarray, mask: np.ndarray, cell_m: float = 700.0) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return idx
    gx = np.floor((x[idx] - x[idx].min()) / cell_m).astype(np.int64)
    gy = np.floor((y[idx] - y[idx].min()) / cell_m).astype(np.int64)
    key = gx + (gx.max() + 1) * gy
    chosen: list[int] = []
    for value in np.unique(key):
        group = idx[key == value]
        chosen.append(int(group[np.argmax(score[group])]))
    return np.asarray(chosen, dtype=np.int64)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    event = read_events()
    elevation, valid = load_dtm()
    x, y = lonlat_to_xy(event["lon1"], event["lat1"])
    row, col = xy_to_rowcol(x, y)
    event["x"], event["y"], event["row"], event["col"] = x, y, row, col

    terrain = smooth_terrain(elevation, valid, MAIN_SCALE_M)
    event_elevation = sample_grid(terrain["smoothed"], row, col)
    event_slope = sample_grid(terrain["slope"], row, col)
    event_aspect = sample_grid(terrain["aspect"], row, col)
    event_curvature = sample_grid(terrain["curvature"], row, col)
    terrain_valid = np.isfinite(event_slope) & np.isfinite(event_aspect) & (event_slope >= MIN_SLOPE_DEG)

    region_codes = np.unique(event["nac_code"][terrain_valid])
    region_ids = event["nac_ids"][region_codes]
    metadata = scan_cumindex(region_ids, rebuild=args.rebuild_metadata)
    global_to_local = np.full(len(event["nac_ids"]), -1, dtype=np.int32)
    global_to_local[region_codes] = np.arange(len(region_codes), dtype=np.int32)
    local_nac_code = global_to_local[event["nac_code"]]
    event_metadata_valid = terrain_valid & (local_nac_code >= 0)
    local = np.clip(local_nac_code, 0, len(region_codes) - 1)
    core = (
        event_metadata_valid
        & metadata["matched"][local]
        & np.isfinite(metadata["resolution"][local])
        & np.isfinite(metadata["incidence"][local])
        & (metadata["resolution"][local] <= MAX_RESOLUTION_M)
        & (metadata["incidence"][local] >= INCIDENCE_RANGE_DEG[0])
        & (metadata["incidence"][local] <= INCIDENCE_RANGE_DEG[1])
        & (metadata["line_samples"][local] == 5064)
        & (event["score"] >= MIN_SCORE)
        & (event["marker_span_m"] >= MIN_MARKER_SPAN_M)
    )

    delta = np.abs(((event["bearing"] - event_aspect + 180.0) % 360.0) - 180.0)
    effect, effect_lo, effect_hi, effect_boot = cluster_bootstrap_effect(
        delta[core], local_nac_code[core], args.bootstrap, RANDOM_SEED
    )
    centers, observed_density, null_mean, null_envelope = build_null_distribution(
        event["bearing"][core], event_aspect[core], local_nac_code[core], args.null_replicates, RANDOM_SEED + 1
    )
    np.savez_compressed(
        OUT / "figure5_panel_b_distribution.npz",
        angle_centers_deg=centers,
        observed_density=observed_density,
        null_mean=null_mean,
        null_ci_low=null_envelope[0],
        null_ci_high=null_envelope[1],
        bootstrap_D=effect_boot,
    )

    sensitivity_rows = []
    sensitivity_aspect: dict[float, np.ndarray] = {MAIN_SCALE_M: event_aspect}
    for analysis_scale in SENSITIVITY_SCALES_M:
        if analysis_scale == MAIN_SCALE_M:
            scale_slope, scale_aspect = event_slope, event_aspect
        else:
            other = smooth_terrain(elevation, valid, analysis_scale)
            scale_slope = sample_grid(other["slope"], row, col)
            scale_aspect = sample_grid(other["aspect"], row, col)
            sensitivity_aspect[analysis_scale] = scale_aspect
        mask = core & np.isfinite(scale_slope) & np.isfinite(scale_aspect) & (scale_slope >= MIN_SLOPE_DEG)
        scale_delta = np.abs(((event["bearing"] - scale_aspect + 180.0) % 360.0) - 180.0)
        scale_effect, scale_lo, scale_hi, _ = cluster_bootstrap_effect(
            scale_delta[mask], local_nac_code[mask], args.bootstrap, RANDOM_SEED + int(analysis_scale)
        )
        sensitivity_rows.append(
            {
                "analysis_scale_m": analysis_scale,
                "D": scale_effect,
                "ci_low": scale_lo,
                "ci_high": scale_hi,
                "n_events": int(mask.sum()),
                "n_images": int(len(np.unique(local_nac_code[mask]))),
                "median_delta_deg": float(np.median(scale_delta[mask])),
                "fraction_delta_le_30": float(np.mean(scale_delta[mask] <= 30.0)),
            }
        )
    with (OUT / "figure5_panel_b_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    # Compact 50 m terrain-opportunity grid for the within-image conditional model.
    rr = np.arange(CONTROL_STEP_PX // 2, elevation.shape[0], CONTROL_STEP_PX)
    cc = np.arange(CONTROL_STEP_PX // 2, elevation.shape[1], CONTROL_STEP_PX)
    grid_col, grid_row = np.meshgrid(cc, rr)
    control_x = DTM_X0_M + (grid_col.ravel() + 0.5) * DTM_SCALE_M
    control_y = DTM_Y0_M - (grid_row.ravel() + 0.5) * DTM_SCALE_M
    control_slope = terrain["slope"][grid_row, grid_col].ravel()
    control_aspect = terrain["aspect"][grid_row, grid_col].ravel()
    control_elevation = terrain["smoothed"][grid_row, grid_col].ravel()
    control_curvature = terrain["curvature"][grid_row, grid_col].ravel()
    control_valid = (
        np.isfinite(control_slope)
        & np.isfinite(control_aspect)
        & np.isfinite(control_elevation)
        & np.isfinite(control_curvature)
        & (control_slope >= MIN_SLOPE_DEG)
    )
    control = {
        "x": control_x[control_valid],
        "y": control_y[control_valid],
        "latitude": np.rad2deg(control_y[control_valid] / MOON_RADIUS_M),
        "slope": control_slope[control_valid],
        "aspect": control_aspect[control_valid],
        "elevation": control_elevation[control_valid],
        "curvature": control_curvature[control_valid],
    }
    slope_edges = np.array([MIN_SLOPE_DEG, 15.0, 25.0, 35.0, 90.0])
    elevation_edges = np.r_[-np.inf, np.quantile(control["elevation"], [1 / 3, 2 / 3]), np.inf]
    curvature_edges = np.r_[-np.inf, np.quantile(control["curvature"], [1 / 3, 2 / 3]), np.inf]
    model_rows = []
    for offset, half_width in enumerate((45.0, 60.0, 75.0)):
        model_rows.append(
            build_conditional_model(
                half_width,
                core,
                event,
                local_nac_code,
                event_slope,
                event_elevation,
                event_curvature,
                event_aspect,
                metadata,
                control,
                slope_edges,
                elevation_edges,
                curvature_edges,
                args.bootstrap,
                RANDOM_SEED + 100 + offset,
            )
        )
    with (OUT / "figure5_panel_c_conditional_odds.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["half_width_deg", "estimate", "ci_low", "ci_high", "n_events", "n_strata", "n_images"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in model_rows)

    # Freeze a 12 km panel-a window by maximum event count, without using alignment strength.
    window_low, window_high = select_panel_a_window(y, core)
    x_low = DTM_X0_M
    x_high = DTM_X0_M + elevation.shape[1] * DTM_SCALE_M
    panel_mask = core & (x >= x_low) & (x <= x_high) & (y >= window_low) & (y <= window_high)
    arrow_idx = spatial_thin(x, y, event["score"], panel_mask)

    row_top = max(0, int(math.floor((DTM_Y0_M - window_high) / DTM_SCALE_M)))
    row_bottom = min(elevation.shape[0], int(math.ceil((DTM_Y0_M - window_low) / DTM_SCALE_M)))
    col_left, col_right = 0, elevation.shape[1]
    panel_step = 2
    rows = np.arange(row_top, row_bottom, panel_step)
    cols = np.arange(col_left, col_right, panel_step)
    crop_elevation = terrain["smoothed"][np.ix_(rows, cols)]
    crop_slope = terrain["slope"][np.ix_(rows, cols)]
    crop_aspect = terrain["aspect"][np.ix_(rows, cols)]
    crop_x = DTM_X0_M + (cols + 0.5) * DTM_SCALE_M
    crop_y = DTM_Y0_M - (rows + 0.5) * DTM_SCALE_M
    np.savez_compressed(
        OUT / "figure5_panel_a_terrain.npz",
        x=crop_x.astype(np.float32),
        y=crop_y.astype(np.float32),
        elevation=crop_elevation.astype(np.float32),
        slope=crop_slope.astype(np.float32),
        aspect=crop_aspect.astype(np.float32),
        arrow_x=x[arrow_idx].astype(np.float32),
        arrow_y=y[arrow_idx].astype(np.float32),
        arrow_bearing=event["bearing"][arrow_idx].astype(np.float32),
        arrow_downslope=event_aspect[arrow_idx].astype(np.float32),
        arrow_delta=delta[arrow_idx].astype(np.float32),
    )

    core_idx = np.flatnonzero(core)
    np.savez_compressed(
        OUT / "figure5_event_terrain_cache.npz",
        event_index=core_idx.astype(np.int32),
        x=x[core_idx].astype(np.float32),
        y=y[core_idx].astype(np.float32),
        latitude=event["lat1"][core_idx].astype(np.float32),
        bearing=event["bearing"][core_idx].astype(np.float32),
        downslope_aspect=event_aspect[core_idx].astype(np.float32),
        delta=delta[core_idx].astype(np.float32),
        slope=event_slope[core_idx].astype(np.float32),
        elevation=event_elevation[core_idx].astype(np.float32),
        curvature=event_curvature[core_idx].astype(np.float32),
        score=event["score"][core_idx].astype(np.float32),
        marker_span_m=event["marker_span_m"][core_idx].astype(np.float32),
        local_nac_code=local_nac_code[core_idx].astype(np.int16),
        nac_ids=region_ids,
    )

    summary = {
        "dtm_product": "NAC_DTM_CHAPLYGIN",
        "dtm_pixel_scale_m": DTM_SCALE_M,
        "dtm_analysis_scale_m": MAIN_SCALE_M,
        "dtm_valid_fraction": float(valid.mean()),
        "terrain_matched_events_before_quality_filter": int(terrain_valid.sum()),
        "core_events": int(core.sum()),
        "core_nac_images": int(len(np.unique(local_nac_code[core]))),
        "quality_rules": {
            "minimum_slope_deg": MIN_SLOPE_DEG,
            "minimum_detection_score": MIN_SCORE,
            "minimum_marker_span_m": MIN_MARKER_SPAN_M,
            "maximum_nac_resolution_m_per_pixel": MAX_RESOLUTION_M,
            "incidence_angle_deg": list(INCIDENCE_RANGE_DEG),
            "line_samples": 5064,
        },
        "downslope_alignment": {
            "D": effect,
            "ci_low": effect_lo,
            "ci_high": effect_hi,
            "median_delta_deg": float(np.median(delta[core])),
            "fraction_delta_le_30": float(np.mean(delta[core] <= 30.0)),
            "bootstrap_cluster": "nac_id",
            "bootstrap_replicates": args.bootstrap,
            "null_model": "independent random rotation of each NAC-image cluster",
            "null_replicates": args.null_replicates,
        },
        "panel_c_estimand": "within-NAC conditional event odds ratio, pole-facing/equator-facing",
        "panel_c_controls": [
            "NAC image stratum",
            "slope bin",
            "elevation tercile",
            "curvature tercile",
            "nominal footprint opportunity sampled at 50 m",
        ],
        "panel_c_limitation": (
            "Nominal NAC footprint opportunity is not effective searchable area; "
            "the estimate is observation-conditioned and not a lunar activity rate."
        ),
        "panel_a_window_y_m": [window_low, window_high],
        "panel_a_arrow_count": int(len(arrow_idx)),
    }
    (OUT / "figure5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
