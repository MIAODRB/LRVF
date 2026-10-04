#!/usr/bin/env python
"""Prepare a compact NAC metadata/event-quality cache for Figure 4."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR

import numpy as np


ROOT = DATA_DIR
FIG4 = Path(__file__).resolve().parent
OUT = OUTPUT_DIR / "figure4"
EVENT_CACHE = DATA_DIR / "figure3" / "figure3_event_cache.npz"
QUALITY_CACHE = DATA_DIR / "figure6" / "fig4_data.npz"
FROZEN_METADATA_CACHE = DATA_DIR / "figure4" / "figure4_outputs" / "nac_metadata_cache.npz"
DBF_PATH = ROOT / "GLOBAL_rockfall_directions_dedup.dbf"
TAB_PATH = DATA_DIR / "CUMINDEX.TAB"
RECORD_BYTES = 902


def clean_float(value: str, missing: tuple[float, ...] = ()) -> float:
    try:
        result = float(value.strip())
    except ValueError:
        return float("nan")
    if any(np.isclose(result, sentinel) for sentinel in missing):
        return float("nan")
    return result


def prepare_metadata_cache(target_ids: np.ndarray, rebuild: bool = False) -> dict[str, np.ndarray]:
    cache_path = OUT / "nac_metadata_cache.npz"
    read_path = cache_path if cache_path.exists() else FROZEN_METADATA_CACHE
    if read_path.exists() and not rebuild:
        with np.load(read_path) as cached:
            return {key: cached[key] for key in cached.files}

    n_target = len(target_ids)
    target_clean = np.char.strip(target_ids.astype("S13"))
    sort_order = np.argsort(target_clean)
    target_sorted = target_clean[sort_order]

    matched = np.zeros(n_target, dtype=bool)
    incidence = np.full(n_target, np.nan, dtype=np.float32)
    resolution = np.full(n_target, np.nan, dtype=np.float32)
    scaled_width = np.full(n_target, np.nan, dtype=np.float32)
    scaled_height = np.full(n_target, np.nan, dtype=np.float32)
    frame = np.full(n_target, b"", dtype="S5")
    line_samples = np.full(n_target, -1, dtype=np.int32)
    image_lines = np.full(n_target, -1, dtype=np.int32)
    instrument_mode = np.full(n_target, -1, dtype=np.int16)
    data_quality = np.full(n_target, -1, dtype=np.int16)
    emission = np.full(n_target, np.nan, dtype=np.float32)
    phase = np.full(n_target, np.nan, dtype=np.float32)

    n_records = TAB_PATH.stat().st_size // RECORD_BYTES
    if n_records * RECORD_BYTES != TAB_PATH.stat().st_size:
        raise ValueError("CUMINDEX.TAB is not an integer number of 902-byte records")

    block_records = 50_000
    duplicate_hits = 0
    parsed_hits = 0
    with TAB_PATH.open("rb") as handle:
        for block_start in range(0, n_records, block_records):
            count = min(block_records, n_records - block_start)
            raw = handle.read(count * RECORD_BYTES)
            records = np.frombuffer(raw, dtype=np.uint8).reshape(count, RECORD_BYTES)
            keys = np.ascontiguousarray(records[:, 122:135]).view("S13").ravel()
            keys = np.char.strip(keys)
            positions = np.searchsorted(target_sorted, keys)
            inside = positions < n_target
            candidate_rows = np.flatnonzero(inside)
            if len(candidate_rows):
                candidate_positions = positions[candidate_rows]
                exact = target_sorted[candidate_positions] == keys[candidate_rows]
                candidate_rows = candidate_rows[exact]
                candidate_positions = candidate_positions[exact]
            for local_row, sorted_position in zip(candidate_rows, candidate_positions):
                code = int(sort_order[int(sorted_position)])
                if matched[code]:
                    duplicate_hits += 1
                    continue
                start = int(local_row) * RECORD_BYTES
                line = raw[start : start + RECORD_BYTES].decode("ascii")
                row = next(csv.reader([line]))
                if len(row) != 83:
                    raise ValueError(f"Expected 83 columns, found {len(row)}")
                product_id = row[5].strip()
                if product_id.encode("ascii") != target_clean[code]:
                    raise ValueError(f"Product ID mismatch: {product_id}")
                matched[code] = True
                parsed_hits += 1
                data_quality[code] = int(row[12].strip())
                frame[code] = row[26].strip().encode("ascii")
                instrument_mode[code] = int(row[30].strip())
                image_lines[code] = int(row[52].strip())
                line_samples[code] = int(row[53].strip())
                scaled_width[code] = clean_float(row[55], (0.0,))
                scaled_height[code] = clean_float(row[56], (0.0,))
                resolution[code] = clean_float(row[57], (0.0,))
                emission[code] = clean_float(row[58], (99.99,))
                incidence[code] = clean_float(row[59], (999.99,))
                phase[code] = clean_float(row[60], (999.99,))

    arrays = {
        "product_id": target_clean,
        "matched": matched,
        "incidence": incidence,
        "resolution": resolution,
        "scaled_width": scaled_width,
        "scaled_height": scaled_height,
        "frame": frame,
        "line_samples": line_samples,
        "image_lines": image_lines,
        "instrument_mode": instrument_mode,
        "data_quality": data_quality,
        "emission": emission,
        "phase": phase,
        "source_record_count": np.array([n_records], dtype=np.int64),
        "duplicate_hits_after_first_match": np.array([duplicate_hits], dtype=np.int64),
        "parsed_hits": np.array([parsed_hits], dtype=np.int64),
    }
    np.savez_compressed(cache_path, **arrays)
    return arrays


def load_event_quality() -> dict[str, np.ndarray]:
    with np.load(EVENT_CACHE) as cached:
        event = {key: cached[key] for key in cached.files}

    # The frozen quality array is byte-equivalent to DBF scores cast to float32.
    # It allows the compact reproduction bundle to be used without the raw DBF.
    if QUALITY_CACHE.exists():
        with np.load(QUALITY_CACHE) as cached:
            score = cached["score"].astype(np.float32)
        if score.shape != event["latitude"].shape:
            raise ValueError("Quality and event cache lengths do not match")
    else:
        # Optional raw-data route: deletion flag + nac_id (80) + score (24) + lat_zone (80).
        dbf_dtype = np.dtype([("deleted", "S1"), ("nac_id", "S80"), ("score", "S24"), ("lat_zone", "S80")])
        dbf = np.memmap(DBF_PATH, mode="r", dtype=dbf_dtype, offset=129, shape=(len(event["latitude"]),))
        score = np.array([float(value) for value in dbf["score"]], dtype=np.float32)

    event["score"] = score
    return event


def distribution_summary(event: dict[str, np.ndarray], meta: dict[str, np.ndarray]) -> dict:
    code = event["nac_code"]
    event_matched = meta["matched"][code]
    summary = {
        "events": int(len(code)),
        "unique_nac_ids": int(len(meta["matched"])),
        "matched_unique_nac_ids": int(meta["matched"].sum()),
        "matched_events": int(event_matched.sum()),
        "tab_records_scanned": int(meta["source_record_count"][0]),
        "duplicate_tab_hits_after_first_match": int(meta["duplicate_hits_after_first_match"][0]),
    }
    for name, values in (
        ("incidence_deg", meta["incidence"][code]),
        ("resolution_m_per_px", meta["resolution"][code]),
        ("score", event["score"]),
    ):
        finite = values[np.isfinite(values) & event_matched]
        summary[name] = {
            "n": int(len(finite)),
            "q05": float(np.percentile(finite, 5)),
            "q25": float(np.percentile(finite, 25)),
            "median": float(np.percentile(finite, 50)),
            "q75": float(np.percentile(finite, 75)),
            "q95": float(np.percentile(finite, 95)),
        }

    frame_values, frame_counts = np.unique(meta["frame"][code][event_matched], return_counts=True)
    summary["event_counts_by_frame"] = {
        value.decode("ascii"): int(count) for value, count in zip(frame_values, frame_counts)
    }
    sample_values, sample_counts = np.unique(meta["line_samples"][code][event_matched], return_counts=True)
    order = np.argsort(sample_counts)[::-1][:12]
    summary["top_line_sample_counts"] = {
        str(int(sample_values[idx])): int(sample_counts[idx]) for idx in order
    }
    quality_values, quality_counts = np.unique(meta["data_quality"][code][event_matched], return_counts=True)
    summary["event_counts_by_data_quality"] = {
        str(int(value)): int(count) for value, count in zip(quality_values, quality_counts)
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    event = load_event_quality()
    meta = prepare_metadata_cache(event["nac_ids"])
    summary = distribution_summary(event, meta)
    (OUT / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
