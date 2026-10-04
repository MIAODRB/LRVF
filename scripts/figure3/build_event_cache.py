#!/usr/bin/env python3
"""Build the float32 event cache used by the submitted Figure 3.

Adapted from the original strict SHP/DBF binary reader. No legacy Figure 3
visualization, heatmap or bootstrap algorithm is included here.
"""

from __future__ import annotations
import argparse
import struct
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR

def read_dbf_layout(dbf_path: Path) -> tuple[int, int, int, list[tuple[str, str, int, int]]]:
    with dbf_path.open("rb") as handle:
        header = handle.read(32)
        if len(header) != 32:
            raise ValueError(f"Invalid DBF header: {dbf_path}")
        n_records, header_len, record_len = struct.unpack("<xxxxIHH20x", header)
        fields: list[tuple[str, str, int, int]] = []
        while handle.tell() < header_len - 1:
            descriptor = handle.read(32)
            name = descriptor[:11].split(b"\0", 1)[0].decode("ascii", "replace")
            field_type = chr(descriptor[11])
            fields.append((name, field_type, descriptor[16], descriptor[17]))
    return n_records, header_len, record_len, fields


def load_event_cache(shp_path: Path, output_dir: Path, rebuild: bool = False) -> dict[str, np.ndarray]:
    cache_path = output_dir / "figure3_event_cache.npz"
    if cache_path.exists() and not rebuild:
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}

    dbf_path = shp_path.with_suffix(".dbf")
    if not dbf_path.exists():
        raise FileNotFoundError(f"Missing DBF attribute table: {dbf_path}")

    n_dbf, dbf_header_len, dbf_record_len, fields = read_dbf_layout(dbf_path)
    expected_fields = [("nac_id", "C", 80, 0), ("score", "N", 24, 15), ("lat_zone", "C", 80, 0)]
    if fields != expected_fields:
        raise ValueError(f"Unexpected DBF schema: {fields}")

    # Each geometry is a fixed two-point PolyLine: 8-byte record header plus
    # 80-byte content.  The strict validation prevents silent misreads if the
    # source file is later replaced by a general multipart shapefile.
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
    if shp_dtype.itemsize != 88:
        raise AssertionError("Unexpected SHP record dtype size")
    n_shp = (shp_path.stat().st_size - 100) // shp_dtype.itemsize
    if 100 + n_shp * shp_dtype.itemsize != shp_path.stat().st_size:
        raise ValueError("Input SHP is not a fixed-size two-point PolyLine file")
    if n_shp != n_dbf:
        raise ValueError(f"SHP/DBF record mismatch: {n_shp} vs {n_dbf}")

    shp = np.memmap(shp_path, mode="r", dtype=shp_dtype, offset=100, shape=(n_shp,))
    if not (
        np.all(shp["shape_type"] == 3)
        and np.all(shp["n_parts"] == 1)
        and np.all(shp["n_points"] == 2)
        and np.all(shp["content_length"] == 40)
    ):
        raise ValueError("All SHP records must be one-part, two-point PolyLines")

    points = np.asarray(shp["points"])
    lon1 = points[:, 0]
    lat1 = points[:, 1]
    lon2 = points[:, 2]
    lat2 = points[:, 3]

    # Initial great-circle bearing, clockwise from geographic north.  Wrapping
    # delta-longitude handles features that straddle the 0/360-degree seam.
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    delta_lon = np.deg2rad(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    x = np.sin(delta_lon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lon)
    bearing = (np.rad2deg(np.arctan2(x, y)) + 360.0) % 360.0
    latitude = 0.5 * (lat1 + lat2)

    # Local poleward reference is 0 degrees in the north and 180 degrees in the
    # south.  Alpha is signed and cyclic on [-180, 180).
    poleward_bearing = np.where(latitude >= 0.0, 0.0, 180.0)
    alpha = ((bearing - poleward_bearing + 180.0) % 360.0) - 180.0
    poleward_component = np.cos(np.deg2rad(alpha))

    dbf_dtype = np.dtype(
        [
            ("deleted", "S1"),
            ("nac_id", "S80"),
            ("score", "S24"),
            ("lat_zone", "S80"),
        ]
    )
    if dbf_dtype.itemsize != dbf_record_len:
        raise ValueError("DBF record layout does not match its header")
    dbf = np.memmap(
        dbf_path,
        mode="r",
        dtype=dbf_dtype,
        offset=dbf_header_len,
        shape=(n_dbf,),
    )
    if np.any(dbf["deleted"] == b"*"):
        raise ValueError("Deleted DBF records are present; clean the source before plotting")
    nac_ids, nac_code = np.unique(dbf["nac_id"], return_inverse=True)

    valid = (
        np.isfinite(latitude)
        & np.isfinite(bearing)
        & np.isfinite(alpha)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
        & ((np.abs(lat2 - lat1) + np.abs(((lon2 - lon1 + 180.0) % 360.0) - 180.0)) > 0.0)
    )
    if not np.all(valid):
        latitude = latitude[valid]
        bearing = bearing[valid]
        alpha = alpha[valid]
        poleward_component = poleward_component[valid]
        nac_code = nac_code[valid]

    arrays = {
        "event_id": np.arange(1, len(latitude) + 1, dtype=np.int32),
        "latitude": latitude.astype(np.float32),
        "bearing": bearing.astype(np.float32),
        "alpha": alpha.astype(np.float32),
        "poleward_component": poleward_component.astype(np.float32),
        "nac_code": nac_code.astype(np.int32),
        "nac_ids": nac_ids,
        "source_record_count": np.array([n_shp], dtype=np.int64),
    }
    np.savez_compressed(cache_path, **arrays)
    return arrays



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "GLOBAL_rockfall_directions_dedup.shp")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "prepared" / "figure3")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = load_event_cache(args.input, args.output_dir, rebuild=True)
    print(f"Saved {len(events['latitude']):,} records and {len(events['nac_ids']):,} NAC products.")
