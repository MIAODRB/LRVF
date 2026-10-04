#!/usr/bin/env python3
"""Recreate the frozen six-degree Figure 2 zonal strip.

Intervals retain the submitted event-level normal approximation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR
import argparse
import numpy as np
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--event-cache", type=Path, default=DATA_DIR / "figure3" / "figure3_event_cache.npz")
parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "prepared" / "figure2")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
ev = np.load(args.event_cache)
lat_ev = ev['latitude'].astype(np.float64)
pole_ev = ev['poleward_component'].astype(np.float64)
z_edges = np.arange(-90, 90.1, 6.0)
z_lat = 0.5 * (z_edges[:-1] + z_edges[1:])
z_mu = np.full(z_lat.size, np.nan)
z_lo = np.full(z_lat.size, np.nan)
z_hi = np.full(z_lat.size, np.nan)
for i, (a, b) in enumerate(zip(z_edges[:-1], z_edges[1:])):
    v = pole_ev[(lat_ev >= a) & (lat_ev < b)]
    if v.size < 200:
        continue
    mu = v.mean(); ci = 1.96 * v.std(ddof=1) / np.sqrt(v.size)
    z_mu[i], z_lo[i], z_hi[i] = mu, mu - ci, mu + ci

# source data for the new strip
import csv
with open(args.output_dir / 'figure2_zonal_strip_source_data.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['# Figure 2a zonal-mean strip: full catalogue, 6-deg bins'])
    w.writerow(['lat_center', 'mean_poleward', 'ci_low', 'ci_high'])
    for c_, m_, l_, h_ in zip(z_lat, z_mu, z_lo, z_hi):
        w.writerow([c_] + ['' if np.isnan(x) else round(float(x), 4) for x in (m_, l_, h_)])
print("done")
