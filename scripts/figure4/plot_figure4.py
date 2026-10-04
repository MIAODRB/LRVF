#!/usr/bin/env python3
"""Reproduce the production Figure 4, including its solar-azimuth stratum."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR
import plot_figure4_base as base

base.FIG_HEIGHT_MM = 150.0

ORIG_BUILD = base.build_masks

def build_masks_v2(event, meta):
    strata, core = ORIG_BUILD(event, meta)
    code = event['nac_code']
    incidence = meta['incidence'][code]
    resolution = meta['resolution'][code]
    matched = meta['matched'][code]
    all_mask = matched & np.isfinite(incidence) & np.isfinite(resolution)
    saz = SOLAR_AZ[code]
    east = (saz % 360.0) < 180.0
    strata['Spatial summing'] = [
        ('Non-summed (5064)', strata['Spatial summing'][0][1]),
        ('2× summed (2532)', strata['Spatial summing'][1][1]),
    ]
    strata['Sub-solar azimuth'] = [
        ('East (0–180°)', all_mask & np.isfinite(saz) & east),
        ('West (180–360°)', all_mask & np.isfinite(saz) & ~east),
    ]
    strata.pop('Track length', None)
    return strata, core

base.build_masks = build_masks_v2

# --------------------------------------------------- label fix inside draw
ORIG_DRAW = base.draw_figure

def draw_figure_v2(forest, all_curve, core_curve, high_rows):
    forest = [dict(row) for row in forest]
    forest[0]['level'] = 'All data (30–60° band)'
    fig, panels = ORIG_DRAW(forest, all_curve, core_curve, high_rows)
    for ax in panels['c']:
        for t in ax.texts:
            if t.get_text().startswith('NAC'):
                t.set_x(min(t.get_position()[0], 0.68))
                t.set_fontsize(5.0)
    return fig, panels

base.draw_figure = draw_figure_v2


def main():
    # Parse first so --help remains available without downloading the data bundle.
    base.parse_args()
    global SOLAR_AZ, SOLAR_IDS
    with np.load(DATA_DIR / "solar_per_image.npz") as solar:
        SOLAR_AZ = solar["solar_az"]
        SOLAR_IDS = solar["nac_ids"]
    with np.load(DATA_DIR / "figure3" / "figure3_event_cache.npz") as events:
        if not np.array_equal(np.char.strip(events["nac_ids"].astype("S13")), SOLAR_IDS):
            raise ValueError("Solar and event cache NAC IDs are not aligned")
    base.main()


if __name__ == "__main__":
    main()
