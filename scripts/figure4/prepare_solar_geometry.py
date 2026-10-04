#!/usr/bin/env python3
"""Rebuild per-NAC ground-frame solar bearings from the frozen CUMINDEX extract.

The Figure 4 production wrapper uses the frozen copy under DATA_DIR.
This optional helper writes a rebuilt cache under OUTPUT_DIR for comparison.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR

with np.load(DATA_DIR / "figure3" / "figure3_event_cache.npz") as cached:
    nac_ids = np.char.strip(np.char.decode(cached["nac_ids"], "ascii"))

sol = pd.read_csv(DATA_DIR / "cumindex_solar_extract.csv.gz",
                  header=None, names=['pid', 'north_az', 'subsolar_az', 'ss_lat',
                                      'ss_lon', 'c_lat', 'c_lon'],
                  dtype={'pid': str}, na_values=['999.99'], engine='c')
sol['pid'] = sol['pid'].str.strip()
sol = sol.drop_duplicates('pid', keep='first').set_index('pid')
idx = sol.index.get_indexer(nac_ids)
found = idx >= 0
print('solar metadata matched images: %d / %d' % (found.sum(), len(nac_ids)))
north_az = np.full(len(nac_ids), np.nan)
subsolar_az = np.full(len(nac_ids), np.nan)
ss_lat = np.full(len(nac_ids), np.nan)
ss_lon = np.full(len(nac_ids), np.nan)
c_lat = np.full(len(nac_ids), np.nan)
c_lon = np.full(len(nac_ids), np.nan)
for name, arr in (('north_az', north_az), ('subsolar_az', subsolar_az),
                  ('ss_lat', ss_lat), ('ss_lon', ss_lon),
                  ('c_lat', c_lat), ('c_lon', c_lon)):
    arr[found] = sol[name].to_numpy()[idx[found]]

# ground-frame solar azimuth (bearing from image centre toward sub-solar point,
# clockwise from north): two independent estimates.
sol_az_img = (subsolar_az - north_az) % 360.0
la1, lo1 = np.radians(c_lat), np.radians(c_lon)
la2, lo2 = np.radians(ss_lat), np.radians(ss_lon)
dlon = lo2 - lo1
brg = np.degrees(np.arctan2(np.sin(dlon) * np.cos(la2),
                            np.cos(la1) * np.sin(la2) -
                            np.sin(la1) * np.cos(la2) * np.cos(dlon))) % 360.0
dd = np.abs((sol_az_img - brg + 180.0) % 360.0 - 180.0)
ok = np.isfinite(dd)
print('solar azimuth cross-check |img-frame - great-circle| deg: median %.2f  p90 %.2f'
      % (np.nanmedian(dd[ok]), np.nanpercentile(dd[ok], 90)))
solar_az = np.where(np.isfinite(brg), brg, sol_az_img)   # great-circle estimate preferred


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUTPUT_DIR / "solar_per_image.npz",
                    nac_ids=nac_ids.astype("S13"), solar_az=solar_az,
                    north_az=north_az, subsolar_az=subsolar_az)
print("Saved", OUTPUT_DIR / "solar_per_image.npz")
