# Extract stored endpoint coordinates and encoded trajectory bearing.
# Stored endpoint separation has no physical track-length or displacement meaning.
# Bearing is clockwise from local north, from the first stored point toward the
# polarity-defining second point. The legacy cache key length_m is retained.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR
import argparse
import numpy as np
import pyogrio
import shapely

parser = argparse.ArgumentParser(description="Extract the Figure 2 event cache from the two-point direction catalogue.")
parser.add_argument("--input", type=Path, default=DATA_DIR / "GLOBAL_rockfall_directions_dedup.shp")
parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "prepared" / "figure2" / "cache")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
SHP = args.input
OUT = args.output_dir / "events.npz"

print("reading shapefile ...")
gdf = pyogrio.read_dataframe(SHP, columns=["nac_id", "score", "lat_zone"])
print("features:", len(gdf))

# vectorized coordinate extraction: lines may have >2 vertices; take first & last
coords, index = shapely.get_coordinates(gdf.geometry.values, return_index=True)
# first and last vertex per feature
first_idx = np.searchsorted(index, np.arange(len(gdf)), side="left")
last_idx = np.searchsorted(index, np.arange(len(gdf)), side="right") - 1
x0, y0 = coords[first_idx, 0], coords[first_idx, 1]
x1, y1 = coords[last_idx, 0], coords[last_idx, 1]

# normalize longitudes to [-180, 180)
def norm_lon(x):
    return (x + 180.0) % 360.0 - 180.0

lon0, lat0 = norm_lon(x0), y0
lon1, lat1 = norm_lon(x1), y1

# forward azimuth on sphere from (lat0,lon0) to (lat1,lon1)
la0, la1 = np.radians(lat0), np.radians(lat1)
dlon = np.radians(x1 - x0)  # use raw dlon (small segments, avoids wrap issues at ±180)
az = np.degrees(np.arctan2(np.sin(dlon) * np.cos(la1),
                           np.cos(la0) * np.sin(la1) - np.sin(la0) * np.cos(la1) * np.cos(dlon)))
az = az % 360.0

# Stored endpoint separation in metres (quality-control/provenance only;
# it is not a measured trajectory length). Moon radius 1737.4 km.
R = 1_737_400.0
dlat = la1 - la0
h = np.sin(dlat / 2) ** 2 + np.cos(la0) * np.cos(la1) * np.sin(dlon / 2) ** 2
length_m = 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

np.savez_compressed(
    OUT,
    lon0=lon0.astype(np.float64), lat0=lat0.astype(np.float64),
    lon1=lon1.astype(np.float64), lat1=lat1.astype(np.float64),
    az=az.astype(np.float32), length_m=length_m.astype(np.float32),
    score=gdf["score"].to_numpy(np.float32),
)
# nac ids saved separately (object strings)
np.save(args.output_dir / "nac_ids.npy", gdf["nac_id"].to_numpy())
print("saved.", "az stats:", np.nanmin(az), np.nanmax(az),
      "stored endpoint-separation median m:", np.median(length_m))
