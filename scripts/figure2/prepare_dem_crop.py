# Prepare the Hercules G DEM window used by Figure 2 panel b.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR
import argparse
import math
import numpy as np
import rasterio
from rasterio.windows import from_bounds

parser = argparse.ArgumentParser(description="Extract the original Figure 2 regional DEM window.")
parser.add_argument("--dem", default="/vsicurl/https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/LolaKaguya_Topo/Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif", help="Local GeoTIFF or GDAL-readable URL; default is the original upstream source.")
parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "prepared" / "figure2" / "cache")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
R = 1737400.0
DEG2M = math.pi * R / 180.0

# window centred on the vector ring
LA_C, LO_C = 46.45, 39.30
# panel b aspect 118:60 ~ 1.967; window height 26 km -> width 51 km
H_KM, W_KM = 26.0, 51.0
la0 = LA_C - H_KM / 2 / (DEG2M / 1000)
la1 = LA_C + H_KM / 2 / (DEG2M / 1000)
coslat = math.cos(math.radians(LA_C))
lo0 = LO_C - W_KM / 2 / (DEG2M / 1000 * coslat)
lo1 = LO_C + W_KM / 2 / (DEG2M / 1000 * coslat)
print(f"window: lon {lo0:.3f}..{lo1:.3f}, lat {la0:.3f}..{la1:.3f}")

with rasterio.open(args.dem) as src:
    win = from_bounds(lo0 * DEG2M, la0 * DEG2M, lo1 * DEG2M, la1 * DEG2M, src.transform)
    dem = src.read(1, window=win).astype(float)
print("dem:", dem.shape, dem.min(), dem.max())
np.save(args.output_dir / "panelb_dem.npy", dem)

