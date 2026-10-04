# Prepare the selected NAC browse crop used by Figure 2 panel c.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window

parser = argparse.ArgumentParser(description="Extract the original selected Figure 2 NAC browse crop.")
parser.add_argument("--nac", default="/vsicurl/https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1005/EXTRAS/BROWSE/2010291/M142007158LC_pyr.tif", help="Local browse GeoTIFF or GDAL-readable URL; default is the original upstream source.")
parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "prepared" / "figure2" / "cache")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
URL = args.nac
# footprint corners (from LROC product page)
UL = (47.15, 39.23); UR = (47.15, 39.13); LR = (46.19, 39.15); LL = (46.19, 39.24)
W, H = 5064, 52224

def ll_to_px(lat, lon):
    # bilinear inverse (footprint is near-affine for NAC)
    v = (UL[0] - lat) / (UL[0] - LL[0])          # 0 top -> 1 bottom
    lon_left = UL[1] + v * (LL[1] - UL[1])
    lon_right = UR[1] + v * (LR[1] - UR[1])
    u = (lon_left - lon) / (lon_left - lon_right)  # 0 left -> 1 right
    return u * W, v * H

# The first original crop was selected for the submitted display.
targets = [(46.47, 39.205)]
with rasterio.open(URL) as src:
    for k, (la, lo) in enumerate(targets):
        x, y = ll_to_px(la, lo)
        x0, y0 = int(x - 512), int(y - 512)
        x0 = max(0, min(W - 1024, x0)); y0 = max(0, min(H - 1024, y0))
        arr = src.read(1, window=Window(x0, y0, 1024, 1024))
        np.save(args.output_dir / f"nac_crop_{k}.npy", arr)
        np.save(args.output_dir / f"nac_crop_{k}_origin.npy",
                np.array([x0, y0]))
        print(k, la, lo, "px", int(x), int(y), "crop origin", x0, y0,
              "range", arr.min(), arr.max())
