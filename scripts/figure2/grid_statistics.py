# Grid statistics for Panel a: circular mean direction, resultant length (R-bar),
# mean poleward component, and event counts on a lat/lon grid.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Generate the legacy 6-degree Figure 2 field and density grid.")
parser.add_argument("--events", type=Path, default=DATA_DIR / "figure2" / "cache" / "events.npz")
parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "prepared" / "figure2" / "cache")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
d = np.load(args.events)
lon, lat, az = d["lon0"], d["lat0"], d["az"].astype(np.float64)
# Legacy representative position: arithmetic midpoint of the two stored points.
# Preserved for frozen-grid reproduction; see docs/figure2_3.md for seam caveat.
lonm = (d["lon0"] + d["lon1"]) / 2.0
latm = (d["lat0"] + d["lat1"]) / 2.0

az_r = np.radians(az)
u = np.sin(az_r)  # east component of unit direction
v = np.cos(az_r)  # north component

for CELL in (6.0,):
    nx, ny = int(360 / CELL), int(180 / CELL)
    ix = np.clip(((lonm + 180) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((latm + 90) / CELL).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    cnt = np.bincount(flat, minlength=nx * ny)
    su = np.bincount(flat, weights=u, minlength=nx * ny)
    sv = np.bincount(flat, weights=v, minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        Rbar = np.sqrt(su**2 + sv**2) / np.maximum(cnt, 1)
        mean_az = (np.degrees(np.arctan2(su, sv))) % 360.0
    # poleward component per event: cos(angle to local poleward bearing)
    # northern hemi: poleward bearing = 0 -> p = cos(az); southern: bearing=180 -> p=-cos(az)
    p = np.where(latm >= 0, v, -v)
    sp = np.bincount(flat, weights=p, minlength=nx * ny)
    pmean = sp / np.maximum(cnt, 1)
    np.savez(args.output_dir / f"grid_{int(CELL)}deg.npz",
             cnt=cnt.reshape(ny, nx), Rbar=Rbar.reshape(ny, nx),
             mean_az=mean_az.reshape(ny, nx), pmean=pmean.reshape(ny, nx),
             cell=CELL)
    ok = cnt.reshape(ny, nx) >= 30
    print(f"cell={CELL}: cells with N>=30: {ok.sum()}/{nx*ny}",
          f"median N of valid: {np.median(cnt[cnt>=30]):.0f}",
          f"Rbar range: {np.nanmin(Rbar[cnt>=30]):.3f}-{np.nanmax(Rbar[cnt>=30]):.3f}",
          f"pmean range: {np.nanmin(pmean[cnt>=30]):.3f}..{np.nanmax(pmean[cnt>=30]):.3f}")

# density layer: 0.25 deg counts
CELL = 0.25
nx, ny = int(360 / CELL), int(180 / CELL)
ix = np.clip(((lonm + 180) / CELL).astype(int), 0, nx - 1)
iy = np.clip(((latm + 90) / CELL).astype(int), 0, ny - 1)
cnt = np.bincount(iy * nx + ix, minlength=nx * ny).reshape(ny, nx)
np.save(args.output_dir / "density_0p25.npy", cnt.astype(np.float32))
print("density grid saved; nonzero cells:", (cnt > 0).sum(), "max:", cnt.max())

# global stats for annotation
print("global Rbar:", np.sqrt(u.sum()**2 + v.sum()**2) / len(u))
print("hemisphere counts N/S:", (latm >= 0).sum(), (latm < 0).sum())
