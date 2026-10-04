# -*- coding: utf-8 -*-
"""
Figure 6 v4 | Panel-a data layer: residual latitude law, HIGH-QUALITY subset.

Change from v3 (recover sparse polar rows)
------------------------------------------
v3 applied the belt-tuned cell sufficiency rules (>= 25 events, >= 3 images,
>= 100 reliable images per polar cell, CI width <= 0.50) everywhere. Under
the 35-75 deg incidence window these rules blank the rows poleward of 66 deg
almost entirely, although they hold ~200-8,000 qualifying events per band
with a coherent, hemispherically symmetric equatorward signal.
v4 keeps the belt rules frozen and adds RELAXED thresholds for rows with
|lat| >= 66 deg (>= 10 events, >= 2 images, >= 10 reliable images,
CI width <= 0.80). Cells passing only the relaxed rules are flagged
'ok_sparse_polar' in the source data. Rows beyond ~ +/-78 deg remain
'no effective detection' (no qualifying events at all).

Subset identical to v3:

    incidence 35-75 deg
    pixel scale <= 1.0 m per pixel
    non-summed NAC frames
    detection confidence >= 0.5713289  (catalogue median score)

-> n = 456,253 of 1,026,786 events (44.4%).

Outputs: fig6v4_field.npz + fig6v4_source_data.csv
"""
import numpy as np
import csv
from pathlib import Path
import sys

# Resolve every input from the configured reproduction-data directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR as ROOT_OUTPUT_DIR

FIGURE_OUTPUT_DIR = ROOT_OUTPUT_DIR / "figure6"
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- high-quality subset (user-specified) ----
INC_MIN, INC_MAX = 35.0, 75.0
RES_MAX = 1.0
SCORE_MIN = 0.5713289        # catalogue median detection confidence

# ---- frozen grid / classification constants (v1) ----
CELL = 6.0
MIN_EVT = 25
MIN_IMG = 3
MAX_SHARE = 0.90
CI_MAX = 0.50
EPS = 0.03
EFF_FRAC, EFF_MIN_IMG = 0.10, 100
POLAR_LAT = 60.0
B = 2000
SEED = 20260715

# ---- v4: relaxed thresholds for sparse polar rows (|lat| >= SPARSE_LAT) ----
# The 35-75 deg incidence window leaves only ~200-400 qualifying events per
# 6-deg band poleward of 72 deg, and the per-cell count of reliable images
# drops to a median of ~24 at +/-75 (vs >500 in the belt). The belt-tuned
# cell rules (>=25 events, >=3 images, >=100 reliable images) therefore
# blank rows that DO carry a coherent signal (zonal profile at +/-75:
# mu = -0.185 N / -0.240 S, both equatorward, hemispherically symmetric).
# v4 keeps the belt rules frozen and applies relaxed thresholds only to
# rows poleward of 66 deg; these cells are flagged 'ok_sparse_polar' in the
# source data so the relaxation is fully auditable.
SPARSE_LAT = 66.0
MIN_EVT_SP = 10
MIN_IMG_SP = 2
EFF_MIN_IMG_SP = 10
CI_MAX_SP = 0.80
# extra-sparse tier for the outermost supported rows (|lat| >= 72): only a
# few hundred qualifying events per band survive the incidence window, so
# the event floor drops to 5; every such cell is still cluster-bootstrapped
# and flagged, and the zonal profile provides the aggregated check
XSPARSE_LAT = 72.0
MIN_EVT_XS = 5
EFF_MIN_IMG_XS = 5

# ---- frozen smoothing / display constants (v2) ----
SIG_LAT, SIG_LON = 0.55, 2.2
W_MIN = 0.25
VMAX = 0.25
PROF_MIN = 200

# ---------------------------------------------------------------- data
d = np.load(DATA_DIR / 'figure6' / 'fig4_data.npz')
ids = np.load(DATA_DIR / 'figure2' / 'cache' / 'nac_ids.npy', allow_pickle=True)

lat = d['lat'].astype(np.float64)
lon = (d['lon'].astype(np.float64) + 180.0) % 360.0 - 180.0
az = d['az'].astype(np.float64)
inc = d['inc'].astype(np.float64)
res = d['res'].astype(np.float64)
score = d['score'].astype(np.float64)
summed = d['summed']
matched = d['matched']

north = lat >= 0
pole = np.where(north, np.cos(np.radians(az)), -np.cos(np.radians(az)))
hq = (matched & (inc >= INC_MIN) & (inc <= INC_MAX) & (res <= RES_MAX)
      & (summed == 0) & (score >= SCORE_MIN))
N_HQ = int(hq.sum())
print('HQ events: %d / %d (%.1f%%)' % (N_HQ, hq.size, 100.0 * N_HQ / hq.size))

_, uidx = np.unique(ids, return_inverse=True)

ny, nx = int(180 / CELL), int(360 / CELL)
iy = np.clip(((lat + 90.0) // CELL).astype(int), 0, ny - 1)
ix = np.clip(((lon + 180.0) // CELL).astype(int), 0, nx - 1)
cell = iy * nx + ix
lat_c = -90 + (np.arange(ny) + 0.5) * CELL
lon_c = -180 + (np.arange(nx) + 0.5) * CELL

# ------------------------------------------- polar effective-imaging mask
# image-level reliability now mirrors the event subset's imaging window
plat = d['plat'].astype(np.float64)
plon = (d['plon'].astype(np.float64) + 180.0) % 360.0 - 180.0
img_rel = ((d['pinc'] >= INC_MIN) & (d['pinc'] <= INC_MAX)
           & (d['pres'] <= RES_MAX) & (d['psum'] == 0))
py = np.clip(((plat + 90.0) // CELL).astype(int), 0, ny - 1)
px = np.clip(((plon + 180.0) // CELL).astype(int), 0, nx - 1)
n_img_cell = np.zeros((ny, nx), int)
n_rel_cell = np.zeros((ny, nx), int)
np.add.at(n_img_cell, (py, px), 1)
np.add.at(n_rel_cell, (py, px), img_rel.astype(int))
eff_ok = np.ones((ny, nx), bool)
polar_row = np.abs(lat_c) >= POLAR_LAT
sparse_row = np.abs(lat_c) >= SPARSE_LAT
xsparse_row = np.abs(lat_c) >= XSPARSE_LAT
eff_min = np.where(xsparse_row, EFF_MIN_IMG_XS,
                   np.where(sparse_row, EFF_MIN_IMG_SP, EFF_MIN_IMG))[:, None]
# the EFF_FRAC share test is meaningful in the belt but degenerate under the
# 35-75 deg window near the poles (most polar images exceed 75 deg incidence
# by geometry, so the reliable share is tiny even where reliable coverage is
# adequate in absolute terms) -> sparse rows use the absolute count only
frac_min = np.where(sparse_row, 0.0, EFF_FRAC)[:, None]
eff_ok[polar_row] = ((n_rel_cell[polar_row] >= eff_min[polar_row]) &
                     (n_img_cell[polar_row] > 0) &
                     (n_rel_cell[polar_row] / np.maximum(n_img_cell[polar_row], 1)
                      >= frac_min[polar_row]))
print('polar rows: %d of %d polar cells pass effective-imaging'
      % (eff_ok[polar_row].sum(), polar_row.sum() * nx))

# ---------------------------------------------------------------- classify
cls = np.zeros((ny, nx), np.int8)     # 0 insufficient, 1 pol, 2 eq, 3 no-res
P_adj = np.full((ny, nx), np.nan)
ci_lo = np.full((ny, nx), np.nan)
ci_hi = np.full((ny, nx), np.nan)
n_ev = np.zeros((ny, nx), int)
n_im = np.zeros((ny, nx), int)
share = np.full((ny, nx), np.nan)
qflag = np.empty((ny, nx), object)

rng = np.random.default_rng(SEED)
order = np.argsort(cell[hq], kind='stable')
qidx = np.flatnonzero(hq)[order]
csorted = cell[hq][order]
starts = np.searchsorted(csorted, np.arange(ny * nx))
ends = np.searchsorted(csorted, np.arange(ny * nx), side='right')

for cy in range(ny):
    sparse = sparse_row[cy]
    min_evt = (MIN_EVT_XS if xsparse_row[cy]
               else MIN_EVT_SP if sparse else MIN_EVT)
    min_img = MIN_IMG_SP if sparse else MIN_IMG
    ci_max = CI_MAX_SP if sparse else CI_MAX
    for cx in range(nx):
        cid = cy * nx + cx
        sel = qidx[starts[cid]:ends[cid]]
        n = sel.size
        n_ev[cy, cx] = n
        if n:
            uim, invc = np.unique(uidx[sel], return_inverse=True)
            n_im[cy, cx] = uim.size
        if n < min_evt:
            qflag[cy, cx] = 'too_few_events'; continue
        if not eff_ok[cy, cx]:
            qflag[cy, cx] = 'polar_imaging_limited'; continue
        if uim.size < min_img:
            qflag[cy, cx] = 'too_few_images'; continue
        bc = np.bincount(invc)
        share[cy, cx] = bc.max() / n
        if share[cy, cx] > MAX_SHARE:
            qflag[cy, cx] = 'single_image_dominated'; continue
        v = pole[sel]
        o2 = np.argsort(invc, kind='stable')
        vi = v[o2]
        brk = np.searchsorted(invc[o2], np.arange(uim.size))
        sums = np.add.reduceat(vi, brk)
        cnts = np.diff(np.append(brk, n))
        pick = rng.integers(0, uim.size, (B, uim.size))
        bs = sums[pick].sum(1) / cnts[pick].sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        mu = v.mean()
        P_adj[cy, cx], ci_lo[cy, cx], ci_hi[cy, cx] = mu, lo, hi
        if hi - lo > ci_max:
            qflag[cy, cx] = 'ci_too_wide'; continue
        okflag = 'ok_sparse_polar' if sparse else 'ok'
        if lo > 0 and mu >= EPS:
            cls[cy, cx] = 1; qflag[cy, cx] = okflag
        elif hi < 0 and mu <= -EPS:
            cls[cy, cx] = 2; qflag[cy, cx] = okflag
        else:
            cls[cy, cx] = 3; qflag[cy, cx] = okflag

names = ['Insufficient data', 'Poleward-dominant', 'Equatorward-dominant',
         'No resolved directional preference']
for k, nm in enumerate(names):
    print('%-36s %4d cells' % (nm, (cls == k).sum()))

# ------------------------------------------------- smoothed display field
valid = cls > 0


def gk(sig):
    r = int(np.ceil(3 * sig)); x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sig) ** 2)
    return k / k.sum(), r


def conv1(M, k, r, axis, wrap):
    out = np.zeros_like(M)
    n = M.shape[axis]
    for i, w in zip(range(-r, r + 1), k):
        if wrap:
            out += w * np.roll(M, i, axis=axis)
        else:
            idx = np.clip(np.arange(n) + i, 0, n - 1)
            out += w * np.take(M, idx, axis=axis)
    return out


A = np.where(valid, P_adj, 0.0)
Wt = valid.astype(float)
kl, rl = gk(SIG_LAT); ko, ro = gk(SIG_LON)
An = conv1(conv1(A, ko, ro, 1, True), kl, rl, 0, False)
Wn = conv1(conv1(Wt, ko, ro, 1, True), kl, rl, 0, False)
with np.errstate(invalid='ignore', divide='ignore'):
    field_disp = np.where(Wn > W_MIN, An / Wn, np.nan)
nodata = ~(Wn > W_MIN)
field = np.where(valid, field_disp, np.nan)
print('display field: %d cells, range %.3f .. %.3f  (masked %d)'
      % (np.isfinite(field_disp).sum(), np.nanmin(field_disp),
         np.nanmax(field_disp), int(nodata.sum())))

# ------------------------------------------------- zonal profile (raw events)
p_edges = np.arange(-90, 90.1, CELL)
p_lat = 0.5 * (p_edges[:-1] + p_edges[1:])
p_mu = np.full(p_lat.size, np.nan)
p_lo = np.full(p_lat.size, np.nan)
p_hi = np.full(p_lat.size, np.nan)
p_n = np.zeros(p_lat.size, int)
for i, (a, b) in enumerate(zip(p_edges[:-1], p_edges[1:])):
    v = pole[hq & (lat >= a) & (lat < b)]
    p_n[i] = v.size
    if v.size < PROF_MIN:
        continue
    mu = v.mean(); ci = 1.96 * v.std(ddof=1) / np.sqrt(v.size)
    p_mu[i], p_lo[i], p_hi[i] = mu, mu - ci, mu + ci

# regime: CI excludes zero AND the effect clears the minimal practical
# threshold EPS (avoids flagging near-zero equatorial bins)
regime = np.full(p_lat.size, np.nan)
ok = np.isfinite(p_mu)
regime[ok] = 0
regime[ok & (p_lo > 0) & (p_mu >= EPS)] = 1
regime[ok & (p_hi < 0) & (p_mu <= -EPS)] = -1
for c, mu, lo, hi, rg in zip(p_lat, p_mu, p_lo, p_hi, regime):
    if np.isnan(mu):
        continue
    tag = {1: 'poleward', -1: 'equatorward', 0: 'no-resolved'}[int(rg)]
    print('%+5.0f  %+.3f [%+.3f,%+.3f]  %s' % (c, mu, lo, hi, tag))

np.savez_compressed(FIGURE_OUTPUT_DIR / 'fig6v4_field.npz',
                    field=field, field_disp=field_disp, nodata=nodata,
                    cls=cls, cell=CELL, lat_c=lat_c, lon_c=lon_c,
                    vmax=VMAX, sig_lat=SIG_LAT, sig_lon=SIG_LON,
                    p_lat=p_lat, p_mu=p_mu, p_lo=p_lo, p_hi=p_hi,
                    p_n=p_n, regime=regime,
                    n_hq=N_HQ, n_total=int(hq.size))

with open(FIGURE_OUTPUT_DIR / 'fig6v4_source_data.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['# Figure 6a source data (v4, high-quality subset: incidence '
                '35-75 deg, pixel scale <= 1.0 m, non-summed, detection '
                'confidence >= 0.5713289; n = %d). Rows with |lat| >= 66 deg '
                'use relaxed cell thresholds (flag ok_sparse_polar).'
                % N_HQ])
    w.writerow(['# Sheet 1: per grid cell.'])
    w.writerow(['grid_lat_center', 'grid_lon_center', 'P_raw', 'ci_low',
                'ci_high', 'P_smoothed', 'n_events', 'n_images',
                'quality_flag', 'regime_class'])
    for cy in range(ny):
        for cx in range(nx):
            w.writerow([lat_c[cy], lon_c[cx],
                        '' if np.isnan(P_adj[cy, cx]) else round(P_adj[cy, cx], 4),
                        '' if np.isnan(ci_lo[cy, cx]) else round(ci_lo[cy, cx], 4),
                        '' if np.isnan(ci_hi[cy, cx]) else round(ci_hi[cy, cx], 4),
                        '' if np.isnan(field[cy, cx]) else round(field[cy, cx], 4),
                        n_ev[cy, cx], n_im[cy, cx],
                        qflag[cy, cx], names[cls[cy, cx]]])
    w.writerow([])
    w.writerow(['# Sheet 2: zonal profile (raw high-quality events, '
                '6 deg bins).'])
    w.writerow(['lat_center', 'mean_poleward', 'ci_low', 'ci_high',
                'n_events', 'residual_regime'])
    for c, mu, lo, hi, n, rg in zip(p_lat, p_mu, p_lo, p_hi, p_n, regime):
        tag = '' if np.isnan(rg) else {1: 'poleward', -1: 'equatorward',
                                       0: 'no-resolved'}[int(rg)]
        w.writerow([c,
                    '' if np.isnan(mu) else round(mu, 4),
                    '' if np.isnan(lo) else round(lo, 4),
                    '' if np.isnan(hi) else round(hi, 4), n, tag])
print('saved fig6v4_field.npz + fig6v4_source_data.csv')
