#!/usr/bin/env python3
"""Figure 3 v3 — adds the architecture-document fixes:
   panel a: Rayleigh statement + mean bearing (no unqualified 'isotropy')
   panel b: N-S profile agreement r, poleward-fraction effect-size annotation
   panel c: unchanged construction
Recomputed directly from figure3_event_cache.npz (base script was absent).
"""
import json, math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse import coo_matrix

import argparse
_parser = argparse.ArgumentParser(description='Reproduce the submitted Figure 3 v3 from the frozen event cache.')
_parser.add_argument('--event-cache', type=Path, default=DATA_DIR / 'figure3' / 'figure3_event_cache.npz')
_args = _parser.parse_args()
OUT = OUTPUT_DIR / 'figure3'; OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
DARK = '#333333'
BLUE = '#3B6FB6'
ORANGE = '#D68132'
ROSE = '#93A1AF'
REFC = '#5F6265'
LIGHT = '#E9E9E9'
GRID = '#D8D8D8'
B = 800
SEED = 20260714
REV_N, REV_S = 63.48970201322201, 66.66960727698721
REV_MID = 65.07965464510461
POLAR = 80.0

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7.0, 'axes.labelsize': 7.0,
    'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'legend.fontsize': 6.2,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})

ev = dict(np.load(_args.event_cache))
lat = ev['latitude'].astype(np.float64)
bearing = ev['bearing'].astype(np.float64)
alpha = ev['alpha'].astype(np.float64)
pole = ev['poleward_component'].astype(np.float64)
code = ev['nac_code']
n_ev = len(lat)
abs_lat = np.abs(lat)
north = lat >= 0

# ---------------------------------------------------------------- global stats
sinb, cosb = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
R = float(np.hypot(sinb.mean(), cosb.mean()))
mean_brg = float(np.degrees(np.arctan2(sinb.mean(), cosb.mean())) % 360)
z = n_ev * R * R

# ---------------------------------------------------------------- panel b stats
edges = np.arange(0.0, 92.0, 2.0)
centers = 0.5 * (edges[:-1] + edges[1:])


def smoothed_ratio(num, den):
    n = gaussian_filter1d(num, 1.0, axis=-1, mode='nearest')
    d = gaussian_filter1d(den, 1.0, axis=-1, mode='nearest')
    with np.errstate(divide='ignore', invalid='ignore'):
        return n / d


def hemi_curve(mask, seed):
    bi = np.searchsorted(edges, abs_lat, side='right') - 1
    keep = mask & (bi >= 0) & (bi < len(centers))
    ac, ri = np.unique(code[keep], return_inverse=True)
    ci = bi[keep]
    vals = pole[keep]
    shape = (len(ac), len(centers))
    sums = coo_matrix((vals, (ri, ci)), shape=shape).tocsr()
    cnts = coo_matrix((np.ones_like(vals), (ri, ci)), shape=shape).tocsr()
    evc = np.asarray(cnts.sum(axis=0)).ravel()
    imc = np.asarray((cnts > 0).sum(axis=0)).ravel()
    mean = smoothed_ratio(np.asarray(sums.sum(axis=0)).ravel(), evc)
    rng = np.random.default_rng(seed)
    boot = np.empty((B, len(centers)), np.float32)
    for s in range(0, B, 40):
        e = min(s + 40, B)
        w = rng.poisson(1.0, size=(e - s, shape[0])).astype(np.float32)
        boot[s:e] = smoothed_ratio(np.asarray((sums.T @ w.T).T),
                                   np.asarray((cnts.T @ w.T).T))
    lo = np.full(len(centers), np.nan); hi = np.full(len(centers), np.nan)
    fin = np.any(np.isfinite(boot), axis=0)
    lo[fin], hi[fin] = np.nanpercentile(boot[:, fin], [2.5, 97.5], axis=0)
    valid = (evc >= 500) & (imc >= 10)
    ff = np.flatnonzero((centers >= 70) & ~valid)
    if len(ff):
        valid[ff[0]:] = False
    return dict(mean=mean, lo=lo, hi=hi, n=evc, im=imc, valid=valid)


cn = hemi_curve(north, SEED + 1)
cs = hemi_curve(~north, SEED + 2)
both = cn['valid'] & cs['valid'] & (centers >= 5)
r_prof = float(np.corrcoef(cn['mean'][both], cs['mean'][both])[0, 1])
r_lo, r_hi = float(centers[both].min()), float(centers[both].max())
mid = (abs_lat >= 30) & (abs_lat < 60)
frac_mid = float((np.abs(alpha[mid]) < 90).mean())
print('r=%.3f over %.0f-%.0f; poleward fraction mid = %.3f' % (r_prof, r_lo, r_hi, frac_mid))

# ---------------------------------------------------------------- panel c stats
a_edges = np.arange(-180, 181, 5.0)
l_edges = np.arange(0, 91, 1.0)
H, _, _ = np.histogram2d(alpha, abs_lat, bins=[a_edges, l_edges])
col_n = H.sum(axis=0)
# image support per 1-deg column
li = np.clip(np.searchsorted(l_edges, abs_lat, side='right') - 1, 0, 89)
pairs = np.unique(np.stack([li, code]), axis=1)
col_im = np.bincount(pairs[0], minlength=90)
with np.errstate(divide='ignore', invalid='ignore'):
    P = 100.0 * H / col_n[None, :]
Ps = gaussian_filter(P, sigma=(1.1, 1.0), mode=('wrap', 'nearest'))
sup = (col_n >= 500) & (col_im >= 10)
# The raw support test is non-monotonic near the polar observing limit: an
# isolated column can pass while its neighbours fail, which rendered a detached
# stripe in the earlier version. Truncate at the first failing column instead of
# showing islands of support.
first_fail = np.flatnonzero(~sup)
if first_fail.size:
    sup[first_fail.min():] = False
last = np.flatnonzero(sup).max()
VMIN, VMAX = 0.34475549524578175, 2.5477512054109464

# ---------------------------------------------------------------- draw
fig = plt.figure(figsize=(180 * MM, 135 * MM))

# panel a — rose
ax_a = fig.add_axes([0.065, 0.68, 0.21, 0.27], projection='polar')
rose_edges = np.linspace(0, 360, 37)
cnt, _ = np.histogram(bearing, bins=rose_edges)
pct = cnt / cnt.sum() * 100
theta = np.deg2rad(0.5 * (rose_edges[:-1] + rose_edges[1:]))
width = np.deg2rad(np.diff(rose_edges) * 0.97)
uniform = 100.0 / 36
r_max = max(pct.max() * 1.13, uniform * 1.24)
ax_a.bar(theta, pct, width=width, color=ROSE, edgecolor='white', linewidth=0.42, zorder=2)
for ang in np.deg2rad([0, 90, 180, 270]):
    ax_a.plot([ang, ang], [0, r_max], color='#B8B8B8', lw=0.65, zorder=3)
ax_a.plot(np.linspace(0, 2 * np.pi, 361), np.full(361, uniform), color=REFC,
          lw=0.95, ls=(0, (3.0, 2.1)), zorder=4)
ax_a.set_theta_zero_location('N'); ax_a.set_theta_direction(-1)
ax_a.set_xticks(np.deg2rad([0, 90, 180, 270]))
ax_a.set_xticklabels(['N', 'E', 'S', 'W'])
ax_a.tick_params(axis='x', pad=5.5, labelsize=7.0)
ax_a.set_ylim(0, r_max); ax_a.set_yticks([]); ax_a.grid(False)
ax_a.spines['polar'].set_color('#55585B'); ax_a.spines['polar'].set_linewidth(0.75)
ax_a.text(0.5, -0.19, rf'$R = {R:.3f}$  ($n = {n_ev:,}$)', transform=ax_a.transAxes,
          ha='center', va='top', fontsize=6.6, color=DARK, clip_on=False)
ax_a.text(0.5, -0.31, 'absolute azimuth', transform=ax_a.transAxes, ha='center',
          va='top', fontsize=6.6, fontstyle='italic', color=DARK, clip_on=False)
ax_a.text(0.5, -0.43,
          f'uniformity formally rejected (Rayleigh $z\\approx{z:.0f}$);\n'
          f'residual mean bearing {mean_brg:.1f}° reflects the N–S\n'
          'asymmetry of the latitude signal (see caption)',
          transform=ax_a.transAxes, ha='center', va='top', fontsize=5.4,
          color='#6A6A6A', clip_on=False, zorder=10)
ax_a.text(-0.19, 1.13, 'a', transform=ax_a.transAxes, fontsize=9, fontweight='bold')

# panel b — hemispheric profiles
ax_b = fig.add_axes([0.415, 0.615, 0.545, 0.335])
ax_b.axhline(0, color='#7A7A7A', lw=0.8, zorder=1)
ax_b.axvspan(0, 2, color=LIGHT, alpha=0.55, lw=0, zorder=0)
ax_b.axvspan(min(REV_N, REV_S), max(REV_N, REV_S), color='#BDBDBD', alpha=0.30, lw=0, zorder=0)
ax_b.axvline(REV_MID, color='#666666', lw=0.8, ls=(0, (2.5, 2.0)), zorder=1)
ax_b.axvspan(POLAR, 90, color=LIGHT, alpha=0.85, lw=0, zorder=0)
for st, color, label, ls in ((cn, BLUE, 'Northern hemisphere', '-'),
                             (cs, ORANGE, 'Southern hemisphere', (0, (4.0, 2.0)))):
    m = np.where(st['valid'], st['mean'], np.nan)
    lo = np.where(st['valid'], st['lo'], np.nan)
    hi = np.where(st['valid'], st['hi'], np.nan)
    ax_b.fill_between(centers, lo, hi, color=color, alpha=0.16, lw=0, zorder=2)
    ax_b.plot(centers, m, color=color, ls=ls, lw=1.25, label=label, zorder=3)
ax_b.set_xlim(0, 90); ax_b.set_ylim(-0.40, 0.40)
ax_b.yaxis.set_major_locator(MultipleLocator(0.05))
ax_b.xaxis.set_major_locator(MultipleLocator(15))
ax_b.grid(axis='y', color=GRID, lw=0.45, zorder=0)
ax_b.set_xlabel('Absolute latitude, |latitude| (°)')
ax_b.set_ylabel('Mean poleward component, P')
ax_b.legend(loc='lower left', frameon=False, handlelength=2.4, borderaxespad=0.2)
ax_b.annotate('High-latitude\ndirectional reversal', xy=(REV_MID, 0.02),
              xytext=(76.5, 0.24), fontsize=6.0, ha='center', va='bottom',
              arrowprops=dict(arrowstyle='-', color='#666666', lw=0.7))
ax_b.text(80.8, -0.27, 'Polar\nobserving limit', fontsize=5.8, color='#9A9A9A',
          ha='left', va='center')
ax_b.text(27, 0.345, f'hemispheric profile agreement '
          rf'$r = {r_prof:.2f}$ ({r_lo:.0f}–{r_hi:.0f}°)', fontsize=6.0,
          ha='center', va='center', color=DARK)
ax_b.text(27, 0.29, rf'$\approx${frac_mid * 100:.0f}% of 30–60° events move poleward',
          fontsize=6.0, ha='center', va='center', color='#6A6A6A')
for s in ('top', 'right'):
    ax_b.spines[s].set_visible(False)
ax_b.text(-0.115, 1.10, 'b', transform=ax_b.transAxes, fontsize=9, fontweight='bold')

# panel c — conditional density
ax_c = fig.add_axes([0.105, 0.075, 0.77, 0.40])
disp = Ps.copy()
disp[:, ~sup] = np.nan
cmap_c = plt.get_cmap('Blues').copy()
cmap_c.set_bad('#EBEBEB')
mesh = ax_c.pcolormesh(l_edges, a_edges, np.ma.masked_invalid(np.clip(disp, VMIN, VMAX)),
                       cmap=cmap_c, vmin=VMIN, vmax=VMAX, shading='flat', rasterized=True)
ax_c.axvspan(POLAR, 90, color='#E3E3E3', lw=0, zorder=2)
ax_c.axvline(REV_MID, color='#40474D', lw=0.8, ls=(0, (2.5, 2.0)), zorder=4)
ax_c.annotate('High-latitude reversal', xy=(REV_MID, 5), xytext=(48, 62),
              fontsize=6.2, ha='center',
              arrowprops=dict(arrowstyle='-', color='#5A5A5A', lw=0.7), zorder=5)
ax_c.set_xlim(0, 90); ax_c.set_ylim(-180, 180)
ax_c.set_yticks([-180, -90, 0, 90, 180])
ax_c.set_yticklabels(['−180°\nEquatorward', '−90°\nAlong latitude', '0°\nPoleward',
                      '+90°\nAlong latitude', '+180°\nEquatorward'], fontsize=6.0)
ax_c.xaxis.set_major_locator(MultipleLocator(15))
ax_c.set_xlabel('Absolute latitude, |latitude| (°)')
ax_c.set_ylabel('Relative direction, α', labelpad=2)
cax = fig.add_axes([0.905, 0.075, 0.018, 0.40])
cb = fig.colorbar(mesh, cax=cax)
cb.set_ticks(np.arange(0.5, 2.6, 0.25))
cax.tick_params(labelsize=6.0, width=0.5, length=2)
cb.outline.set_linewidth(0.5)
ax_c.text(-0.095, 1.06, 'c', transform=ax_c.transAxes, fontsize=9, fontweight='bold')

for ext in ('png', 'pdf', 'tiff'):
    kw = dict(dpi=600) if ext != 'pdf' else {}
    if ext == 'tiff':
        kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
    fig.savefig(OUT / f'Figure3.{ext}', facecolor='white', **kw)
plt.close(fig)

# ---------------------------------------------------------------- source data
import csv
with open(OUT / 'figure3_source_data.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['# Figure 3 source data. Panel a: 10-deg azimuth histogram; '
                'panel b: 2-deg latitude profiles; panel c: conditional density grid '
                'available on request (72x90).'])
    w.writerow(['# Panel a'])
    w.writerow(['azimuth_bin_start_deg', 'azimuth_bin_end_deg', 'percent'])
    e10 = np.arange(0, 361, 10)
    c10, _ = np.histogram(bearing, bins=e10)
    for a0, a1, p in zip(e10[:-1], e10[1:], c10 / c10.sum() * 100):
        w.writerow([a0, a1, round(p, 4)])
    w.writerow([])
    w.writerow(['# Panel b'])
    w.writerow(['abs_lat_center', 'north_P', 'north_lo', 'north_hi', 'north_events',
                'north_images', 'south_P', 'south_lo', 'south_hi', 'south_events',
                'south_images'])
    for i, cc in enumerate(centers):
        w.writerow([cc] + [round(float(x), 5) if np.isfinite(x) else '' for x in
                           (cn['mean'][i], cn['lo'][i], cn['hi'][i])] +
                   [int(cn['n'][i]), int(cn['im'][i])] +
                   [round(float(x), 5) if np.isfinite(x) else '' for x in
                    (cs['mean'][i], cs['lo'][i], cs['hi'][i])] +
                   [int(cs['n'][i]), int(cs['im'][i])])

summary = dict(R=R, mean_bearing_deg=mean_brg, rayleigh_z=z,
               NS_profile_r=r_prof, r_lat_range=[r_lo, r_hi],
               poleward_fraction_mid=frac_mid,
               reversal_north=REV_N, reversal_south=REV_S,
               bootstrap_replicates=B, bootstrap_cluster='nac_id')
(OUT / 'figure3_v3_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('done', summary)
