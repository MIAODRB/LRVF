# ============================================================
# Figure 2 | The global distribution and direction field of lunar rockfalls
# Mollweide projection with uniform central display density and zonal profile.
#   a (0,0,180,98)   Mollweide global: density base + 6x6 deg mean-direction arrows
#   b (0,102,118,60) Hercules G region: hillshade + per-event direction glyphs
#   c (122,102,58,60) NAC frame without trajectory-direction overlays
# Deviations from spec (documented): 6 deg grid instead of 5 deg to guarantee
# >=3 mm printed arrow spacing; panel-b event directions are drawn as
# fixed-length glyphs because event-level line length is not interpreted.
#
# This non-destructive derivative retains the uniform-density Mollweide trial
# and restores the authoritative Figure 2 marginal zonal-mean profile on the
# right. The profile uses the frozen full-catalogue 6-degree source table from
# paper/fig_v2/figure2; panel b and all non-overlay panel-c elements are
# unchanged.
#
# Panel-c cleanup (2026-09-03): remove the blue trajectory-direction indicator
# overlay from the NAC close-up. The underlying NAC crop, scale bar, north
# arrow, image identifier, locator and all other panels remain unchanged.
#
# Inherited v2 revisions (2026-07-13):
#   1. Panel a arrows enlarged + colour scale stretched to +/-0.25 with a
#      mid-grey pivot so weak poleward/equatorward tendencies are visible.
#   2. Panel b glyphs display-thinned on a 0.9 mm occupancy grid to unclutter
#      the crater floor (all events still used in stats; n label unchanged).
#   3. No-data hatching removed; polar cells now drawn for n >= 5 (all
#      sub-30 cells are poleward of 60 deg); n < 5 left as plain density.
#   4. Panel letters b/c lifted clear of the panel-b image edge.
# ============================================================
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Rectangle, FancyArrow
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs

# ---------------- global config ----------------
MM = 1 / 25.4
FIG_W, FIG_H = 180.0, 162.0
C_POLE, C_EQ = "#2166AC", "#B2182B"
C_NODATA = "#E0E0E0"
DENS_TOP = "#C9D2DA"
import argparse
_parser = argparse.ArgumentParser(description="Reproduce the approved Figure 2 display from frozen inputs.")
_parser.add_argument("--cache-dir", type=Path, default=DATA_DIR / "figure2" / "cache")
_parser.add_argument("--zonal-source", type=Path, default=DATA_DIR / "figure2" / "figure2_zonal_strip_source_data.csv")
_args = _parser.parse_args()
CACHE = str(_args.cache_dir)
OUT = str(OUTPUT_DIR / "figure2")
Path(OUT).mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 6,
    "axes.linewidth": 0.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "hatch.linewidth": 0.3,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

def mm_ax(fig, x, y, w, h, **kw):
    """axes from mm coords, origin top-left of canvas."""
    return fig.add_axes([x / FIG_W, (FIG_H - y - h) / FIG_H,
                         w / FIG_W, h / FIG_H], **kw)

# v2: |pmean| median ~0.09, p95 ~0.30 -> +/-0.5 rendered most arrows grey.
# Stretch to +/-0.25 and pivot on a mid-grey (visible on the pale density base),
# with early saturation stops so weak tendencies already read as red/blue.
cmap_dir = LinearSegmentedColormap.from_list("dir", [
    (0.00, C_EQ), (0.28, "#C4514B"), (0.50, "#9C9C9C"),
    (0.72, "#4B7FB5"), (1.00, C_POLE)])
norm_dir = Normalize(vmin=-0.25, vmax=0.25)

fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))

# ============================================================
# PANEL A
# ============================================================
proj = ccrs.Mollweide(central_longitude=0)
pc = ccrs.PlateCarree()
MAP_W, MAP_H = 146.0, 74.0
MAP_X, MAP_Y = 4.0, 7.5
ax_a = mm_ax(fig, MAP_X, MAP_Y, MAP_W, MAP_H, projection=proj)
ax_a.set_global()
ax_a.spines["geo"].set_linewidth(0.65)
ax_a.spines["geo"].set_edgecolor("#34383A")

# --- density base ---
dens = np.load(CACHE + r"/density_0p25.npy")
d2 = dens.reshape(360, 2, 720, 2).sum(axis=(1, 3))
dlog = np.log10(1 + gaussian_filter(d2, sigma=1.2))
dmax = np.percentile(dlog[dlog > 0], 98.0)
cmap_dens = LinearSegmentedColormap.from_list("dens", ["#FFFFFF", "#B8C4CF"])
lon_e = np.linspace(-180, 180, 721)
lat_e = np.linspace(-90, 90, 361)
ax_a.pcolormesh(lon_e, lat_e, np.clip(dlog / dmax, 0, 1) ** 0.8,
                cmap=cmap_dens, vmin=0, vmax=1, transform=pc,
                rasterized=True, zorder=1)
ax_a.gridlines(xlocs=np.arange(-180, 181, 60), ylocs=np.arange(-60, 61, 30),
               linewidth=0.30, color="#FFFFFF", alpha=0.72, zorder=2)

# --- arrows: 6x6 deg circular means ---
g = np.load(CACHE + r"/grid_6deg.npz")
cnt, Rbar, mean_az, pmean = g["cnt"], g["Rbar"], g["mean_az"], g["pmean"]
CELL = float(g["cell"])
ny, nx = cnt.shape
lat_c = -90 + (np.arange(ny) + 0.5) * CELL
lon_c = -180 + (np.arange(nx) + 0.5) * CELL
N_MIN = 30
N_MIN_POLAR = 5          # relaxed threshold poleward of 60 deg (v2)
POLAR_LAT = 60.0
ARROW_MIN_MM, ARROW_MAX_MM = 1.7, 4.0   # v2: enlarged from 1.2/3.4
MIN_LONGITUDE_STRIDE = 2   # display only: match the central rows to adjacent latitudes
x0, x1 = ax_a.get_xlim()
units_per_mm = (x1 - x0) / MAP_W
polar_row = np.abs(lat_c) >= POLAR_LAT
valid = (cnt >= N_MIN) | (polar_row[:, None] & (cnt >= N_MIN_POLAR))

keep = np.ones_like(valid)
for iy in range(ny):
    la = lat_c[iy]
    p0 = proj.transform_point(0, la, pc)
    p1 = proj.transform_point(CELL, la, pc)
    dx_mm = abs(p1[0] - p0[0]) / units_per_mm
    projected_stride = int(np.ceil(2.8 / max(dx_mm, 1e-6)))
    step = max(MIN_LONGITUDE_STRIDE, projected_stride)
    col = np.zeros(nx, bool)
    col[::step] = True
    keep[iy] &= col

segs, colors, head_polys = [], [], []
EPS = 0.05
for iy in range(ny):
    for ix_ in range(nx):
        if not (valid[iy, ix_] and keep[iy, ix_]):
            continue
        lo, la = lon_c[ix_], lat_c[iy]
        az = math.radians(mean_az[iy, ix_])
        dlon = EPS * math.sin(az) / max(math.cos(math.radians(la)), 0.05)
        dlat = EPS * math.cos(az)
        p0 = proj.transform_point(lo, la, pc)
        p1 = proj.transform_point(lo + dlon, np.clip(la + dlat, -89.9, 89.9), pc)
        v = np.array([p1[0] - p0[0], p1[1] - p0[1]])
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        v /= nv
        L = (ARROW_MIN_MM + (ARROW_MAX_MM - ARROW_MIN_MM) *
             np.clip(Rbar[iy, ix_] / 0.8, 0, 1)) * units_per_mm
        cpt = np.array(p0)
        tail, tip = cpt - v * L / 2, cpt + v * L / 2
        segs.append([tail, tip])
        colors.append(cmap_dir(norm_dir(pmean[iy, ix_])))
        hw, hl = 0.52 * units_per_mm, 1.05 * units_per_mm   # v2: bigger heads
        per = np.array([-v[1], v[0]])
        head_polys.append([tip, tip - v * hl + per * hw, tip - v * hl - per * hw])

ax_a.add_collection(LineCollection(segs, colors="white", linewidths=1.4,
                                   zorder=3, capstyle="round"))
ax_a.add_collection(LineCollection(segs, colors=colors, linewidths=0.85,
                                   zorder=4, capstyle="round"))
ax_a.add_collection(PolyCollection(head_polys, facecolors=colors,
                                   edgecolors="white", linewidths=0.2, zorder=4))

# --- insufficient-data hatching removed in v2: cells with n < 5 are left as
# --- plain density background (all sub-30 cells lie poleward of 60 deg and
# --- those with n >= 5 now receive arrows).

# --- latitude labels along left edge ---
for la in (60, 30, 0, -30, -60):
    p = proj.transform_point(-180, la, pc)
    ax_a.text(p[0] - 1.2 * units_per_mm, p[1], f"{abs(la)}°{'N' if la>0 else 'S' if la<0 else ''}",
              ha="right", va="center", fontsize=5, color="#808080")

# --- directly labelled 60-degree meridians, matching Supplementary Fig. 1 ---
for lo in (-120, -60, 0, 60, 120):
    p = proj.transform_point(lo, -69.0, pc)
    label = "0°" if lo == 0 else f"{abs(lo)}°{'E' if lo > 0 else 'W'}"
    ax_a.text(p[0], p[1], label, ha="center", va="center", fontsize=5,
              color="#707070", zorder=8,
              path_effects=[pe.withStroke(linewidth=1.05, foreground="white")])

# --- panel b locator: white box + label ---
B_LON0, B_LON1 = 38.079, 40.521
B_LAT0, B_LAT1 = 46.021, 46.879
bx = np.array([B_LON0, B_LON1, B_LON1, B_LON0, B_LON0])
by = np.array([B_LAT0, B_LAT0, B_LAT1, B_LAT1, B_LAT0])
bpts = proj.transform_points(pc, bx, by)[:, :2]
ax_a.plot(bpts[:, 0], bpts[:, 1], color="black", lw=1.6, zorder=5,
          solid_joinstyle="miter")
ax_a.plot(bpts[:, 0], bpts[:, 1], color="white", lw=0.75, zorder=6,
          solid_joinstyle="miter")
pb = proj.transform_point(39.3, 47.6, pc)
ax_a.text(pb[0], pb[1] + 1.0 * units_per_mm, "b", fontsize=7,
          fontweight="bold", ha="center", va="bottom", zorder=7,
          path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])

# --- marginal zonal-mean profile: frozen full catalogue, 6-degree bins ---
zonal = np.genfromtxt(_args.zonal_source,
                      delimiter=",", skip_header=2)
z_lat, z_mu, z_lo, z_hi = zonal.T
ok = np.isfinite(z_mu)
yz = np.array([proj.transform_point(0.0, float(la), pc)[1]
               for la in z_lat[ok]])

STRIP_X, STRIP_W = MAP_X + MAP_W + 4.5, 21.0
ax_z = mm_ax(fig, STRIP_X, MAP_Y, STRIP_W, MAP_H)
y_map_min, y_map_max = ax_a.get_ylim()
ax_z.set_ylim(y_map_min * 1.005, y_map_max * 1.005)
zfin = np.concatenate([z_lo[ok], z_hi[ok]])
ZLO = min(-0.30, float(np.floor(zfin.min() / 0.1) * 0.1) - 0.04)
ZHI = max(0.24, float(np.ceil(zfin.max() / 0.1) * 0.1) + 0.04)
ax_z.set_xlim(ZLO, ZHI)
ax_z.axvspan(ZLO, 0.0, color="#FBF2F1", zorder=0)
ax_z.axvspan(0.0, ZHI, color="#F2F6FA", zorder=0)
ax_z.axvline(0, lw=0.6, color="#555555", zorder=2)
ax_z.fill_betweenx(yz, z_lo[ok], z_hi[ok], color="#9AA4AD",
                   alpha=0.35, lw=0, zorder=3)
profile_colors = np.where(z_mu[ok] > 0, C_POLE, C_EQ)
ax_z.plot(z_mu[ok], yz, "-", lw=1.0, color="#5A6570", zorder=4)
ax_z.scatter(z_mu[ok], yz, s=7, c=profile_colors, edgecolors="white",
             linewidths=0.3, zorder=5, clip_on=True)
ax_z.text(0.0, 1.004, "0", transform=ax_z.get_xaxis_transform(),
          ha="center", va="bottom", fontsize=4.6, color="#8A8A8A", zorder=6)
ax_z.annotate("equatorward", xy=(0.02, 1.035), xycoords="axes fraction",
              fontsize=4.6, color=C_EQ, ha="left", va="bottom",
              style="italic", zorder=6, annotation_clip=False)
ax_z.annotate("poleward", xy=(0.98, 1.035), xycoords="axes fraction",
              fontsize=4.6, color=C_POLE, ha="right", va="bottom",
              style="italic", zorder=6, annotation_clip=False)
ax_z.set_yticks([])
for sp in ("top", "right", "left"):
    ax_z.spines[sp].set_visible(False)
ax_z.tick_params(axis="x", labelsize=4.5, length=2, pad=1.5)
ax_z.set_xticks([-0.4, -0.2, 0.0, 0.2])
ax_z.set_xticklabels(["-0.4", "-0.2", "0", "0.2"])
ax_z.set_xlabel("Zonal mean\npoleward comp.", fontsize=5.2, labelpad=1.5,
                linespacing=1.1, color="#555555")

# --- colorbar (direction) ---
cb_ax = mm_ax(fig, 62, 93.2, 54, 2.6)
cb = fig.colorbar(cm.ScalarMappable(norm=norm_dir, cmap=cmap_dir),
                  cax=cb_ax, orientation="horizontal")
cb.set_ticks([-0.2, -0.1, 0, 0.1, 0.2])
cb.ax.tick_params(labelsize=5, width=0.4, length=1.6, pad=1.2)
cb.outline.set_linewidth(0.4)
cb_ax.text(-0.02, 0.5, "equatorward", transform=cb_ax.transAxes, ha="right",
           va="center", fontsize=6, color=C_EQ)
cb_ax.text(1.02, 0.5, "poleward", transform=cb_ax.transAxes, ha="left",
           va="center", fontsize=6, color=C_POLE)
cb_ax.set_title("Mean poleward component", fontsize=6, pad=1.5)

# --- density mini-legend (bottom-right corner) ---
dl_ax = mm_ax(fig, 150, 94.0, 18, 1.8)
grad = np.linspace(0, 1, 128)[None, :]
dl_ax.imshow(grad, cmap=cmap_dens, aspect="auto", vmin=0, vmax=1)
dl_ax.set_xticks([]); dl_ax.set_yticks([])
for s in dl_ax.spines.values():
    s.set_linewidth(0.4); s.set_edgecolor("#909090")
dl_ax.set_title("Rockfall density (log)", fontsize=5, pad=1.2, color="#606060")
dl_ax.text(-0.04, 0.5, "0", transform=dl_ax.transAxes, ha="right", va="center",
           fontsize=5, color="#606060")
dl_ax.text(1.04, 0.5, "max", transform=dl_ax.transAxes, ha="left", va="center",
           fontsize=5, color="#606060")

# --- projection note (bottom-left corner) ---
note_ax = mm_ax(fig, 2, 88, 50, 10); note_ax.axis("off")
note_ax.text(0, 0.5,
             "Mollweide projection; 6° × 6° grid\n"
             "arrows: circular mean direction; length $\\propto\\ \\bar{R}$;\n"
             "n ≥ 30 (n ≥ 5 poleward of 60°)\n"
             "longitude display-thinning applied",
             fontsize=5, color="#606060", va="center")
# global n annotation, below the panel letter
na_ax = mm_ax(fig, 2, 6.0, 40, 5); na_ax.axis("off")
na_ax.text(0, 1, "n = 1,026,786 rockfalls", fontsize=6, color="#404040",
           va="top")

# ============================================================
# PANEL B  (Hercules G, hillshade + event direction glyphs)
# ============================================================
R_MOON = 1737400.0
DEG2M = math.pi * R_MOON / 180.0
LA_C, LO_C = 46.45, 39.30
coslat = math.cos(math.radians(LA_C))
la0b, la1b = B_LAT0, B_LAT1
lo0b, lo1b = B_LON0, B_LON1

dem = np.load(CACHE + r"/panelb_dem.npy")
gy, gx = np.gradient(dem, DEG2M / 512, DEG2M / 512 * coslat)
alt, az_l = math.radians(45), math.radians(315)
slope = np.arctan(np.hypot(gx, gy) * 2.0)
aspect_t = np.arctan2(-gx, gy)
hs = np.clip(np.sin(alt) * np.cos(slope) +
             np.cos(alt) * np.sin(slope) * np.cos(az_l - aspect_t), 0, 1) ** 0.9

ax_b = mm_ax(fig, 0, 102, 118, 60)
ax_b.imshow(hs, cmap="gray", vmin=-0.05, vmax=1.1,
            extent=[lo0b, lo1b, la0b, la1b], aspect=1 / coslat,
            interpolation="bilinear")
ax_b.set_xlim(lo0b, lo1b); ax_b.set_ylim(la0b, la1b)
ax_b.set_xticks([]); ax_b.set_yticks([])
for s in ax_b.spines.values():
    s.set_linewidth(0.5); s.set_edgecolor("#404040")

d = np.load(CACHE + r"/events.npz")
lon0e, lat0e, lon1e, lat1e = d["lon0"], d["lat0"], d["lon1"], d["lat1"]
lonm, latm = (lon0e + lon1e) / 2, (lat0e + lat1e) / 2
mB = (latm >= la0b) & (latm <= la1b) & (lonm >= lo0b) & (lonm <= lo1b)
azB = np.radians(d["az"][mB].astype(float))
pB = np.cos(azB)  # poleward component, northern hemisphere
lonB, latB = lonm[mB], latm[mB]

# v2: display thinning — the crater floor collects hundreds of converging
# glyphs into an unreadable knot. Keep at most one glyph per 0.9 mm occupancy
# cell (shuffled with a fixed seed so the pick is unbiased & reproducible).
# All mB.sum() events remain in the stats / n label.
THIN_MM = 0.9
mm_per_deg_x = 118 / (lo1b - lo0b)
mm_per_deg_y = 60 / (la1b - la0b)
cell_ix = ((lonB - lo0b) * mm_per_deg_x / THIN_MM).astype(int)
cell_iy = ((latB - la0b) * mm_per_deg_y / THIN_MM).astype(int)
cell_id = cell_iy * 10000 + cell_ix
rng = np.random.default_rng(42)
orderB = rng.permutation(len(lonB))
_, first_idx = np.unique(cell_id[orderB], return_index=True)
show = orderB[first_idx]
print(f"panel b glyphs: {len(show)} shown of {mB.sum()} events")

AMP_MM = 1.55
deg_per_mm_x = (lo1b - lo0b) / 118
segsB, colB, headB = [], [], []
for cx, cy, azi, pi in zip(lonB[show], latB[show], azB[show], pB[show]):
    dx = math.sin(azi) / coslat
    dy = math.cos(azi)
    nvv = math.hypot(dx * coslat, dy)
    s = AMP_MM * deg_per_mm_x * coslat / nvv
    tail = (cx - dx * s / 2, cy - dy * s / 2)
    tip = (cx + dx * s / 2, cy + dy * s / 2)
    segsB.append([tail, tip])
    colB.append(cmap_dir(norm_dir(pi)))
    hw = 0.28 * deg_per_mm_x
    hl = 0.62 * deg_per_mm_x
    # unit vector in display-metric space
    ux, uy = dx * s / (AMP_MM * deg_per_mm_x), dy * s / (AMP_MM * deg_per_mm_x)
    per = (-uy * coslat, ux / coslat)
    headB.append([tip,
                  (tip[0] - ux * hl + per[0] * hw, tip[1] - uy * hl + per[1] * hw),
                  (tip[0] - ux * hl - per[0] * hw, tip[1] - uy * hl - per[1] * hw)])
ax_b.add_collection(LineCollection(segsB, colors="white", linewidths=0.75,
                                   zorder=3, capstyle="round", alpha=0.9))
ax_b.add_collection(LineCollection(segsB, colors=colB, linewidths=0.45,
                                   zorder=4, capstyle="round"))
ax_b.add_collection(PolyCollection(headB, facecolors=colB, edgecolors="none",
                                   zorder=4))

import matplotlib.patheffects as pe
halo = [pe.withStroke(linewidth=1.6, foreground="black")]
ax_b.text(0.015, 0.955, "Hercules G   46.5° N, 39.3° E", transform=ax_b.transAxes,
          fontsize=6.5, color="white", va="top", path_effects=halo)
ax_b.text(0.015, 0.865, f"n = {mB.sum():,} rockfalls", transform=ax_b.transAxes,
          fontsize=5.5, color="white", va="top", path_effects=halo)

# scale bar 10 km
sb_km = 10.0
sb_deg = sb_km * 1000 / (DEG2M * coslat)
sx0, sy0 = lo0b + 0.045 * (lo1b - lo0b), la0b + 0.075 * (la1b - la0b)
ax_b.plot([sx0, sx0 + sb_deg], [sy0, sy0], color="white", lw=1.2,
          solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
ax_b.text(sx0 + sb_deg / 2, sy0 + 0.025 * (la1b - la0b), "10 km", ha="center",
          va="bottom", fontsize=5.5, color="white", path_effects=halo)

# north arrow (upper-right, clear of dark terrain)
nx0, ny0 = lo1b - 0.035 * (lo1b - lo0b), la1b - 0.24 * (la1b - la0b)
ax_b.annotate("", xy=(nx0, ny0 + 0.16 * (la1b - la0b)), xytext=(nx0, ny0),
              arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.3",
                              lw=1.0, color="white",
                              path_effects=[pe.withStroke(linewidth=2.0, foreground="black")]))
ax_b.text(nx0, ny0 - 0.035 * (la1b - la0b), "N", ha="center", va="top",
          fontsize=6, color="white", path_effects=halo)

# ============================================================
# PANEL C  (NAC frame without trajectory-direction overlays)
# ============================================================
UL = (47.15, 39.23); UR = (47.15, 39.13); LR = (46.19, 39.15); LL = (46.19, 39.24)
W_IMG, H_IMG = 5064, 52224

def ll_to_px(lat, lon):
    v = (UL[0] - lat) / (UL[0] - LL[0])
    lon_left = UL[1] + v * (LL[1] - UL[1])
    lon_right = UR[1] + v * (LR[1] - UR[1])
    u = (lon_left - lon) / (lon_left - lon_right)
    return u * W_IMG, v * H_IMG

def px_to_ll(x, y):
    v = y / H_IMG
    lon_left = UL[1] + v * (LL[1] - UL[1])
    lon_right = UR[1] + v * (LR[1] - UR[1])
    lat = UL[0] - v * (UL[0] - LL[0])
    lon = lon_left - (x / W_IMG) * (lon_left - lon_right)
    return lat, lon

# metric pixel scale from footprint
w_m = abs(UR[1] - UL[1]) * DEG2M * math.cos(math.radians(46.67))
h_m = abs(UL[0] - LL[0]) * DEG2M
PXW, PXH = w_m / W_IMG, h_m / H_IMG   # ~0.41, ~0.56 m

x0c, y0c = np.load(CACHE + r"/nac_crop_0_origin.npy")
arr = np.load(CACHE + r"/nac_crop_0.npy")
# sub-region: aspect matched to 58x60 panel, tight on the boulder cluster
SX0, SY0, SW, SH = 380, 550, 620, 470
sub = arr[SY0:SY0 + SH, SX0:SX0 + SW]
loP, hiP = np.percentile(sub, [0.5, 99.5])
sub_img = np.clip((sub.astype(float) - loP) / (hiP - loP), 0, 1)

ax_c = mm_ax(fig, 122, 102, 58, 60)
ext = [0, SW * PXW, SH * PXH, 0]  # metres, y down
ax_c.imshow(sub_img, cmap="gray", extent=ext, aspect=1, interpolation="bilinear")
ax_c.set_xticks([]); ax_c.set_yticks([])
for s in ax_c.spines.values():
    s.set_linewidth(0.5); s.set_edgecolor("#404040")

gx0, gy0 = x0c + SX0, y0c + SY0

# scale bar 100 m
sbx0, sby0 = 0.05 * SW * PXW, 0.93 * SH * PXH
ax_c.plot([sbx0, sbx0 + 100], [sby0, sby0], color="white", lw=1.2,
          solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
ax_c.text(sbx0 + 50, sby0 - 6, "100 m", ha="center", va="bottom", fontsize=5.5,
          color="white", path_effects=halo)
ax_c.text(0.97, 0.03, "LROC NAC M142007158LC", transform=ax_c.transAxes,
          ha="right", va="bottom", fontsize=5, color="white", path_effects=halo)

# north arrow for NAC frame (from footprint geometry)
cx_px, cy_px = gx0 + SW / 2, gy0 + SH / 2
la_c0, lo_c0 = px_to_ll(cx_px, cy_px)
pxN, pyN = ll_to_px(la_c0 + 0.01, lo_c0)
vN = np.array([(pxN - cx_px) * PXW, (pyN - cy_px) * PXH])
vN /= np.linalg.norm(vN)
nax, nay = 0.93 * SW * PXW, 0.175 * SH * PXH
ax_c.annotate("", xy=(nax + vN[0] * 26, nay + vN[1] * 26), xytext=(nax, nay),
              arrowprops=dict(arrowstyle="-|>,head_width=0.16,head_length=0.3",
                              lw=1.0, color="white",
                              path_effects=[pe.withStroke(linewidth=2.0, foreground="black")]))
ax_c.text(nax, nay + 8, "N", ha="center", va="top",
          fontsize=6, color="white", path_effects=halo)

# --- panel c locator inside panel b: white box + leader lines ---
corner_px = [(gx0, gy0), (gx0 + SW, gy0), (gx0 + SW, gy0 + SH), (gx0, gy0 + SH)]
corner_ll = [px_to_ll(x, y) for x, y in corner_px]
cbx = [ll[1] for ll in corner_ll] + [corner_ll[0][1]]
cby = [ll[0] for ll in corner_ll] + [corner_ll[0][0]]
ax_b.plot(cbx, cby, color="black", lw=1.5, zorder=6)
ax_b.plot(cbx, cby, color="white", lw=0.7, zorder=7)
# leader lines from box right corners to panel c left corners (figure coords)
from matplotlib.patches import ConnectionPatch
lat_top = max(ll[0] for ll in corner_ll)
lat_bot = min(ll[0] for ll in corner_ll)
lon_r = max(ll[1] for ll in corner_ll)
for (yy, cy_ax) in [(lat_top, 1.0), (lat_bot, 0.0)]:
    con = ConnectionPatch(xyA=(lon_r, yy), coordsA=ax_b.transData,
                          xyB=(0, cy_ax), coordsB=ax_c.transAxes,
                          color="#909090", lw=0.5, linestyle=(0, (2, 2)))
    fig.add_artist(con)

# ============================================================
# panel letters
# ============================================================
# v2: letters b/c bottom-anchored 1 mm above the panel tops (y = 102 mm) —
# previously they overhung the panel-b image, which is flush with x = 0.
fig.text(1.0 / FIG_W, 1 - 1.0 / FIG_H, "a", fontsize=8, fontweight="bold",
         va="top", ha="left")
fig.text(1.0 / FIG_W, 1 - 101.0 / FIG_H, "b", fontsize=8, fontweight="bold",
         va="bottom", ha="left")
fig.text(120.0 / FIG_W, 1 - 101.0 / FIG_H, "c", fontsize=8, fontweight="bold",
         va="bottom", ha="left")

# ============================================================
# export
# ============================================================
fig.savefig(OUT + r"/Figure2.pdf", dpi=600)
fig.savefig(OUT + r"/Figure2.png", dpi=600, facecolor="white")
fig.savefig(OUT + r"/Figure2_preview.png", dpi=200, facecolor="white")
print("done")
