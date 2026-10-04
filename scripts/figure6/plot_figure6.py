# -*- coding: utf-8 -*-
"""
Figure 6 | A framework for latitude-organized lunar mass wasting
180 x 110 mm, Nature Communications double-column.

The approved design uses the 0-degree central-meridian
Mollweide projection and the authoritative equatorial dashed locator
and dagger for the equatorial anomaly. The frozen field, zonal profile, colour
scale, sample size, panel-b artwork and all scientific values remain unchanged.
"""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR as ROOT_OUTPUT_DIR
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch
import matplotlib.cm as cm
import cartopy.crs as ccrs

# ------------------------------------------------------------------ constants
MM = 1 / 25.4
W_MM, H_MM = 180.0, 110.0
BLUE, RED = '#2166AC', '#B2182B'
INK, INK2 = '#1A1A1A', '#555555'
BOX_EC, TERRAIN_F, TERRAIN_E = '#D8D8D8', '#F1EFEC', '#8B8B8B'
B_BLUE, B_RED = '#C9DCEC', '#EFD3D1'
NEUTRAL = '#ECE8E1'
HATCH_EC = '#AEB6BD'

plt.rcParams.update({
    'font.family': 'Arial',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.linewidth': 0.5,
    'text.color': INK, 'axes.edgecolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'axes.labelcolor': INK,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'hatch.linewidth': 0.4, 'hatch.color': HATCH_EC,
})

CMAP = LinearSegmentedColormap.from_list('dir', [
    (0.00, RED), (0.22, '#CE7A72'), (0.42, '#E7C9C4'),
    (0.50, NEUTRAL), (0.58, '#BBD0E3'), (0.78, '#5E93C2'), (1.00, BLUE)])

# ------------------------------------------------------------------ paths/data
SOURCE_DIR = DATA_DIR / 'figure6'
OUTPUT_DIR = ROOT_OUTPUT_DIR / 'figure6'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# By default render the frozen field. To render a rebuilt field, set
# LRVF_FIGURE6_FIELD to the NPZ written by build_field.py.
FIELD_PATH = Path(os.environ.get('LRVF_FIGURE6_FIELD', str(SOURCE_DIR / 'fig6v4_field.npz')))
d = np.load(FIELD_PATH)
F = d['field_disp'].astype(float)
lat_c = d['lat_c']; lon_c = d['lon_c']
CELL = float(d['cell']); VMAX = float(d['vmax'])
N_HQ = int(d['n_hq'])
p_lat = d['p_lat']; p_mu = d['p_mu']; p_lo = d['p_lo']; p_hi = d['p_hi']
regime = d['regime']

proj = ccrs.Mollweide(central_longitude=0)
PC = ccrs.PlateCarree()
Y90 = proj.transform_point(0, 90, PC)[1]


def make_fig(w_mm, h_mm):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM), dpi=600)

    def add_panel(x, y, w, h, **kw):
        return fig.add_axes([x / w_mm, 1 - (y + h) / h_mm,
                             w / w_mm, h / h_mm], **kw)

    def ftext(x, y, s, **kw):
        return fig.text(x / w_mm, 1 - y / h_mm, s, **kw)

    return fig, add_panel, ftext


# ================================================================== panel a
def draw_panel_a(fig, add_panel, ftext, x0=0.0, y0=0.0):
    MAP_X, MAP_Y, MAP_W = x0 + 10.0, y0 + 6.0, 112.0
    MAP_H = MAP_W * (Y90 / (proj.x_limits[1]))
    PROF_X, PROF_W = MAP_X + MAP_W + 3.0, 20.0
    CB_X = PROF_X + PROF_W + 9.5

    # ---- smooth field map ----
    ax = add_panel(MAP_X, MAP_Y, MAP_W, MAP_H, projection=proj)
    ax.set_global()
    ax.spines['geo'].set_linewidth(0.65); ax.spines['geo'].set_edgecolor('#34383A')

    # wrap in longitude and extend to the poles so the grids cover the globe
    lon_p = np.concatenate([[lon_c[0] - CELL], lon_c, [lon_c[-1] + CELL]])
    lat_p = np.concatenate([[-90.0], lat_c, [90.0]])

    def wrap(M):
        Mw = np.column_stack([M[:, -1], M, M[:, 0]])
        return np.vstack([Mw[0], Mw, Mw[-1]])

    # fill the few unsupported grid cells inside the belt by longitude-wrapped
    # Laplace relaxation (display only): scattered hatched holes fragment the
    # field, so the hatched 'no effective detection' mask is reserved for the
    # contiguous polar caps beyond the last majority-supported row
    Ff = F.copy()
    hole = ~np.isfinite(Ff)
    Ff[hole] = 0.0
    for _ in range(400):
        up = np.vstack([Ff[:1], Ff[:-1]])
        dn = np.vstack([Ff[1:], Ff[-1:]])
        Ff[hole] = 0.25 * (up + dn + np.roll(Ff, 1, axis=1)
                           + np.roll(Ff, -1, axis=1))[hole]

    F_p = wrap(Ff)
    Lon, Lat = np.meshgrid(lon_p, lat_p)
    levels = np.linspace(-VMAX, VMAX, 15)

    ax.set_facecolor(NEUTRAL)   # neutral globe base: holes read as 'no preference'
    cf = ax.contourf(Lon, Lat, np.clip(F_p, -VMAX, VMAX), levels=levels,
                     cmap=CMAP, norm=Normalize(-VMAX, VMAX), transform=PC,
                     transform_first=True, extend='both', zorder=1)
    cf.set_edgecolor('face')

    # hatched no-effective-detection mask: smooth polar-cap bands poleward of
    # the last latitude row carrying any supported cells (v4: the sparse
    # 66-78 deg rows recovered by the relaxed thresholds stay visible; the
    # interior of those rows is completed by the display-only fill above)
    frac = np.isfinite(F).mean(axis=1)
    sup = frac >= 0.15
    lat_hi = lat_c[sup].max() + CELL / 2
    lat_lo = lat_c[sup].min() - CELL / 2
    lat_f = np.linspace(-90, 90, 721)
    lon_f = np.linspace(-180, 180, 145)
    LonF, LatF = np.meshgrid(lon_f, lat_f)
    CAPF = ((LatF >= lat_hi) | (LatF <= lat_lo)).astype(float)
    nd = ax.contourf(LonF, LatF, CAPF, levels=[0.5, 1.5], colors=['#F4F4F2'],
                     hatches=['////'], transform=PC, zorder=1.4)
    # hatch lines take the collection edgecolor; keep boundary stroke invisible
    nd.set_edgecolor(HATCH_EC)
    nd.set_linewidth(0.0)
    # label the polar masks directly (replaces the v2 explanatory line)
    for la in ((90.0 + lat_hi) / 2, (-90.0 + lat_lo) / 2):
        p = proj.transform_point(0, la, PC)
        ax.text(p[0], p[1], 'no effective detection', ha='center', va='center',
                fontsize=4.5, color='#8A9097', zorder=5,
                bbox=dict(boxstyle='round,pad=0.25', fc='#F4F4F2', ec='none'))

    ax.gridlines(xlocs=np.arange(-120, 121, 60), ylocs=np.arange(-60, 61, 30),
                 linewidth=0.30, color='#FFFFFF', alpha=0.72, zorder=3)
    eq = proj.transform_points(PC, np.linspace(-180, 180, 181),
                               np.zeros(181))[:, :2]
    ax.plot(eq[:, 0], eq[:, 1], color='#5A6570', lw=0.5, zorder=3.5)

    # Restore the Figure 6 v5 equatorial-anomaly locator. The display box is
    # deliberately wider than the strongest cells (~120-165 deg W) so the
    # anomaly remains legible at final print size.
    box_lon = [-180, -112, -112, -180, -180]
    box_lat = [-5, -5, 13, 13, -5]
    for i in range(4):
        xs = np.linspace(box_lon[i], box_lon[i + 1], 40)
        ys = np.linspace(box_lat[i], box_lat[i + 1], 40)
        ax.plot(xs, ys, transform=PC, color='#6E4B49', lw=0.7,
                ls=(0, (2.2, 1.8)), zorder=6)
    pd_ = proj.transform_point(-106, 11.5, PC)
    ax.text(pd_[0], pd_[1], '†', ha='left', va='center', fontsize=6.5,
            color='#6E4B49', zorder=6)

    x_l, x_r = ax.get_xlim()
    upmm = (x_r - x_l) / MAP_W
    for la in (60, 30, 0, -30, -60):
        p = proj.transform_point(-180, la, PC)
        lbl = f"{abs(la)}°{'N' if la > 0 else 'S' if la < 0 else ''}"
        ax.text(p[0] - 1.4 * upmm, p[1], lbl, ha='right', va='center',
                fontsize=5, color='#7A8691')

    # Directly labelled key meridians, matching Supplementary Figure 1.
    for lo in (-120, -60, 0, 60, 120):
        p = proj.transform_point(lo, -69.0, PC)
        lbl = '0°' if lo == 0 else f"{abs(lo)}°{'E' if lo > 0 else 'W'}"
        ax.text(p[0], p[1], lbl, ha='center', va='center', fontsize=5,
                color='#707070', zorder=8,
                path_effects=[pe.withStroke(linewidth=1.05, foreground='white')])

    # ---- aligned marginal zonal profile ----
    axp = add_panel(PROF_X, MAP_Y, PROF_W, MAP_H)

    def ytr(latd):
        return np.array([proj.transform_point(0, float(l), PC)[1]
                         for l in np.atleast_1d(latd)])
    ok = np.isfinite(p_mu)
    yv = ytr(p_lat[ok])
    mu = p_mu[ok]; lo = p_lo[ok]; hi = p_hi[ok]; rg = regime[ok]

    axp.set_ylim(proj.y_limits); axp.set_xlim(-0.40, 0.26)
    axp.axvline(0, lw=0.6, color=INK2, zorder=2)
    axp.axvspan(0.0, 0.26, color='#F2F6FA', zorder=0)
    axp.axvspan(-0.40, 0.0, color='#FBF2F1', zorder=0)
    axp.fill_betweenx(yv, lo, hi, color='#9AA4AD', alpha=0.30, lw=0, zorder=3)
    cols = np.where(rg > 0, BLUE, np.where(rg < 0, RED, '#9AA4AD'))
    axp.plot(mu, yv, '-', lw=1.1, color='#5A6570', zorder=4)
    axp.scatter(mu, yv, s=8, c=cols, edgecolors='white', linewidths=0.3,
                zorder=5, clip_on=False)

    def ptext(latd, xx, s, col):
        axp.text(xx, ytr(latd)[0], s, fontsize=4.6, color=col, ha='center',
                 va='center', zorder=6, style='italic')
    ptext(40, 0.18, 'poleward', BLUE)
    ptext(-33, 0.18, 'poleward', BLUE)
    ptext(84, -0.22, 'equatorward', RED)
    ptext(-84, -0.22, 'equatorward', RED)

    axp.set_yticks([]); axp.spines[['top', 'right', 'left']].set_visible(False)
    axp.tick_params(axis='x', labelsize=5, length=2, pad=1.5)
    axp.set_xticks([-0.2, 0.0, 0.2])
    axp.set_xticklabels(['−0.2', '0', '0.2'])
    axp.set_xlabel('Poleward\ncomponent', fontsize=5.5, labelpad=1.5,
                   linespacing=1.1, color=INK2)

    # ---- vertical diverging colorbar at the right margin ----
    cby, cbh = MAP_Y + 11.0, MAP_H - 20.0
    cax = add_panel(CB_X, cby, 2.4, cbh)
    cb = fig.colorbar(cm.ScalarMappable(norm=Normalize(-VMAX, VMAX), cmap=CMAP),
                      cax=cax, orientation='vertical')
    cb.set_ticks([-0.2, -0.1, 0, 0.1, 0.2])
    cb.set_ticklabels(['−0.2', '−0.1', '0', '0.1', '0.2'])
    cax.tick_params(labelsize=5, width=0.4, length=1.6, pad=1.2)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor('#8A8A8A')
    ftext(CB_X + 1.2, cby - 8.8, 'Mean poleward\ncomponent', ha='center',
          va='top', fontsize=5.5, color=INK, linespacing=1.25)
    ftext(CB_X + 1.2, cby - 2.2, 'poleward', ha='center', va='center',
          fontsize=5.5, color=BLUE, style='italic')
    ftext(CB_X + 1.2, cby + cbh + 2.2, 'equatorward', ha='center', va='center',
          fontsize=5.5, color=RED, style='italic')

    # ---- method note (two lines, kept within the map width) ----
    ftext(MAP_X, MAP_Y + MAP_H + 3.2,
          'High-quality subset: incidence 35–75°, pixel scale ≤ 1.0 m, '
          f'non-summed NAC frames, detection confidence ≥ 0.571 ($n$ = {N_HQ:,}).',
          ha='left', va='center', fontsize=5, color='#707070')
    ftext(MAP_X, MAP_Y + MAP_H + 5.6,
          'Sparse rows poleward of 66° use relaxed thresholds (flagged in '
          'source data).  † equatorial anomaly (~120–165° W); cf. 0–2° in Fig. 3b.',
          ha='left', va='center', fontsize=5, color='#707070')


# ================================================================== panel b
# module artwork verbatim from v1/v2
def _terrain_m1(ax):
    xs = np.linspace(4, 28, 60)
    ys = 5 + 10 * ((28 - xs) / 24) ** 1.6
    xs = np.append(xs, 44); ys = np.append(ys, 5)
    ax.fill_between(xs, 3, ys, color=TERRAIN_F, lw=0, zorder=1)
    ax.plot(xs, ys, color=TERRAIN_E, lw=0.8, zorder=2, solid_capstyle='round')
    ax.add_patch(Circle((9.2, 13.0), 0.95, facecolor='#6B6B6B',
                        edgecolor='none', zorder=4))
    ax.add_patch(FancyArrowPatch((11.0, 11.4), (25.5, 5.9), arrowstyle='-|>',
                                 mutation_scale=7, lw=1.1, color=INK, zorder=5))
    ax.add_patch(FancyArrowPatch((14.2, 13.4), (28.0, 8.2), arrowstyle='-|>',
                                 mutation_scale=6, lw=0.8, color='#A6ADB4',
                                 linestyle=(0, (3, 2)), zorder=3))
    ax.text(29.0, 9.0, 'steepest\ndescent', fontsize=5, color='#8B98A5',
            ha='left', va='center', linespacing=1.2)


def _terrain_m2(ax):
    ax.fill([5, 24, 43], [4, 14, 4], color=TERRAIN_F, lw=0, zorder=1)
    ax.plot([5, 24, 43], [4, 14, 4], color=TERRAIN_E, lw=0.8, zorder=2,
            solid_joinstyle='round')
    ax.plot([3, 45], [4, 4], color='#D8D8D8', lw=0.5, zorder=1)

    def events(ts, side):
        for t in ts:
            fx = 5 if side < 0 else 43
            px, py = 24 + t * (fx - 24), 14 + t * (4 - 14)
            dx, dy = fx - 24.0, 4.0 - 14.0
            n = np.hypot(dx, dy); dx, dy = dx / n, dy / n
            ox, oy = -dy, dx
            if oy < 0: ox, oy = -ox, -oy
            px, py = px + ox * 0.9, py + oy * 0.9
            ax.add_patch(Circle((px, py), 0.55, facecolor='#4F4F4F',
                                edgecolor='none', zorder=4))
            ax.add_patch(FancyArrowPatch((px + dx * 0.9, py + dy * 0.9),
                                         (px + dx * 3.1, py + dy * 3.1),
                                         arrowstyle='-|>', mutation_scale=4.5,
                                         lw=0.7, color='#4F4F4F', zorder=4))
    events([0.18, 0.34, 0.50, 0.66, 0.82], -1)
    events([0.30, 0.62], +1)
    ax.text(9.0, 12.6, 'more', fontsize=5, color='#7C7C7C', ha='center',
            va='center', style='italic')
    ax.text(39.0, 12.6, 'fewer', fontsize=5, color='#7C7C7C', ha='center',
            va='center', style='italic')


def _disk_m3(ax):
    cx, cy, r = 24.0, 9.6, 8.6
    disk = Circle((cx, cy), r, transform=ax.transData)
    ax.add_patch(Circle((cx, cy), r, facecolor=NEUTRAL, edgecolor='none',
                        zorder=1))

    def band(f0, f1, color, hatch=None):
        for s in (+1, -1):
            yy0, yy1 = sorted((cy + s * f0 * r, cy + s * f1 * r))
            rect = Rectangle((cx - r, yy0), 2 * r, yy1 - yy0,
                             facecolor=color if not hatch else 'white',
                             edgecolor=HATCH_EC if hatch else 'none',
                             lw=0, hatch=hatch, zorder=2)
            rect.set_clip_path(disk); ax.add_patch(rect)
    band(0.28, 0.62, B_BLUE)
    band(0.62, 0.88, B_RED)
    band(0.88, 1.00, None, hatch='/////')
    ax.plot([cx - r, cx + r], [cy, cy], color='#8B98A5', lw=0.5,
            ls=(0, (4, 2.5)), zorder=3)

    def arrow(px, dyf, up, color, L=2.3):
        y = cy + dyf * r
        ax.add_patch(FancyArrowPatch((px, y - up * L / 2), (px, y + up * L / 2),
                                     arrowstyle='-|>', mutation_scale=5,
                                     lw=0.9, color=color, zorder=4))
    for s in (+1, -1):
        for px in (cx - 5.2, cx, cx + 5.2):
            arrow(px, s * 0.45, s, BLUE)
        for px in (cx - 2.9, cx + 2.9):
            arrow(px, s * 0.75, -s, RED, L=1.9)
    ax.add_patch(Circle((cx, cy), r, facecolor='none', edgecolor='#707070',
                        lw=0.7, zorder=5))


def draw_panel_b(fig, add_panel, ftext, w_mm, h_mm, x0=0.0, y0=0.0):
    """Horizontal local -> global chain along the bottom of the figure."""
    BW, BH = 48.0, 20.0
    BY = y0 + 76.5
    XS = [x0 + 6.0, x0 + 66.0, x0 + 126.0]
    modules = [
        ('Local transport', 'Rockfalls follow local downslope directions',
         _terrain_m1, (-1.0, 19.0)),
        ('Source-slope selection',
         'Differently oriented source slopes\ncontribute unequally',
         _terrain_m2, (-2.0, 18.0)),
        ('Global organization', 'Latitude-organized global mass wasting',
         _disk_m3, (-0.2, 19.8)),
    ]
    for bx, (title, caption, painter, ylim) in zip(XS, modules):
        cxm = bx + BW / 2
        ftext(cxm, BY - 2.2, title, ha='center', va='center',
              fontsize=6.5, fontweight='bold')
        ax = add_panel(bx, BY, BW, BH)
        ax.set_xlim(0, BW); ax.set_ylim(*ylim)
        ax.set_aspect('equal'); ax.axis('off')
        box = FancyBboxPatch((0.012, 0.03), 0.976, 0.94,
                             boxstyle='round,pad=0.008,rounding_size=0.03',
                             transform=ax.transAxes, facecolor='white',
                             edgecolor=BOX_EC, lw=0.6, zorder=0, clip_on=False)
        ax.add_patch(box)
        painter(ax)
        cy_txt = BY + BH + (4.4 if '\n' in caption else 3.0)
        ftext(cxm, cy_txt, caption, ha='center', va='center', fontsize=6,
              linespacing=1.35)

    # chain arrows between modules
    ymid = 1 - (BY + BH / 2) / h_mm
    for xa, xb in ((XS[0] + BW + 1.2, XS[1] - 1.2),
                   (XS[1] + BW + 1.2, XS[2] - 1.2)):
        fig.add_artist(FancyArrowPatch(
            (xa / w_mm, ymid), (xb / w_mm, ymid),
            transform=fig.transFigure, arrowstyle='-|>', mutation_scale=8,
            lw=1.2, color='#555555'))



# ================================================================== assemble
fig, add_panel, ftext = make_fig(W_MM, H_MM)
draw_panel_a(fig, add_panel, ftext)
draw_panel_b(fig, add_panel, ftext, W_MM, H_MM)
ftext(1.5, 4.5, 'a', fontsize=8, fontweight='bold', ha='left', va='center')
ftext(1.5, 71.5, 'b', fontsize=8, fontweight='bold', ha='left', va='center')

fig.savefig(OUTPUT_DIR / 'Figure6.png', dpi=600, facecolor='white')
fig.savefig(OUTPUT_DIR / 'Figure6.svg')
fig.savefig(OUTPUT_DIR / 'Figure6.tiff', dpi=600, facecolor='white',
            pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig(OUTPUT_DIR / 'Figure6_preview.png', dpi=200, facecolor='white')
try:
    fig.savefig(OUTPUT_DIR / 'Figure6.pdf')
except PermissionError:
    print('WARNING: Figure6.pdf is locked (close the PDF viewer and rerun)')
plt.close(fig)

fa, ap_a, ft_a = make_fig(180, 72)
draw_panel_a(fa, ap_a, ft_a)
ft_a(1.5, 4.5, 'a', fontsize=8, fontweight='bold', ha='left', va='center')
fa.savefig(OUTPUT_DIR / 'Figure6a.png', dpi=300, facecolor='white')
plt.close(fa)

fb, ap_b, ft_b = make_fig(180, 42)
draw_panel_b(fb, ap_b, ft_b, 180, 42, y0=-70.0)
ft_b(1.5, 3.0, 'b', fontsize=8, fontweight='bold', ha='left', va='center')
fb.savefig(OUTPUT_DIR / 'Figure6b.png', dpi=300, facecolor='white')
plt.close(fb)
print('done')
