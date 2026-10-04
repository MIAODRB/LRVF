#!/usr/bin/env python3
"""Reproduce the production Figure 5, including its direction-polarity annotation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts._paths import DATA_DIR, OUTPUT_DIR
import matplotlib
import plot_figure5_base as base

ORIG_B = base.draw_panel_b

def draw_panel_b_v2(ax, distribution, summary):
    ORIG_B(ax, distribution, summary)
    matplotlib.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans']
    ax.annotate('a systematic polarity error\nwould peak near 180°',
                xy=(171.0, 0.0022), xytext=(103.0, 0.0068),
                fontsize=5.4, color='#8A6A6A', ha='center', va='center',
                linespacing=1.25,
                arrowprops=dict(arrowstyle='-', color='#B09090', lw=0.6))

base.draw_panel_b = draw_panel_b_v2


if __name__ == "__main__":
    base.main()
