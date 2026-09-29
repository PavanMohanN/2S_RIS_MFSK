r"""
Fig. 10 -- Beam tracking on a straight line
===========================================
Output : paper/figures/F9rev_tracking_cv.pdf   (\includegraphics{F9rev_tracking_cv} in the manuscript)
Data   : results/figdata/F9rev_tracking_cv.json + .npz, written by experiments/exp05_tracking.py
Run    : python figures/fig10_tracking_cv.py   (re-plots from the saved data;
         run the experiment first if the data file is missing)
Edit the plotting code below and re-run this file to regenerate the figure.
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import argparse
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mfsk import common as mc
from mfsk import figio

FIGDIR = figio.FIGDIR


NAMES = ['alpha-smoother (submitted)', 'alpha-beta', 'CV-EKF', 'CA-EKF', 'IMM (CV+CA)']


SHORT = ['T0', 'T1', 'T2', 'T3', 'T4']


CL = {'raw': '#bbbbbb', 'T0': '#d95f02', 'T1': '#7b3294', 'T2': '#1b9e77', 'T3': '#e7298a', 'T4': '#1f5fa8'}


@figio.paper_figure('F9rev_tracking_cv')
def fig_cv(out, P, res, geo):
    fig = plt.figure(figsize=(mc.IEEE_COL_W, 4.4))

    # 2 rows, 2 columns:
    # Row 0 col 0: a1 + colorbar (left)
    # Row 0 col 1: legend (right)
    # Row 1 col 0-1: a2 spanning the full width underneath
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], hspace=0.45, wspace=0.2)
    a1 = fig.add_subplot(gs[0, 0])
    a_leg = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, :])

    # Invisible spanning axes on row 0 to center the title across both a1 and a_leg
    a_title1 = fig.add_subplot(gs[0, :])
    a_title1.axis('off')
    a_title1.set_title(
        '(a) Straight line, 2 m/s at 45$^\\circ$ (grey arrows: R-RIS beam)',
        fontsize=7.5,
        loc='center'
    )

    fx = np.array([res['fixes'][k] for k in sorted(res['fixes'])])
    a1.plot(P[:, 0], P[:, 1], 'k-', lw=1.2, label='true trajectory')
    sc = a1.scatter(fx[:, 0], fx[:, 1], c=np.arange(len(fx)), cmap='cool', s=7, label='per-CPI fixes', zorder=3)
    a1.plot(res['est']['T0'][:, 0, 0], res['est']['T0'][:, 0, 1], '-', color=CL['T0'], lw=0.9, label=NAMES[0])
    a1.plot(res['est']['T4'][:, 0, 0], res['est']['T4'][:, 0, 1], '-', color=CL['T4'], lw=1.0, label=NAMES[4])

    for k in range(0, len(P), 10):
        th = res['th_true'][k]
        a1.annotate('', xy=P[k], xytext=P[k] - 0.18 * np.array([np.sin(th), np.cos(th)]),
                    arrowprops=dict(arrowstyle='->', color='0.5', lw=0.6))

    cb = fig.colorbar(sc, ax=a1, pad=0.04)
    cb.set_label('CPI index', fontsize=7)

    a1.set_aspect('equal')
    a1.set_anchor('W')  # Keeps a1 flush with the left boundary of a2
    a1.set_xlabel('$x$ (m)')
    a1.set_ylabel('$y$ (m)')

    # Display a1's legend in the dedicated right-hand cell
    a_leg.axis('off')
    handles, labels = a1.get_legend_handles_labels()
    a_leg.legend(
        handles,
        labels,
        fontsize=5.6,
        loc='center left',
        borderaxespad=0,
    )

    a1.grid(True)

    # Subplot 2 spans the entire width below
    t = np.arange(len(P))
    a2.plot(t, np.degrees(res['th_true']), 'k-', lw=1.2, label='true $\\theta_1$')
    fb = np.degrees(np.arctan2(fx[:, 0], fx[:, 1]))
    a2.plot(t, fb, '.', color=CL['raw'], ms=3, label='per-CPI fix')
    for k in ('T0', 'T4'):
        a2.plot(t, np.degrees(res['th_true'] + res['perr'][k][:, 0]), '-', color=CL[k], lw=1.0,
                label='command, ' + NAMES[SHORT.index(k)])
    a2.set_xlabel('CPI index')
    a2.set_ylabel('Beam angle (deg)')
    a2.legend(fontsize=5.6, loc='upper right')
    a2.grid(True)
    a2.set_title('(b) Beam commands (one trial)', fontsize=7.5)

    return mc.save_fig(fig, out, 'F9rev_tracking_cv')


if __name__ == '__main__':
    print(figio.replot(fig_cv))