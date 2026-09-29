r"""
Fig. 12 -- M-FSK versus FMCW, three-target scene
================================================
Output : paper/figures/F11rev_fmcw_fair.pdf   (\includegraphics{F11rev_fmcw_fair} in the manuscript)
Data   : results/figdata/F11rev_fmcw_fair.json + .npz, written by experiments/exp07_benchmarks.py
Run    : python figures/fig12_fmcw_fair.py   (re-plots from the saved data;
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


TARGETS_L = np.array([4.0, 7.0, 10.0])


TARGETS_LD = np.array([0.0, 3.0, -2.0])


@figio.paper_figure('F11rev_fmcw_fair')
def fig_fmcw(out, a, b, c):
    fig, axs = plt.subplots(1, 3, figsize=(mc.IEEE_DBL_W, 2.0), gridspec_kw={'wspace': 0.28})
    cols = ['#1f5fa8', '#c0392b', '#2e7d32']
    titles = ['(a) M-FSK, 2-D processing', '(b) FMCW, one CPI-long chirp', '(c) Chirp-sequence FMCW, 2-D FFT']
    for ax, r, t in zip(axs, (a, b, c), titles):
        ax.plot(r['Laxis'], 10 * np.log10(r['prof'] + 1e-6), color='0.3', lw=0.8)
        for L, col in zip(TARGETS_L, cols):
            ax.axvline(L, color=col, ls='--', lw=0.9)
        dets = [d[0] if isinstance(d, tuple) else d for d in r['det']]
        for d in dets:
            ax.plot(d, 1.5, 'v', color='k', ms=4)
        ax.set_xlim(0, 13 if r is not b else 16)
        ax.set_ylim(-40, 5)
        ax.set_xlabel('Bistatic path (m)')
        ax.set_title(t, fontsize=7)
        ax.grid(True)
    axs[0].set_ylabel('Normalized power (dB)')
    """
    for (L, Ld, col), yy in zip(zip(TARGETS_L, TARGETS_LD, cols), (-30, -30, -35)):
        if Ld != 0:
            axs[1].annotate('', xy=(L + Ld * mc.FC * mc.TCPI / (mc.M_TONES * mc.DELTA_F), yy), xytext=(L, yy),
                            arrowprops=dict(arrowstyle='->', color=col, lw=0.9))
                            """
    #axs[1].text(0.3, -39, 'dashed: true paths; arrows: coupling shift', fontsize=5.5)
    return mc.save_fig(fig, out, 'F11rev_fmcw_fair')


if __name__ == '__main__':
    print(figio.replot(fig_fmcw))
