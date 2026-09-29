r"""
Fig. 9 -- Estimator efficiency and zero-padding
===============================================
Output : paper/figures/FA_estimator_efficiency.pdf   (\includegraphics{FA_estimator_efficiency} in the manuscript)
Data   : results/figdata/FA_estimator_efficiency.json + .npz, written by experiments/exp02_estimation_crb.py
Run    : python figures/fig09_estimator_efficiency.py   (re-plots from the saved data;
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
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from mfsk import common as mc
from mfsk import figio

FIGDIR = figio.FIGDIR


NFFT_R_FACTOR = 8


def _ladder_axes(ax, rows, labels, title):
    from matplotlib.patches import Patch
    y = np.arange(len(rows))[::-1]
    cb, cv, co = '#c0392b', '#1f5fa8', '#9a9a9a'
    for yi, r in zip(y, rows):
        tot = r['ratio']
        fr = np.array([r['mse_bias'], r['mse_var'], r['mse_out']]) / max(r['mse'], 1e-30)
        left = 0.0
        for f_, c_ in zip(fr, (cb, cv, co)):
            ax.barh(yi, tot * f_, left=left, color=c_, height=0.62, edgecolor='w', lw=0.3)
            left += tot * f_
        ax.text(tot + 0.08, yi, '%.2f cm (%.2f$\\times$)' % (r['rmse'] * 100, tot), va='center', fontsize=5.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.axvline(1.0, color='k', lw=0.6, ls='--')
    ax.set_xlabel('RMSE / bound')
    ax.set_xlim(0, max(r['ratio'] for r in rows) * 1.62)
    ax.legend(handles=[Patch(color=cb, label='bias$^2$'), Patch(color=cv, label='variance'),
                       Patch(color=co, label='outliers')], fontsize=5.6, loc='center right', ncol=1)
    ax.set_title(title, fontsize=7)
    ax.grid(True, axis='x')


@figio.paper_figure('FA_estimator_efficiency')
def fig_efficiency(out, ladder, zp, crb_op):
    """Paper figure: estimator variants (ladder L2-L6) + zero-padding sweep."""
    rows = ladder[2:]
    labels = [r['paper'].replace('N_sw', '$N_{\\rm sw}$') for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(mc.IEEE_COL_W, 4.2),
                                 gridspec_kw={'hspace': 0.5, 'height_ratios': [1.0, 1.0]})
    _ladder_axes(a1, rows, labels, '(a) Path RMSE / CRB at SNR$_1$ (bar split by MSE share)')
    f = np.array(sorted(zp))
    a2.loglog(f, [zp[k]['argmax'] for k in f], 's-', color=mc.COLORS['coarse'], mfc='none', label='argmax')
    a2.loglog(f, [zp[k]['parabolic'] for k in f], 'o-', color=mc.COLORS['refined'], label='parabolic interpolation')
    a2.axhline(crb_op, color=mc.COLORS['crb'], ls='--', label='CRB')
    a2.axvline(NFFT_R_FACTOR, color='0.5', ls=':', lw=0.7)
    a2.text(NFFT_R_FACTOR / 1.07, 0.2, 'used here\n$N_{\\rm FFT}=8M$', fontsize=6, color='0.35',
            transform=a2.get_xaxis_transform(), va='center', ha='right')
    a2.set_xlabel('IFFT length $N_{\\rm FFT}/M$')
    a2.set_ylabel('Path RMSE at SNR$_1$ (m)')
    a2.grid(True, which='both')
    a2.legend(fontsize=5.8, loc='upper right')
    a2.set_title('(b) Zero-padding versus interpolation (coherent processing)', fontsize=7)
    return mc.save_fig(fig, out, 'FA_estimator_efficiency')


if __name__ == '__main__':
    print(figio.replot(fig_efficiency))
