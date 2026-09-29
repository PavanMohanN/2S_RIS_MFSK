r"""
Fig. 8 -- Velocity RMSE versus SNR
==================================
Output : paper/figures/F8rev_rmse_velocity.pdf   (\includegraphics{F8rev_rmse_velocity} in the manuscript)
Data   : results/figdata/F8rev_rmse_velocity.json + .npz, written by experiments/exp02_estimation_crb.py
Run    : python figures/fig08_rmse_velocity.py   (re-plots from the saved data;
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


@figio.paper_figure('F8rev_rmse_velocity')
def fig_rmse_velocity(out, sw, op_db):
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 2.2))
    s = sw['snr_db']
    dv = mc.LAMBDA / (2 * mc.TCPI)
    ax.semilogy(s, sw['crb_v'], '--', color=mc.COLORS['crb'], lw=1.4, zorder=5, label='CRB (exact FIM)')
    ax.semilogy(s, sw['v_fft'], 's-', color=mc.COLORS['coarse'], mfc='none', label='$N_{\\rm sw}$-point FFT, argmax')
    ax.semilogy(s, sw['v_ml'], 'o', color=mc.COLORS['refined'], ms=3.2, zorder=4, label='FFT + ML refinement')
    ax.axhline(dv, color='0.35', ls=':', lw=0.8)
    ax.axhline(dv / np.sqrt(12), color='0.55', ls='-.', lw=0.7)
    ax.text(s[0] - 2.25, dv * 1.2, '$\\Delta v=\\lambda/(2T_{\\rm CPI})$', fontsize=6)
    ax.text(s[0] - 2.25, dv / np.sqrt(12) * 1.2, '$\\Delta v/\\sqrt{12}$', fontsize=6)
    ax.axvline(op_db, color='#e08e0b', ls=':', lw=1.0)
    ax.set_xlabel('Single-symbol SNR (dB)')
    ax.set_ylabel('Velocity RMSE (m/s)')
    ax.set_ylim(1e-4, 30)
    ax.grid(True, which='both')
    ax.legend(fontsize=5.8, loc='upper right')
    return mc.save_fig(fig, out, 'F8rev_rmse_velocity')


if __name__ == '__main__':
    print(figio.replot(fig_rmse_velocity))
