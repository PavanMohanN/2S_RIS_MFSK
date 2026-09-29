r"""
Fig. 7 -- Path and position RMSE versus SNR
===========================================
Output : paper/figures/F7rev_rmse_path_position.pdf   (\includegraphics{F7rev_rmse_path_position} in the manuscript)
Data   : results/figdata/F7rev_rmse_path_position.json + .npz, written by experiments/exp02_estimation_crb.py
Run    : python figures/fig07_rmse_path_position.py   (re-plots from the saved data;
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


@figio.paper_figure('F7rev_rmse_path_position')
def fig_rmse_path_position(out, sw, op_db, beta):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(mc.IEEE_COL_W, 4.3), gridspec_kw={'hspace': 0.55})
    s = sw['snr_db']
    a1.semilogy(s, sw['crb_L'], '--', color=mc.COLORS['crb'], lw=1.4, zorder=5, label='CRB')
    a1.semilogy(s, sw['L_argmax'], 's-', color=mc.COLORS['coarse'], mfc='none',
                label='coherent IFFT, argmax ($N_{\\rm FFT}=8M$)')
    a1.semilogy(s, sw['L_ml'], 'o', color=mc.COLORS['refined'], ms=3.2, zorder=4, label='coherent IFFT + ML refinement')
    a1.axhline(mc.path_resolution(), color='0.35', ls=':', lw=0.8)
    a1.text(-22.5, mc.path_resolution() * 1.3, '$c/(M\\Delta f)$', fontsize=6)
    a2.semilogy(s, sw['crb_pos'], '--', color=mc.COLORS['crb'], lw=1.4, zorder=5, label='CRB (bistatic Jacobian)')
    a2.semilogy(s, sw['pos_coarse'], 's-', color=mc.COLORS['coarse'], mfc='none', label='argmax range + 0.5$^\\circ$ beam scan')
    a2.semilogy(s, sw['pos_ml'], 'o', color=mc.COLORS['refined'], ms=3.2, zorder=4, label='refined range + refined angle')
    for ax, t in ((a1, '(a) Bistatic path'), (a2, f'(b) 2-D position ($\\beta={beta:.0f}^\\circ$)')):
        ax.axvline(op_db, color="#e00b0b", ls=':', lw=1.0)
        ax.text(op_db + 0.6, 0.06, 'SNR$_1$', color='#e00b0b', fontsize=6,
                transform=ax.get_xaxis_transform())
        ax.set_xlabel('Single-symbol SNR (dB)')
        ax.set_ylabel('RMSE (m)')
        ax.grid(True, which='both')
        ax.legend(fontsize=5.8, loc='upper right')
        ax.set_title(t, fontsize=7.5)
    a1.set_ylim(1e-4, 20)
    a2.set_ylim(2e-4, 50)
    return mc.save_fig(fig, out, 'F7rev_rmse_path_position')


if __name__ == '__main__':
    print(figio.replot(fig_rmse_path_position))
