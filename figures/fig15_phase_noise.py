r"""
Fig. 15 -- Element and oscillator phase noise
=============================================
Output : paper/figures/FF_phase_noise.pdf   (\includegraphics{FF_phase_noise} in the manuscript)
Data   : results/figdata/FF_phase_noise.json + .npz, written by experiments/exp04_hardware_coupling.py
Run    : python figures/fig15_phase_noise.py   (re-plots from the saved data;
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


@figio.paper_figure('FF_phase_noise')
def fig_phase_noise(out, A1, A3):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(mc.IEEE_COL_W, 4.4), gridspec_kw={'hspace': 0.5})
    s = np.array([r['sigma_deg'] for r in A1])
    a1.plot(s, [r['pos_rmse'] * 100 for r in A1], 'o-', color=mc.COLORS['refined'], label='position RMSE')
    a1.plot(s, [r['pos_crb'] * 100 for r in A1], '--', color=mc.COLORS['refined'], lw=0.9, label='position CRB')
    a1.plot(s, [r['path_rmse'] * 100 for r in A1], 's-', color=mc.COLORS['coarse'], mfc='none', label='path RMSE')
    a1.plot(s, [r['path_crb'] * 100 for r in A1], '--', color=mc.COLORS['coarse'], lw=0.9, label='path CRB')
    a1.axvline(A1[0]['q3_equiv_deg'], color='0.5', ls=':', lw=0.8)
    a1.text(A1[0]['q3_equiv_deg'] + 8, 0.16, '3-bit step\nequivalent', fontsize=5.8, color='0.1',
            transform=a1.get_xaxis_transform())
    a1.set_xlabel('Element phase-noise std $\\sigma_\\phi$ (deg)')
    a1.set_ylabel('RMSE (cm)')
    a1.set_ylim(0, 4.4)
    b1 = a1.twinx()
    b1.plot(s, [r['gain_loss_db'] for r in A1], color='0.45', lw=0.9, ls='-.', label='array-gain loss')
    b1.set_ylabel('Array-gain loss (dB)', fontsize=7, color='0.35')
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = b1.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, fontsize=5.6, loc='upper left', ncol=2)
    a1.grid(True)
    a1.set_title('(a) Static element phase noise (3-bit R-RIS)', fontsize=7.5)

    d = np.array([r['dnu'] for r in A3])
    dp = np.where(d > 0, d, d[d > 0].min() / 3)
    base = A3[0]
    for key, lab, c, mk in (('path_rmse', 'path', mc.COLORS['coarse'], 's'),
                            ('vel_rmse', 'velocity', mc.COLORS['green'], '^'),
                            ('pos_rmse', 'position', mc.COLORS['refined'], 'o')):
        a2.loglog(dp, [r[key] / base[key] for r in A3], mk + '-', color=c, ms=3.2, label=lab + ' RMSE')
    a2.axhline(1.0, color='k', lw=0.5)
    a2.set_xlabel('Common-mode oscillator linewidth $\\Delta\\nu$ (Hz)')
    a2.set_ylabel('RMSE / RMSE($\\Delta\\nu$=0)')
    a2.set_ylim(0.8, 80)
    b2 = a2.twinx()
    b2.semilogx(dp, -10 * np.log10([r['eta'] for r in A3]), color='0.2', ls='-.', lw=0.9, label='integration loss')
    b2.plot(dp, [r['eta_mc_db'] for r in A3], 'x', color='0.3', ms=4, label='loss, Monte Carlo')
    b2.set_ylabel('Integration loss (dB)', fontsize=7, color='0.35')
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, fontsize=5.6, loc='upper left')
    a2.grid(True, which='both')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    a2.xaxis.set_major_locator(FixedLocator(dp))
    a2.xaxis.set_major_formatter(FixedFormatter(['0'] + ['%g' % v for v in d[1:]]))
    a2.xaxis.set_minor_locator(NullLocator())
    a2.tick_params(axis='x', labelsize=6)
    a2.set_title('(b) Tx/Rx common-mode phase noise over the CPI', fontsize=7.5)
    return mc.save_fig(fig, out, 'FF_phase_noise')


if __name__ == '__main__':
    print(figio.replot(fig_phase_noise))
