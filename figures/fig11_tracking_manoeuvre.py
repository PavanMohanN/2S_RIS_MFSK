r"""
Fig. 11 -- Tracking under manoeuvre
===================================
Output : paper/figures/FG_tracking_manoeuvre.pdf   (\includegraphics{FG_tracking_manoeuvre} in the manuscript)
Data   : results/figdata/FG_tracking_manoeuvre.json + .npz, written by experiments/exp05_tracking.py
Run    : python figures/fig11_tracking_manoeuvre.py   (re-plots from the saved data;
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


T = mc.TCPI


NAMES = ['alpha-smoother (submitted)', 'alpha-beta', 'CV-EKF', 'CA-EKF', 'IMM (CV+CA)']


SHORT = ['T0', 'T1', 'T2', 'T3', 'T4']


CL = {'raw': '#bbbbbb', 'T0': '#d95f02', 'T1': '#7b3294', 'T2': '#1b9e77', 'T3': '#e7298a', 'T4': '#1f5fa8'}


@figio.paper_figure('FG_tracking_manoeuvre')
def fig_manoeuvre(out, ex, sweep):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(mc.IEEE_COL_W, 4.6), gridspec_kw={'hspace': 0.5})
    P, res = ex['P'], ex['res']
    t = np.arange(P.shape[0]) * T
    for k in ('raw', 'T0', 'T2', 'T4'):
        e = np.sqrt(np.mean(np.sum((res['est'][k] - P[:, None, :]) ** 2, axis=2), axis=1))
        lab = 'per-CPI fix' if k == 'raw' else NAMES[SHORT.index(k)]
        a1.plot(t, e * 100, color=CL[k], lw=1.0 if k != 'raw' else 0.7, label=lab)
    for tr in ex['reversals']:
        a1.axvline(tr, color='0.6', lw=0.5, ls=':')
    a1.set_xlabel('Time (s)')
    a1.set_ylabel('Position error, rms (cm)')
    a1.set_ylim(0, None)
    a1.set_xlim(0, t[-1])
    a1.legend(fontsize=5.6, loc='upper left', ncol=2)
    a1.grid(True)
    a1.set_title('(a) %s: error versus time (dotted: turn reversals)' % ex['label'], fontsize=7.5)
    for fam, ls in (('turn', '-'), ('acc', '--')):
        pts = [s for s in sweep if s['family'] == fam]
        a = [s['accel'] for s in pts]
        for k in ('raw', 'T0', 'T1', 'T2', 'T3', 'T4'):
            lab = None if fam == 'acc' else ('per-CPI fix' if k == 'raw' else NAMES[SHORT.index(k)])
            a2.semilogy(a, [s['m'][k]['pos_rmse'] * 100 for s in pts], ls, color=CL[k], marker='o', ms=2.5, label=lab)
    a2.set_xlabel('Manoeuvre acceleration (m/s$^2$)')
    a2.set_ylabel('Position RMSE (cm)')
    a2.grid(True, which='both')
    a2.set_ylim(0.6, 40)
    a2.legend(fontsize=5.4, loc='upper left', ncol=2)
    a2.set_title('(b) Solid: coordinated turns; dashed: along-track acceleration', fontsize=7.5)
    return mc.save_fig(fig, out, 'FG_tracking_manoeuvre')


if __name__ == '__main__':
    print(figio.replot(fig_manoeuvre))
