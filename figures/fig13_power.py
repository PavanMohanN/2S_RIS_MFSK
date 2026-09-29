r"""
Fig. 13 -- Transmit-side DC power
=================================
Output : paper/figures/F12rev_power.pdf   (\includegraphics{F12rev_power} in the manuscript)
Data   : results/figdata/F12rev_power.json + .npz, written by experiments/exp07_benchmarks.py
Run    : python figures/fig13_power.py   (re-plots from the saved data;
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


P_BIAS_SUB = 2.0e-3


@figio.paper_figure('F12rev_power')
def fig_power(out, sweep, pts, be_map):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(mc.IEEE_DBL_W, 2.3), gridspec_kw={'wspace': 0.3})
    pch = sweep['p_ch'] * 1e3
    a1.plot(pch, sweep['aesa'], color='#555555', lw=1.2, label='AESA ($N$ T/R channels)')
    styles = [('ris_pin', 'two-stage, PIN R-RIS (2 mW/el.)', '#1f5fa8', '-'),
              ('ris_volt', 'two-stage, voltage-driven R-RIS', '#2e7d32', '--'),
              ('ris_pess', 'two-stage, PIN, $E_{\\rm sw}$=10 pJ', '#c0392b', ':')]
    for key, lab, col, ls in styles:
        a1.axhline(sweep[key], color=col, ls=ls, lw=1.1, label=lab)
    for p, lab in ((5, 'CMOS'), (15, 'SiGe')):
        a1.axvline(p, color='0.6', lw=0.5, ls=':')
        a1.text(p + 0.2, 0.25, lab, fontsize=6, color='0.35')
    a1.plot(sweep['be_pin'] * 1e3, sweep['ris_pin'], 'o', color='#1f5fa8', ms=4)
    a1.annotate('break-even %.1f mW' % (sweep['be_pin'] * 1e3), (sweep['be_pin'] * 1e3, sweep['ris_pin']),
                xytext=(6.5, 0.6), fontsize=6, arrowprops=dict(arrowstyle='->', lw=0.6))
    a1.set_xlabel('AESA power per channel $P_{\\rm ch}$ (mW)')
    a1.set_ylabel('Transmit-side DC power (W)')
    a1.set_xlim(1, 20)
    a1.set_ylim(0, 6)
    a1.grid(True)
    a1.legend(fontsize=5.5, loc='upper left')
    a1.set_title('(a) $N=256$; common bistatic receiver excluded', fontsize=7.5)
    X, Yb = be_map['p_bias'] * 1e3, be_map['p_ch'] * 1e3
    from matplotlib.colors import BoundaryNorm
    lv = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 4, 1e3]
    cmap = plt.get_cmap('RdYlBu', len(lv) - 1)
    cs = a2.contourf(X, Yb, be_map['ratio'], levels=lv, cmap=cmap, norm=BoundaryNorm(lv, cmap.N))
    cl = a2.contour(X, Yb, be_map['ratio'], levels=[1.0], colors='k', linewidths=1.0)
    a2.clabel(cl, fmt={1.0: 'break-even'}, fontsize=6, manual=[(3.0, 5.0)])
    cb = fig.colorbar(cs, ax=a2, pad=0.02, ticks=[0, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 4])
    cb.ax.set_yticklabels(['0', '0.5', '0.75', '1', '1.25', '1.5', '2', '4+'], fontsize=6)
    cb.set_label('$P_{\\rm AESA}/P_{\\rm RIS}$', fontsize=7)
    for p, lab in ((5, 'CMOS 5 mW'), (15, 'SiGe 15 mW')):
        a2.plot(P_BIAS_SUB * 1e3, p, 'ko', ms=3.5)
        a2.text(P_BIAS_SUB * 1e3 + 0.1, p + 0.5, lab, fontsize=6)
    a2.set_xlabel('PIN bias per element and layer $P_{\\rm bias}$ (mW)')
    a2.set_ylabel('$P_{\\rm ch}$ (mW)')
    a2.set_title('(b) Where the two-stage RIS draws less power (PIN R-RIS)', fontsize=7.5)
    return mc.save_fig(fig, out, 'F12rev_power')


if __name__ == '__main__':
    print(figio.replot(fig_power))
