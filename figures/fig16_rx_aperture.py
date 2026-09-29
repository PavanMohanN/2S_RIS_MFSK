r"""
Fig. 16 -- Receive-aperture scaling
===================================
Output : paper/figures/FI_rx_aperture.pdf   (\includegraphics{FI_rx_aperture} in the manuscript)
Data   : results/figdata/FI_rx_aperture.json + .npz, written by experiments/exp06_rx_aperture.py
Run    : python figures/fig16_rx_aperture.py   (re-plots from the saved data;
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


N_REF = 16


@figio.paper_figure('FI_rx_aperture')
def fig_aperture(out, rows, n_star, r3):
    n = np.array([r['n'] for r in rows])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(mc.IEEE_COL_W, 4.4), gridspec_kw={'hspace': 0.5})
    a1.loglog(n, [r['crb_pos'] * 100 for r in rows], '-', color=mc.COLORS['crb'], lw=1.3, label='position CRB')
    a1.loglog(n, [r['crb_range_part'] * 100 for r in rows], '--', color='0.35', lw=0.9, label='range contribution')
    a1.loglog(n, [r['crb_angle_part'] * 100 for r in rows], ':', color='0.35', lw=1.1, label='angular contribution')
    a1.loglog(n, [r['pos_nf'] * 100 for r in rows], 'o', color=mc.COLORS['refined'], ms=4, label='MC, range-focused scan')
    a1.loglog(n, [r['pos_ff'] * 100 for r in rows], 's', color=mc.COLORS['coarse'], mfc='none', ms=4, label='MC, far-field scan')
    a1.axvline(N_REF, color='0.6', lw=0.6)
    a1.text(N_REF * 1.06, 0.97, 'current \n work', fontsize=6, color='0.25', transform=a1.get_xaxis_transform(), va='top')
    a1.axvline(n_star, color='#2e7d32', lw=0.6, ls='--')
    a1.text(n_star * 1.06, 0.08, 'balance\n$N^*\\approx%d$' % round(n_star), fontsize=6, color='#2e7d32',
            transform=a1.get_xaxis_transform())
    a1.set_xticks(n)
    a1.set_xticklabels([str(v) for v in n])
    a1.set_xlabel('Receive elements $N_{\\rm rx}$')
    a1.set_ylabel('Position RMSE (cm)')
    a1.grid(True, which='both')
    a1.legend(fontsize=5.6, loc='lower left')
    rf = rows[-1]
    a1.annotate('(%.1f dB defocus)' % rf['defocus_db'], (rf['n'], rf['pos_ff'] * 100),
                xytext=(55, 12), fontsize=6, color=mc.COLORS['coarse'],)
          #      arrowprops=dict(arrowstyle='->', lw=0.6, color=mc.COLORS['coarse']))
    a1.set_title('(a) Position accuracy versus receive aperture ($\\beta=60^\\circ$)', fontsize=7.5)

    a2.semilogx(n, [r['defocus_db'] for r in rows], 'o-', color=mc.COLORS['coarse'], ms=3.5, label='far-field defocus loss')
    a2.set_xticks(n)
    a2.set_xticklabels([str(v) for v in n])
    a2.set_xlabel('Receive elements $N_{\\rm rx}$')
    a2.set_ylabel('Defocus loss at %g m (dB)' % r3)
    a2.grid(True, which='both')
    b2 = a2.twinx()
    b2.loglog(n, [r['fraunhofer_m'] for r in rows], 'd--', color='0.4', ms=3, lw=0.8, label='Fraunhofer distance $2D^2/\\lambda$')
    b2.axhline(r3, color='0.4', lw=0.6, ls=':')
    b2.text(n[0] * 1.1, r3 * 1.3, 'target range %g m' % r3, fontsize=6, color='0.15')
    b2.set_ylabel('$2D^2/\\lambda$ (m)', fontsize=7, color='0.35')
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, fontsize=5.6, loc='upper left')
    a2.set_title('(b) Wavefront curvature of large receive arrays', fontsize=7.5)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    a2.xaxis.set_major_locator(FixedLocator(n))
    a2.xaxis.set_major_formatter(FixedFormatter([str(v) for v in n]))
    a2.xaxis.set_minor_locator(NullLocator())
    return mc.save_fig(fig, out, 'FI_rx_aperture')


if __name__ == '__main__':
    print(figio.replot(fig_aperture))
