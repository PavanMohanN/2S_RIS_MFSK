r"""
Fig. 2 -- Multi-tone beams and validity region
==============================================
Output : paper/figures/F4rev_multitone_beam.pdf   (\includegraphics{F4rev_multitone_beam} in the manuscript)
Data   : results/figdata/F4rev_multitone_beam.json + .npz, written by experiments/exp03_beam_squint.py
Run    : python figures/fig02_multitone_beam.py   (re-plots from the saved data;
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


def tone_gains(theta1_deg, nu, n=mc.N_X, f_design=mc.FC, bits=None):
    """Complex normalised gain of each tone toward the target."""
    th = np.deg2rad(theta1_deg)
    return mc.af_line(np.array([th]), mc.FC + nu, th, n=n, f_design=f_design, bits=bits)[:, 0]


@figio.paper_figure('F4rev_multitone_beam')
def fig_multitone(out, nu, theta1, bf, nx, L, op_bf, rule_k):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(mc.IEEE_COL_W, 4.5), gridspec_kw={'hspace': 0.5})
    th = np.deg2rad(np.linspace(theta1 - 16, theta1 + 16, 2001))
    cmap = plt.get_cmap('viridis')
    ref = np.abs(mc.af_line(np.array([np.deg2rad(theta1)]), mc.FC, np.deg2rad(theta1)))[0, 0]
    for i, v in enumerate(nu):
        a = np.abs(mc.af_line(th, mc.FC + v, np.deg2rad(theta1)))[0] / ref
        a1.plot(np.degrees(th), 20 * np.log10(a), color=cmap(i / (nu.size - 1)), lw=0.8)
    a1.axvline(theta1, color='k', lw=0.7, ls='--')
    a1.axhline(-3, color='0.4', lw=0.6, ls=':')
    worst = -20 * np.log10(np.abs(tone_gains(theta1, nu[-1:]))[0])
    a1.text(theta1 + 0.6, -1.6, 'target', fontsize=6)
    a1.text(theta1 - 15.5, -3.9, '$-3$ dB', fontsize=6, color='0.35')
    a1.text(theta1 - 15.5, -8.3, 'worst tone (%d MHz):\nsquint %.2f$^\\circ$, loss %.2f dB at target'
            % (nu[-1] / 1e6, mc.squint_exact_deg(nu[-1], theta1), worst), fontsize=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(nu[0] / 1e6, nu[-1] / 1e6))
    cb = fig.colorbar(sm, ax=a1, pad=0.02)
    cb.set_label('$f_{m,k}$ (MHz)', fontsize=7)
    a1.set_xlim(theta1 - 16, theta1 + 16)
    a1.set_ylim(-10, 0.8)
    a1.set_xlabel('Azimuth (deg)')
    a1.set_ylabel('Array factor (dB)')
    a1.grid(True)
    a1.set_title('(a) All %d tones, gradient designed at $f_c$, $\\theta_1=%d^\\circ$' % (nu.size, theta1), fontsize=7.5)

    X, Y = np.meshgrid(bf * 100, nx)
    cs = a2.contourf(X, Y, np.clip(L, 1e-3, 30), levels=[0, 0.1, 0.25, 0.5, 1, 2, 3, 6, 30],
                     cmap='magma_r')
    cl = a2.contour(X, Y, L, levels=[0.1, 1.0, 3.0], colors='k', linewidths=0.6)
    a2.clabel(cl, fmt={0.1: '0.1 dB', 1.0: '1 dB', 3.0: '3 dB'}, fontsize=5.5)
    cb2 = fig.colorbar(cs, ax=a2, pad=0.02, ticks=[0, 0.1, 0.25, 0.5, 1, 2, 3, 6])
    cb2.ax.set_yticklabels(['0', '0.1', '0.25', '0.5', '1', '2', '3', '6+'], fontsize=6)
    cb2.set_label('worst-tone loss (dB)', fontsize=7)
    a2.plot(bf * 100, rule_k / (bf * np.sin(np.deg2rad(theta1))), 'w--', lw=1.0)
    xl = 0.72                                   # place the label on the dashed rule line
    yl = rule_k / (xl / 100 * np.sin(np.deg2rad(theta1)))
    a2.text(xl, yl * 0.80, '$N_xB_f\\sin\\theta_1=%.2f$' % rule_k, fontsize=6, color='k',
            rotation=-37, rotation_mode='anchor', ha='center', va='top')
    a2.plot(op_bf * 100, mc.N_X, 'o', mfc='#1f5fa8', mec='w', ms=5)
    a2.annotate('this work\n(carrier design)', (op_bf * 100, mc.N_X), xytext=(1.6, 5.0),
                fontsize=6, color='k', arrowprops=dict(arrowstyle='->', lw=0.6))
    a2.plot(op_bf * 100 / 2, mc.N_X, 's', mfc='#2e7d32', mec='w', ms=4.5)
    a2.annotate('centre-frequency\ndesign', (op_bf * 100 / 2, mc.N_X), xytext=(0.105, 5.0),
                fontsize=6, color='k', arrowprops=dict(arrowstyle='->', lw=0.6))
    a2.set_ylim(4, 128)
    a2.set_xscale('log')
    a2.set_yscale('log', base=2)
    a2.set_yticks([4, 8, 16, 32, 64, 128])
    a2.set_yticklabels(['4', '8', '16', '32', '64', '128'])
    a2.set_xlabel('Fractional bandwidth $B_f = f_{m,M}/f_c$ (%)')
    a2.set_ylabel('R-RIS columns $N_x$')
    a2.set_title('(b) Validity region at $\\theta_1=%d^\\circ$' % theta1, fontsize=7.5)
    return mc.save_fig(fig, out, 'F4rev_multitone_beam')


if __name__ == '__main__':
    print(figio.replot(fig_multitone))
