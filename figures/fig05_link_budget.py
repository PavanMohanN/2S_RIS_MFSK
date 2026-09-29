r"""
Fig. 5 -- Link budget of Eq. (17)
=================================
Output : paper/figures/F5rev_link_budget.pdf   (\includegraphics{F5rev_link_budget} in the manuscript)
Data   : computed by this script (reads results of experiments/exp04_hardware_coupling.py)
Run    : python figures/fig05_link_budget.py
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
import re
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arc
from mfsk import common as mc
from mfsk import figio

FIGDIR = figio.FIGDIR


KB, T0 = 1.380649e-23, 290.0


NF_DB, GRX_DBI, PTX_DBM = 5.0, 18.0, 20.0


def link_budget(c0sq_db):
    lam = mc.LAMBDA
    A = mc.N_ELEM * mc.D_ELEM ** 2
    g_rris = 4 * np.pi * A * np.cos(np.deg2rad(mc.THETA1_DEG)) / lam ** 2
    tsb2 = abs(mc.t_sb(mc.FC)) ** 2 / 2
    rx_ap = 10 ** (GRX_DBI / 10) * lam ** 2 / (4 * np.pi)
    stages = [
        ('a', PTX_DBM),
        ('b', 10 * np.log10(A / (4 * np.pi * mc.R1_TX_RIS ** 2))),
        ('c', 10 * np.log10(tsb2)),
        ('d', c0sq_db),
        ('e', 20 * np.log10(mc.GAMMA_MAG)),
        ('f', 10 * np.log10(g_rris)),
        ('g', -10 * np.log10(4 * np.pi * mc.R2_RIS_T ** 2)),
        ('h', 0.0),
        ('i', -10 * np.log10(4 * np.pi * mc.R3_T_RX ** 2)),
        ('j', 10 * np.log10(rx_ap)),
    ]
    prx = sum(v for _, v in stages)
    bn = 1 / mc.TSYM
    noise = 10 * np.log10(KB * T0 * bn * 1e3) + NF_DB
    return dict(stages=stages, prx_dbm=prx, noise_dbm=noise, bn_khz=bn / 1e3, snr1_db=prx - noise,
                margin_db=prx - noise - mc.SNR1_DB, g_rris_dbi=10 * np.log10(g_rris), c0sq_db=c0sq_db,
                capture_db=stages[1][1], rx_ap_db=stages[-1][1])


def fig_link_budget(out, lb):
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 2.9))
    lev = np.cumsum([v for _, v in lb['stages']])
    x = np.arange(len(lev))
    for i, (lab, v) in enumerate(lb['stages']):
        lo = lev[i] - v if i else 0
        col = '#1f5fa8' if v >= 0 else '#c0392b'
        if i == 0:
            ax.bar(i, lev[0] + 130, bottom=-130, color='0.55', width=0.62)
        else:
            ax.bar(i, v, bottom=lo, color=col, width=0.62)
        ax.text(i, max(lev[i], lo) + 2.5, f'{v:+.1f}', ha='center', fontsize=6.5)
    ax.plot(x, lev, 'k.-', lw=0.8, ms=3)
    ax.axhline(lb['noise_dbm'], color='#2e7d32', ls='--', lw=1.0)
    ax.text(len(x) - 3, lb['noise_dbm'] + 1.5, 'noise floor, $1/T_{\\rm sym}$: %.1f dBm' % lb['noise_dbm'],
            ha='right', fontsize=7, color='#2e7d32')
    ax.annotate('', xy=(len(x) - 1.25, lb['prx_dbm']), xytext=(len(x) - 1.25, lb['noise_dbm']),
                arrowprops=dict(arrowstyle='<->', lw=0.7))
    ax.text(len(x) - 1.75, (lb['prx_dbm'] + lb['noise_dbm']) / 2 - 1,
            'SNR$_1$ = %.1f dB' % lb['snr1_db'], ha='right', fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in lb['stages']], ha='right', fontsize=6.8)
    ax.set_ylabel('Level (dBm; dBm/m$^2$ after spreading)', fontsize=6.8)
    ax.set_ylim(-130, 32)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.grid(True, axis='y')
    fig.tight_layout()
    return mc.save_fig(fig, out, 'F5rev_link_budget')


if __name__ == '__main__':
    import json
    res = ROOT / 'results' / 'exp04_hardware_coupling' / 'data' / 'exp04_results.json'
    if not res.exists():
        raise SystemExit('run experiments/exp04_hardware_coupling.py first (coupling C0^2)')
    c0sq = json.loads(res.read_text())['coupling']['nominal_coupling_db']['0.5000']
    mc.set_ieee_style()
    lb = link_budget(c0sq)
    print('P_rx %.1f dBm, noise %.1f dBm, SNR1 %.1f dB, margin to %.1f dB: %.1f dB' % (lb['prx_dbm'], lb['noise_dbm'], lb['snr1_db'], mc.SNR1_DB, lb['margin_db']))
    print(fig_link_budget(FIGDIR, lb))
