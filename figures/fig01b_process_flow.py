r"""
Fig. 1b -- Signal and control flow
==================================
Output : paper/figures/F1b_process_flow.pdf   (\includegraphics{F1b_process_flow} in the manuscript)
Data   : computed by this script
Run    : python figures/fig01b_process_flow.py
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


def fig_flow(out):
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 2.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    def box(x, y, w, h, txt, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.12',
                                    fc=fc, ec='0.25', lw=0.7))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=5.8)

    def arr(p, q, txt=None, dy=0.18, ls='-'):
        ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='->', lw=0.8, ls=ls))
        if txt:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, txt, ha='center', fontsize=5.3)
    box(0.1, 4.4, 2.1, 1.4, 'CW source\n28 GHz, 20 dBm', '#eeeeee')
    box(3.0, 4.4, 2.8, 1.4, 'T-RIS\nPIN modulation at $f_{m,k}$\none DDS, all elements', '#e8dff5')
    box(6.6, 4.4, 3.3, 1.4, 'R-RIS\n3-bit gradient $\\psi_n$\nupdated once per CPI', '#dbe8f7')
    box(6.9, 2.35, 2.7, 1.1, 'target', '#fde7d6')
    box(0.1, 0.2, 2.6, 1.6, 'Rx ULA, 16 el.\nI/Q down-conversion,\nper-symbol correlator', '#e2f1e4')
    box(3.4, 0.2, 2.6, 1.6, 'coherent 2-D processing\npath, Doppler, angle\n(peak refinement)', '#e2f1e4')
    box(6.7, 0.2, 3.2, 1.6, 'bistatic inversion\nIMM tracker\n$\\theta_{\\rm pred}$', '#e2f1e4')
    arr((2.2, 5.1), (3.0, 5.1))
    arr((5.8, 5.1), (6.6, 5.1), 'near field', 0.18)
    arr((8.25, 4.4), (8.25, 3.45), 'M-FSK beam', 0.0)
    ax.text(8.35, 3.9, '', fontsize=5)
    arr((6.9, 2.9), (1.4, 1.8), 'echo', 0.25)
    arr((2.7, 1.0), (3.4, 1.0))
    arr((6.0, 1.0), (6.7, 1.0))
    ax.annotate('', xy=(9.85, 4.4), xytext=(9.85, 1.8), arrowprops=dict(arrowstyle='->', lw=0.8, ls='--', color='#1f5fa8'))
    ax.text(9.6, 3.1, 'beam\ncommand\n(SPI)', fontsize=5.1, color='#1f5fa8', ha='right')
    return mc.save_fig(fig, out, 'F1b_process_flow')


if __name__ == '__main__':
    from mfsk import common as mc
    mc.set_ieee_style()
    print(fig_flow(FIGDIR))
