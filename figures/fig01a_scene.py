r"""
Fig. 1a -- Bistatic geometry
============================
Output : paper/figures/F1a_scene.pdf   (\includegraphics{F1a_scene} in the manuscript)
Data   : computed by this script
Run    : python figures/fig01a_scene.py
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


def fig_scene(out):
    p_rx = mc.rx_position(mc.BETA_DEFAULT_DEG)
    pt, ptx, pr = mc.P_TARGET, np.array([0.0, -mc.R1_TX_RIS]), mc.P_RIS
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 3.3))
    ax.plot(*zip(ptx, pr), '--', color='0.45', lw=1.0)
    ax.plot(*zip(pr, pt), '-', color='#d95f02', lw=1.8)
    ax.plot(*zip(pt, p_rx), '-', color='#2e7d32', lw=1.3)
    ax.plot(*zip(ptx, p_rx), ':', color='0.65', lw=0.8)
    ax.plot(*ptx, 's', color='0.2', ms=7)
    ax.text(ptx[0] + 0.5, ptx[1], 'CW Tx', va='center', fontsize=7)
    ax.add_patch(plt.Rectangle((-0.9, -0.18), 1.8, 0.16, color='#7b3294'))
    ax.add_patch(plt.Rectangle((-0.9, 0.02), 1.8, 0.16, color='#1f5fa8'))
    ax.text(-1.1, -0.5, 'T-RIS', ha='right', va='center', fontsize=6.5, color='#7b3294')
    ax.text(-1.1, 0.5, 'R-RIS', ha='right', va='center', fontsize=6.5, color='#1f5fa8')
    ax.plot(*pt, 'o', color='k', ms=6)
    v = mc.V_VEC / np.linalg.norm(mc.V_VEC)
    ax.annotate('', xy=pt + 1.4 * v, xytext=pt, arrowprops=dict(arrowstyle='->', lw=1.0))
    ax.text(pt[0] + 0.75, pt[1] - 0.8, 'target\n2 m/s', fontsize=6.5)
    u = (pt - p_rx) / np.linalg.norm(pt - p_rx)
    perp = np.array([-u[1], u[0]])
    a, b = p_rx - 0.6 * perp, p_rx + 0.6 * perp
    ax.plot([a[0], b[0]], [a[1], b[1]], color='#2e7d32', lw=3)
    ax.text(p_rx[0] - 0.3, p_rx[1] + 0.75, 'Rx ULA (16 el.)', fontsize=6.5, ha='center')
    ax.add_patch(Arc(pr, 3.2, 3.2, theta1=90 - mc.THETA1_DEG, theta2=90, lw=0.7))
    ax.text(0.35, 1.75, '$\\theta_1=60^\\circ$', fontsize=6.5)
    d1, d2 = pr - pt, p_rx - pt
    a1, a2 = np.degrees(np.arctan2(d1[1], d1[0])), np.degrees(np.arctan2(d2[1], d2[0]))
    lo, hi = sorted([a1, a2])
    if hi - lo > 180:
        lo, hi = hi, lo + 360
    ax.add_patch(Arc(pt, 2.4, 2.4, theta1=lo, theta2=hi, lw=0.7))
    mid = np.deg2rad((lo + hi) / 2)
    ax.text(pt[0] + 1.55 * np.cos(mid) - 2, pt[1] + 1.55 * np.sin(mid), '$\\beta=60^\\circ$', fontsize=6.5)
    ax.text(0.25, -4.2, '$d_{TR}=8$ m', fontsize=6.5)
    mm = (pr + pt) / 2
    ax.text(mm[0] + 0.2, mm[1] - 0.9, '$R_{\\rm ris\\text{-}t}=10$ m', fontsize=6.5, color='#d95f02')
    mm = (pt + p_rx) / 2
    ax.text(mm[0] + 0.3, mm[1] + 0.35, '$R_{\\rm t\\text{-}rx}=8$ m', fontsize=6.5, color='#2e7d32')
    ax.set_aspect('equal')
    ax.set_xlim(-3.2, 11.5)
    ax.set_ylim(-9.0, 11.0)
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.grid(True, lw=0.3)
    fig.tight_layout()
    return mc.save_fig(fig, out, 'F1a_scene')


if __name__ == '__main__':
    from mfsk import common as mc
    mc.set_ieee_style()
    print(fig_scene(FIGDIR))
