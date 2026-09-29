r"""
Fig. 4c -- Impedance magnitude of the PIN-diode element
=======================================================
Output : paper/figures/F2c_impedance_magnitude.pdf   (\includegraphics{F2c_impedance_magnitude} in the manuscript)
Data   : computed by this script from the PIN-diode element model, Eqs. (1)-(8)
Run    : python figures/fig04c_impedance_magnitude.py
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
import matplotlib.pyplot as plt
import numpy as np

from mfsk import common as mc
from mfsk import figio

FIGDIR = figio.FIGDIR
F = np.linspace(20e9, 35e9, 1501)


def fig_impedance_magnitude(out):
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 2.3))
    for st, lab, col in (('on', 'forward bias (on)', '#1f5fa8'), ('off', 'reverse bias (off)', '#c0392b')):
        z = np.abs(mc.z1(F, st))
        ax.semilogy(F / 1e9, z, color=col, lw=1.2, label=lab)
        v = float(np.interp(mc.FC, F, z))
        ax.plot(mc.FC / 1e9, v, 'o', color=col, ms=3.5)
        ax.annotate('%.0f $\\Omega$' % v, (mc.FC / 1e9, v), xytext=(4, 4), textcoords='offset points', fontsize=6.5, color=col)
    ax.axhline(mc.Z0, color='0.5', lw=0.6, ls='--')
    ax.text(F[0] / 1e9 + 0.2, mc.Z0 * 1.12, '$Z_0$', fontsize=6.5, color='0.35')
    ax.axvline(mc.FC / 1e9, color='0.5', lw=0.6, ls=':')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('$|Z_1|$ ($\\Omega$)')
    ax.set_xlim(F[0] / 1e9, F[-1] / 1e9)
    ax.grid(True, which='both')
    ax.legend(fontsize=6.5, loc='best')
    fig.tight_layout()
    return mc.save_fig(fig, out, 'F2c_impedance_magnitude')


if __name__ == '__main__':
    mc.set_ieee_style()
    print(fig_impedance_magnitude(FIGDIR))
