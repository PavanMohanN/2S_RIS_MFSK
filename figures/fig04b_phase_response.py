r"""
Fig. 4b -- Transmission phase of the T-RIS element
==================================================
Output : paper/figures/F2b_phase_response.pdf   (\includegraphics{F2b_phase_response} in the manuscript)
Data   : computed by this script from the PIN-diode element model, Eqs. (1)-(8)
Run    : python figures/fig04b_phase_response.py
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


def fig_phase_response(out):
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 2.3))
    p_on = np.degrees(np.angle(mc.t_n(F, 'on')))
    p_off = np.degrees(np.angle(mc.t_n(F, 'off')))
    ax.plot(F / 1e9, p_on, color='#1f5fa8', lw=1.2, label='on state')
    ax.plot(F / 1e9, p_off, color='#c0392b', lw=1.2, label='off state')
    a = float(np.interp(mc.FC, F, p_on))
    b = float(np.interp(mc.FC, F, p_off))
    ax.plot([mc.FC / 1e9] * 2, [a, b], 'k.-', lw=0.8, ms=4)
    ax.annotate('$\\Delta\\angle T_n = %.1f^\\circ$ at $f_c$' % abs(b - a), (mc.FC / 1e9, (a + b) / 2),
                xytext=(6, 0), textcoords='offset points', fontsize=6.5, va='center')
    ax.axvline(mc.FC / 1e9, color='0.5', lw=0.6, ls=':')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Phase $\\angle T_n$ (deg)')
    ax.set_xlim(F[0] / 1e9, F[-1] / 1e9)
    ax.grid(True)
    ax.legend(fontsize=6.5, loc='best')
    fig.tight_layout()
    return mc.save_fig(fig, out, 'F2b_phase_response')


if __name__ == '__main__':
    mc.set_ieee_style()
    print(fig_phase_response(FIGDIR))
