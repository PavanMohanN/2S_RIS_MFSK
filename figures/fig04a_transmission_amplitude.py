r"""
Fig. 4a -- Transmission amplitude of the T-RIS element
======================================================
Output : paper/figures/F2a_transmission_amplitude.pdf   (\includegraphics{F2a_transmission_amplitude} in the manuscript)
Data   : computed by this script from the PIN-diode element model, Eqs. (1)-(8)
Run    : python figures/fig04a_transmission_amplitude.py
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


def fig_transmission_amplitude(out):
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 2.3))
    curves = [('on state $|T_n(f,\\mathrm{on})|$', mc.t_n(F, 'on'), '#1f5fa8', '-'),
              ('off state $|T_n(f,\\mathrm{off})|$', mc.t_n(F, 'off'), '#c0392b', '-'),
              ('sideband $|T_{\\mathrm{sb}}|$', mc.t_sb(F), '#2e7d32', '--'),
              ('carrier $|T_{\\mathrm{mean}}|$', mc.t_mean(F), '#7b3294', ':')]
    for lab, t, col, ls in curves:
        ax.plot(F / 1e9, mc.db20(t), ls, color=col, lw=1.2, label=lab)
        v = float(mc.db20(np.interp(mc.FC, F, np.abs(t))))
        ax.plot(mc.FC / 1e9, v, 'o', color=col, ms=3.5)
        ax.annotate('%.2f dB' % v, (mc.FC / 1e9, v), xytext=(4, 3), textcoords='offset points', fontsize=6, color=col)
    ax.axvline(mc.FC / 1e9, color='0.5', lw=0.6, ls=':')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Amplitude (dB)')
    ax.set_xlim(F[0] / 1e9, F[-1] / 1e9)
    ax.grid(True)
    ax.legend(fontsize=6, loc='lower left')
    fig.tight_layout()
    return mc.save_fig(fig, out, 'F2a_transmission_amplitude')


if __name__ == '__main__':
    mc.set_ieee_style()
    print(fig_transmission_amplitude(FIGDIR))
