r"""
Fig. 3 -- Per-symbol complex-baseband spectrum
==============================================
Output : paper/figures/F3rev_baseband_spectrum.pdf   (\includegraphics{F3rev_baseband_spectrum} in the manuscript)
Data   : results/figdata/F3rev_baseband_spectrum.json + .npz, written by experiments/exp01_receiver_sideband.py
Run    : python figures/fig03_baseband_spectrum.py   (re-plots from the saved data;
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
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from mfsk import common as mc
from mfsk import figio

FIGDIR = figio.FIGDIR


@figio.paper_figure('F3rev_baseband_spectrum')
def fig_baseband_spectrum(out_fig, f_off):
    """Revised Fig. 3: per-symbol complex-baseband spectrum after I/Q down-
    conversion (LO at f_c), referenced to the incident CW (0 dB).  Each
    symbol's spectrum is computed from a time-domain synthesis over T_sym;
    the envelope over the M symbols is shown.  Tones fall on the nulls of
    every other symbol's sinc response because DELTA_F * T_sym is an integer."""
    nu = mc.tone_offsets(f_off)
    fs, pad = 2.0e9, 16
    nsym = int(round(fs * mc.TSYM))
    a_sb, a_car = abs(mc.t_sb(mc.FC)) / 2.0, abs(mc.t_mean(mc.FC))
    t = np.arange(nsym) / fs
    nfft = nsym * pad
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    env = np.full(nfft, -np.inf)
    for fk in nu:
        x = a_sb * (np.exp(1j * 2 * np.pi * fk * t) + np.exp(-1j * 2 * np.pi * fk * t)) + a_car
        X = np.fft.fftshift(np.fft.fft(x, nfft)) / nsym
        env = np.maximum(env, mc.db20(X))
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 2.05))
    band = nu.max() + mc.DELTA_F / 2
    ax.axvspan(-band / 1e6, band / 1e6, color='#e3ebf5', zorder=0,
               label=r'analog I/Q channel $\pm$%.1f MHz (%.1f %%)' % (band / 1e6, 2 * band / mc.FC * 100))
    lsb, lcar = mc.db20(a_sb), mc.db20(a_car)
    for k, fk in enumerate(nu):
        ax.plot([fk / 1e6] * 2, [-70, lsb], color=mc.COLORS['green'], lw=1.1, zorder=3,
                label=r'desired $+f_{m,k}$' if k == 0 else None)
        ax.plot([-fk / 1e6] * 2, [-70, lsb], color=mc.COLORS['accent'], lw=1.1, ls=(0, (2, 1)), zorder=3,
                label=r'image $-f_{m,k}$' if k == 0 else None)
    ax.plot([0, 0], [-70, lcar], color='#e08e0b', lw=1.6, zorder=3,
            label='carrier + direct path (DC)')
    ax.annotate('', xy=(nu[0] / 1e6, -24), xytext=(-nu[0] / 1e6, -24),
                arrowprops=dict(arrowstyle='<->', lw=0.6))
    ax.text(0, -28, r'$2f_{m,1}$ = %.0f MHz = %d bins of $1/T_{\rm sym}$' % (2 * nu[0] / 1e6, round(2 * nu[0] * mc.TSYM)),
            ha='center', va='top', fontsize=6, bbox=dict(fc='w', ec='none', pad=0.6))
    ax.set_xlim(-1.18 * nu.max() / 1e6, 1.18 * nu.max() / 1e6)
    ax.set_ylim(-50, 3)
    ax.set_xlabel('Complex-baseband frequency (MHz)')
    ax.set_ylabel('Amplitude (dB re. CW)')
    ax.grid(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=5.8,
              handlelength=1.5, columnspacing=0.9, borderaxespad=0.0, frameon=False)
    return mc.save_fig(fig, out_fig, 'F3rev_baseband_spectrum')


if __name__ == '__main__':
    print(figio.replot(fig_baseband_spectrum))
