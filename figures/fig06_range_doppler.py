r"""
Fig. 6 -- Range profile and range-Doppler map
=============================================
Output : paper/figures/F6rev_range_doppler.pdf   (\includegraphics{F6rev_range_doppler} in the manuscript)
Data   : computed by this script
Run    : python figures/fig06_range_doppler.py
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
from experiments import exp02_estimation_crb as p2
from mfsk import figio

FIGDIR = figio.FIGDIR


def fig_range_doppler(out, seed=2026):
    """Single-target detection for the geometry of Table I."""
    rng = np.random.default_rng(seed)
    p_rx = mc.rx_position(mc.BETA_DEFAULT_DEG)
    L0 = float(mc.bistatic_path(mc.P_TARGET, p_rx))
    Ld0 = float(mc.bistatic_path_rate(mc.P_TARGET, p_rx))
    nu = mc.tone_offsets()
    m, n = mc.M_TONES, mc.NSW
    t = np.arange(n)[None, :] * mc.TSW + np.arange(m)[:, None] * mc.TSYM
    snr = 10 ** (mc.SNR1_DB / 10)
    Y = np.sqrt(snr) * np.exp(-1j * 2 * np.pi * (nu[:, None] * L0 + (mc.FC + nu[:, None]) * Ld0 * t) / mc.C_LIGHT)
    Y = Y + (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(2)
    o = p2.est_coherent(Y[None], nu, want=('ml',))
    amb = mc.C_LIGHT / mc.DELTA_F
    Lh = float(o['L_ml'][0]) + amb * np.round((L0 - float(o['L_ml'][0])) / amb)
    v0 = -Ld0 / 2
    vh = float(o['v_ml'][0])
    nd, nr = 4 * n, 64 * m
    D = np.fft.fftshift(np.fft.fft(Y, n=nd, axis=1), axes=1)
    fd = np.fft.fftshift(np.fft.fftfreq(nd, mc.TSW))
    vaxis = fd * mc.LAMBDA / 2
    R = np.fft.ifft(D, n=nr, axis=0)
    Lax = np.arange(nr) * amb / nr + amb * np.floor(L0 / amb)
    P = np.abs(R) ** 2
    P /= P.max()
    j = int(np.argmin(np.abs(vaxis - v0)))
    prof = P[:, j]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(mc.IEEE_COL_W, 4.3), gridspec_kw={'hspace': 0.55})
    a1.plot(Lax, 10 * np.log10(prof + 1e-6), color='#1f5fa8', lw=0.9)
    a1.axvline(L0, color='k', ls='--', lw=0.8, label='true path %.2f m' % L0)
    a1.axvspan(L0 - 0.5, L0 + 0.5, color='0.85', zorder=0, label='resolution cell $c/(M\\Delta f)$')
    a1.set_xlim(Lax[0], Lax[0] + amb)
    a1.set_ylim(-35, 3)
    a1.set_xlabel('Bistatic path $L$ (m), $n_{\\rm amb}=%d$' % int(np.floor(L0 / amb)))
    a1.set_ylabel('Normalized power (dB)')
    a1.legend(fontsize=5.8, loc='upper left')
    a1.grid(True)
    a1.set_title('(a) IFFT path profile at the target Doppler', fontsize=7.5)
    keep = np.abs(vaxis - v0) < 3.0
    im = a2.pcolormesh(Lax, vaxis[keep], 10 * np.log10(P[:, keep].T + 1e-6), vmin=-30, vmax=0,
                       cmap='viridis', shading='auto', rasterized=True)
    a2.plot(L0, v0, 'w+', ms=8, mew=1.2)
    a2.set_xlim(L0 - 3, L0 + 3)
    a2.set_xlabel('Bistatic path $L$ (m)')
    a2.set_ylabel('Radial velocity $v$ (m/s)')
    cb = fig.colorbar(im, ax=a2, pad=0.02)
    cb.set_label('dB', fontsize=7)
    a2.set_title('(b) Range-Doppler map (+: truth)', fontsize=7.5)
    p = mc.save_fig(fig, out, 'F6rev_range_doppler')
    return dict(L0=L0, v0=v0, Lh=Lh, vh=vh, dL_cm=(Lh - L0) * 100, dv_cms=(vh - v0) * 100, n_amb=int(np.floor(L0 / amb)))


if __name__ == '__main__':
    mc.set_ieee_style()
    r = fig_range_doppler(FIGDIR)
    print('true path %.2f m, v %.3f m/s; errors %+.2f cm, %+.2f cm/s' % (r['L0'], r['v0'], r['dL_cm'], r['dv_cms']))
