r"""
Supplementary Fig. S1 -- Image rejection and modulator bandwidth
================================================================
Output : paper/figures/FC_sideband_separation.pdf   (\includegraphics{FC_sideband_separation} in the manuscript)
Data   : results/figdata/FC_sideband_separation.json + .npz, written by experiments/exp01_receiver_sideband.py
Run    : python figures/figS01_sideband_separation.py   (re-plots from the saved data;
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


def irr_analytic_db(gain_db, phase_deg):
    g = 10.0 ** (np.asarray(gain_db) / 20.0)
    phi = np.deg2rad(np.asarray(phase_deg))
    num = 1.0 + 2.0 * g * np.cos(phi) + g ** 2
    den = 1.0 - 2.0 * g * np.cos(phi) + g ** 2
    return 10.0 * np.log10(num / np.maximum(den, 1e-30))


def modulator_response(f, tau_sw):
    """First-order T-RIS modulator transfer (bias network + diode charge
    dynamics lumped into an effective time constant tau_sw)."""
    return 1.0 / (1.0 + 1j * 2 * np.pi * np.asarray(f) * tau_sw)


@figio.paper_figure('FC_sideband_separation')
def fig_sideband_separation(out_fig, sim_pts, irr_req_db, typ, options):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(mc.IEEE_COL_W, 4.3),
                                 gridspec_kw={'hspace': 0.42})
    # (a) IRR map with simulated validation points
    gdb = np.linspace(0.0, 1.05, 211)
    pdeg = np.linspace(0.0, 6.0, 241)
    G, Pd = np.meshgrid(gdb, pdeg)
    IRR = irr_analytic_db(G, Pd)
    cs = a1.contourf(G, Pd, IRR, levels=np.arange(20, 62, 4), cmap='viridis')
    cb = fig.colorbar(cs, ax=a1, pad=0.02)
    cb.set_label('IRR (dB)', fontsize=7)
    cr = a1.contour(G, Pd, IRR, levels=[irr_req_db], colors='w', linewidths=1.2, linestyles='--')
    a1.clabel(cr, fmt={irr_req_db: 'requirement %.1f dB' % irr_req_db}, fontsize=6,
              manual=[(0.72, 3.2)], inline_spacing=2)
    for p in sim_pts:
        a1.plot(p['gain_db'], p['phase_deg'], 'o', mfc='none', mec='w', ms=3.2, mew=0.7)
    a1.plot(typ[0], typ[1], '*', color=mc.COLORS['accent'], ms=8, mec='w', mew=0.5)
    a1.annotate('assumed %.1f dB / %.0f$^\\circ$' % (typ[0], typ[1]), (typ[0], typ[1]),
                xytext=(0.30, 2.45), color='w', fontsize=6,
                arrowprops=dict(arrowstyle='->', color='w', lw=0.6))
    a1.set_xlim(0, 1.03)
    a1.set_xlabel('I/Q gain imbalance (dB)')
    a1.set_ylabel('I/Q phase imbalance (deg)')
    a1.set_title('(a) Image rejection of the complex-baseband receiver', fontsize=7.5)

    # (b) modulator bandwidth trade-off
    tau = np.logspace(np.log10(0.03e-9), np.log10(2e-9), 300)
    cols = ['#1f5fa8', '#d95f02', '#7b3294', '#1b9e77']
    for c, o in zip(cols, options):
        loss = -mc.db20(modulator_response(o['f_max_hz'], tau))
        a2.loglog(tau * 1e9, loss, color=c,
                  label=r'$f_{\rm off}$=%d MHz ($f_{\max}$=%d MHz)' % (o['f_off_mhz'], o['f_max_mhz']))
        a2.plot(o['tau_req_ns'], 1.0, 'o', color=c, ms=3.5)
    a2.axhline(1.0, color='0.3', lw=0.7, ls=':')
    a2.text(0.62, 0.72, '1 dB sideband-loss budget', fontsize=6)
    a2.set_xlabel(r'Effective T-RIS modulator time constant $\tau_{\rm sw}$ (ns)')
    a2.set_ylabel(r'Sideband loss at $f_{\max}$ (dB)')
    a2.set_ylim(0.01, 30)
    a2.grid(True, which='both')
    a2.legend(fontsize=5.8, loc='upper left')
    a2.set_title('(b) Required modulation bandwidth vs tone plan', fontsize=7.5)
    return mc.save_fig(fig, out_fig, 'FC_sideband_separation')


if __name__ == '__main__':
    print(figio.replot(fig_sideband_separation))
