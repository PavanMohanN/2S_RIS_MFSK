r"""
Fig. 14 -- Inter-stage coupling and assembly errors
===================================================
Output : paper/figures/FH_interlayer_coupling.pdf   (\includegraphics{FH_interlayer_coupling} in the manuscript)
Data   : results/figdata/FH_interlayer_coupling.json + .npz, written by experiments/exp04_hardware_coupling.py
Run    : python figures/fig14_interlayer_coupling.py   (re-plots from the saved data;
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


@figio.paper_figure('FH_interlayer_coupling')
def fig_coupling(out, B):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(mc.IEEE_COL_W, 4.4), gridspec_kw={'hspace': 0.5})
    z = np.array(B['gap_mm'])
    cols = ['#c0392b', '#1f5fa8', '#2e7d32']
    for (f, e), c in zip(B['gap_curves'].items(), cols):
        a1.plot(z, e, color=c, label='angular spectrum, $f=%.2f$' % float(f))
        a1.axhline(40 * np.log10(float(f)), color=c, ls=':', lw=0.6)
    a1.plot(z, B['gap_paper_db'], 'k--', lw=1.0, label='submitted Eq. (11)')
    a1.axvline(2.0, color='0.5', lw=0.6)
    a1.text(2.08, -5.4, 'nominal 2 mm', fontsize=7, color='0.15')
    a1.set_xlabel('Inter-stage gap $d_g$ (mm)')
    a1.set_ylabel('Coupling efficiency (dB)')
    a1.set_ylim(-32, 1)
    a1.grid(True)
    a1.legend(fontsize=5.8, loc='lower left')
    a1.set_title('(a) Coupling versus gap (dotted: fundamental-mode limit $f^4$)', fontsize=7.5)

    for key, lab, c, ls in (('gap', 'axial gap $\\pm\\Delta d_g$', '#7b3294', '-'),
                            ('lat_x', 'lateral offset ($x$)', '#1f5fa8', '-'),
                            ('lat_diag', 'lateral offset (diagonal)', '#1f5fa8', '--'),
                            ('rot', 'in-plane rotation', '#d95f02', '-'),
                            ('tilt', 'tilt', '#2e7d32', '-')):
        r = B['mis'][key]
        a2.plot(r['disp_mm'], r['loss_db'], ls, color=c, label=lab)
    r = B['mis']['lat_paper']
    a2.plot(r['disp_mm'], r['loss_db'], ':', color='k', lw=1.0, label='lateral, submitted model')
    a2.axhline(1.0, color='0.4', lw=0.6, ls=':')
    a2.set_xlabel('Maximum element displacement (mm)')
    a2.set_ylabel('Extra loss re. nominal (dB)')
    a2.axhline(0, color='k', lw=0.5)
    a2.set_xlim(0, 2.7)
    a2.set_ylim(-2.5, 8)
    a2.grid(True)
    a2.legend(fontsize=5.6, loc='upper left', ncol=1)
    a2.set_title('(b) Assembly errors at $d_g=2$ mm, $f=%.2f$' % B['f_primary'], fontsize=7.5)
    return mc.save_fig(fig, out, 'FH_interlayer_coupling')


if __name__ == '__main__':
    print(figio.replot(fig_coupling))
