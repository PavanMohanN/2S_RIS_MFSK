"""
exp03_beam_squint.py -- Multi-tone beam coherence and squint
============================================================
Exact phase-only squint of every tone, per-tone gain toward the target,
CRB increase and range bias, validity map N_x B_f sin(theta_1), and the
centre-frequency gradient design.

Paper figures produced: Fig. 2 (F4rev_multitone_beam)
Outputs: results/exp03_beam_squint/ (data, tables, extra figures), results/figdata/ (figure data),
         paper/figures/ (paper figures)
Run    : python experiments/exp03_beam_squint.py
Ends with self-validation checks; exit code 0 means all passed.
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mfsk import common as mc
from figures.fig02_multitone_beam import fig_multitone

logging.getLogger('fontTools').setLevel(logging.ERROR)


def af_peak_deg(f_abs, theta1_deg, n=mc.N_X, f_design=mc.FC, span=6.0):
    """Beam direction from the array factor (ground truth for the formula)."""
    g = np.deg2rad(np.linspace(theta1_deg - span, theta1_deg + span, 120001))
    a = np.abs(mc.af_line(g, f_abs, np.deg2rad(theta1_deg), n=n, f_design=f_design))[0]
    i = int(np.argmax(a))
    return float(np.degrees(g[i]))


def tone_gains(theta1_deg, nu, n=mc.N_X, f_design=mc.FC, bits=None):
    """Complex normalised gain of each tone toward the target."""
    th = np.deg2rad(theta1_deg)
    return mc.af_line(np.array([th]), mc.FC + nu, th, n=n, f_design=f_design, bits=bits)[:, 0]


def range_bias_cm(g, nu):
    """Noise-free bias of the ML path estimate caused by tone-dependent gains."""
    L0 = 6.0
    z = g * np.exp(-1j * 2 * np.pi * nu * L0 / mc.C_LIGHT)
    grid = L0 + np.linspace(-0.05, 0.05, 20001)
    p = np.abs(np.exp(1j * 2 * np.pi * np.outer(grid, nu) / mc.C_LIGHT) @ z) ** 2
    return float((grid[np.argmax(p)] - L0) * 100)


def evaluate(theta1_deg, nu, f_design, n=mc.N_X):
    g = tone_gains(theta1_deg, nu, n=n, f_design=f_design)
    g_ref = tone_gains(theta1_deg, np.array([f_design - mc.FC]), n=n, f_design=f_design)[0]
    rel = np.abs(g / g_ref) ** 2                     # squint-only loss, re. design frequency
    sq_f = mc.squint_exact_deg(mc.FC + nu - f_design, theta1_deg, f_design)
    sq_af = np.array([af_peak_deg(mc.FC + v, theta1_deg, n=n, f_design=f_design) for v in nu]) - theta1_deg
    crb0 = mc.crb_path(mc.SNR1_DB, nu=nu)
    crb1 = mc.crb_path(mc.SNR1_DB, nu=nu, weights=rel)
    return dict(theta1=theta1_deg, hpbw=float(mc.hpbw_deg(theta1_deg, n)),
                max_squint=float(np.max(np.abs(sq_f))),
                max_squint_pct=float(100 * np.max(np.abs(sq_f)) / mc.hpbw_deg(theta1_deg, n)),
                formula_vs_af_max_dev=float(np.max(np.abs(sq_f - sq_af))),
                worst_loss_db=float(-10 * np.log10(rel.min())),
                aggregate_loss_db=float(-10 * np.log10(rel.mean())),
                crb_increase_pct=float(100 * (crb1 / crb0 - 1)),
                bias_cm=range_bias_cm(g, nu),
                per_tone_squint_deg=sq_f.tolist(), per_tone_gain_db=(10 * np.log10(rel)).tolist())


def validity_map(theta1_deg, bf, nx):
    th = np.deg2rad(theta1_deg)
    L = np.zeros((nx.size, bf.size))
    for i, n in enumerate(nx):
        g = mc.af_line(np.array([th]), mc.FC * (1 + bf), th, n=int(n))[:, 0]
        L[i] = -20 * np.log10(np.abs(g))
    return L




def table_squint(out, rows):
    body = []
    for r in rows:
        body.append(f"{r['theta1']:.0f} & {r['design']} & {r['hpbw']:.1f} & {r['max_squint']:.2f} ({r['max_squint_pct']:.1f}) & "
                    f"{r['worst_loss_db']:.3f} & {r['aggregate_loss_db']:.3f} & {r['crb_increase_pct']:.2f} & "
                    f"{r['bias_cm']*10:.2f} \\\\")
    tex = r"""\begin{table}[!t]
\centering
\caption{Multi-Tone Beam Coherence of the Two-Stage R-RIS ($N_x=16$, $M=12$, $f_{m,M}=300$\,MHz)}
\label{tab:squint}
\footnotesize
\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{@{}cccccccc@{}}
\hline\hline
$\theta_1$ & Design & HPBW & Max squint & Worst & Aggr. & CRB & Bias \\
(deg) & freq. & (deg) & (deg, \% HPBW) & (dB) & (dB) & (+\%) & (mm) \\
\hline
""" + "\n".join(body) + r"""
\hline\hline
\multicolumn{8}{@{}p{0.97\columnwidth}@{}}{\scriptsize Worst/Aggr.: loss toward the target of the worst tone and of the mean tone power, relative to the design frequency. CRB: increase of \eqref{eq:crb_R} from the tone-dependent gains. Bias: noise-free bias of the ML path estimate. A single-stage STM-RIS re-steered per symbol has zero squint but reloads $M$ phase maps per sweep.}
\end{tabular}
\end{table}
"""
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'TE_squint_summary.tex').write_text(tex, encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description='multi-tone beam coherence')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'exp03_beam_squint'))
    ap.add_argument('--snr1', type=float, default=None)
    args = ap.parse_args(argv)
    if args.snr1 is not None:
        mc.SNR1_DB = float(args.snr1)
        mc.SNR_CPI_DB = mc.SNR1_DB + 10 * np.log10(mc.NSW)
    t0 = time.time()
    mc.set_ieee_style()
    out = Path(args.out)
    fdir, tdir, ddir = out / 'figures', out / 'tables', out / 'data'
    for d in (fdir, tdir, ddir):
        d.mkdir(parents=True, exist_ok=True)
    log = []

    def say(s=''):
        print(s, flush=True)
        log.append(s)

    say('=' * 74)
    say(' Multi-tone beam coherence and squint')
    say('=' * 74)
    nu = mc.tone_offsets()
    f_mid = mc.FC + 0.5 * (nu[0] + nu[-1])

    say('\n[1] Squint formula check at theta1 = 60 deg, worst tone (300 MHz)')
    th1 = mc.THETA1_DEG
    sub = float(np.degrees(np.arcsin(np.sin(np.deg2rad(th1)) * nu[-1] / mc.FC)))
    exact = float(mc.squint_exact_deg(nu[-1], th1))
    peak = af_peak_deg(mc.FC + nu[-1], th1) - th1
    say(f'    submitted Eq.(14)   : {sub:+.3f} deg')
    say(f'    exact (phase-only)  : {exact:+.3f} deg   small-angle {-np.degrees(np.tan(np.deg2rad(th1))*nu[-1]/mc.FC):+.3f}')
    say(f'    array-factor peak   : {peak:+.3f} deg')

    say('\n[2] Per-design summary (continuous phases; loss re. design frequency)')
    rows = []
    for t in (30.0, 60.0, 75.0):
        for label, fd in (('$f_c$', mc.FC), ('centre', f_mid)):
            r = evaluate(t, nu, fd)
            r['design'] = label
            r['design_hz'] = fd
            rows.append(r)
            say(f"    th1={t:4.0f}  design={('carrier' if fd == mc.FC else 'centre'):7s} HPBW {r['hpbw']:5.2f}  "
                f"max squint {r['max_squint']:.3f} ({r['max_squint_pct']:.1f}%)  worst {r['worst_loss_db']:.3f} dB  "
                f"aggr {r['aggregate_loss_db']:.3f} dB  CRB +{r['crb_increase_pct']:.2f}%  bias {r['bias_cm']*10:+.3f} mm  "
                f"[formula-AF dev {r['formula_vs_af_max_dev']:.4f} deg]")
    r60 = next(r for r in rows if r['theta1'] == 60 and r['design_hz'] == mc.FC)
    r60c = next(r for r in rows if r['theta1'] == 60 and r['design_hz'] == f_mid)
    crb = float(mc.crb_path(mc.SNR1_DB, nu=nu))

    say('\n[3] Validity map (worst-tone loss vs fractional bandwidth and N_x, theta1 = 60 deg)')
    bf = np.logspace(np.log10(0.001), np.log10(0.06), 90)
    nx = np.unique(np.round(np.logspace(np.log10(4), np.log10(128), 60)).astype(int))
    L = validity_map(th1, bf, nx)
    # rule constant from the exact 1-dB contour
    kk = []
    for i, n in enumerate(nx):
        j = np.where(L[i] >= 1.0)[0]
        if j.size and j[0] > 0:
            b1 = np.interp(1.0, L[i, j[0] - 1:j[0] + 1], bf[j[0] - 1:j[0] + 1])
            kk.append(n * b1 * np.sin(np.deg2rad(th1)))
    rule_k = float(np.median(kk))
    op_bf = nu[-1] / mc.FC
    say(f'    1-dB contour: N_x * B_f * sin(theta1) = {rule_k:.3f} (median; spread {np.min(kk):.3f}-{np.max(kk):.3f})')
    say(f'    this work: N_x B_f sin = {mc.N_X*op_bf*np.sin(np.deg2rad(th1)):.3f}  -> worst loss {r60["worst_loss_db"]:.3f} dB')
    big = {}
    for n in (32, 64, 128):
        g = mc.af_line(np.array([np.deg2rad(th1)]), mc.FC + nu[-1], np.deg2rad(th1), n=n)[0, 0]
        big[n] = float(-20 * np.log10(abs(g)))
        say(f'    if the R-RIS had N_x = {n:3d}: worst-tone loss {big[n]:.2f} dB (carrier design)')

    say('\n[4] Writing figure and table')
    p = fig_multitone(fdir, nu, int(th1), bf, nx, L, op_bf, rule_k)
    table_squint(tdir, rows)
    say(f'    {p}\n    {tdir / "TE_squint_summary.tex"}')

    say('\n[checks] Validation')
    checks = [
        ('exact squint formula matches array-factor peak (< 0.01 deg, all angles/designs)',
         max(r['formula_vs_af_max_dev'] for r in rows) < 0.01),
        ('worst-tone squint < HPBW/2 at 30/60/75 deg', all(r['max_squint'] < r['hpbw'] / 2 for r in rows)),
        ('aggregate squint loss < 0.1 dB at 60 deg (carrier design)', r60['aggregate_loss_db'] < 0.1),
        ('squint-induced range bias < 0.1 x CRB at SNR1', abs(r60['bias_cm']) < 0.1 * crb * 100),
        ('rule N_x B_f sin(theta1) constant along 1-dB contour (spread < 15%)',
         (np.max(kk) - np.min(kk)) / rule_k < 0.15),
        ('centre-frequency design halves the maximum squint (ratio 0.45-0.55)',
         0.45 < r60c['max_squint'] / r60['max_squint'] < 0.55),
    ]
    for n_, ok in checks:
        say(f"    [{'PASS' if ok else 'FAIL'}] {n_}")
    say(f'    info: submitted Eq.(14) differs from the array factor by {abs(sub - abs(peak)):.2f} deg '
        f'({abs(sub)/abs(peak)*100:.0f}% of the true magnitude, opposite sign) -> correction required')
    say(f'    time {time.time()-t0:.1f} s')

    res = dict(snr1_db=mc.SNR1_DB, theta1_deg=th1, squint_submitted_eq14_deg=sub, squint_exact_deg=exact,
               squint_af_peak_deg=peak, rows=rows, rule_k=rule_k, rule_k_range=[float(np.min(kk)), float(np.max(kk))],
               this_work_nbs=float(mc.N_X * op_bf * np.sin(np.deg2rad(th1))), crb_path_cm=crb * 100,
               image_squint_deg=float(mc.squint_exact_deg(-nu[-1], th1)),
               f_mid_mhz=float((f_mid - mc.FC) / 1e6), loss_n32_db=big[32], loss_n64_db=big[64],
               loss_n128_db=big[128], gate3={n_: bool(o) for n_, o in checks})
    (ddir / 'exp03_results.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    (ddir / 'exp03_summary.txt').write_text('\n'.join(log), encoding='utf-8')
    return 0 if all(o for _, o in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
