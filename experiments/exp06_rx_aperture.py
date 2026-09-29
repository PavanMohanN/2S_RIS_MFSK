"""
exp06_rx_aperture.py -- Receive-aperture scaling
================================================
Position accuracy versus the number of receive elements, with exact
spherical-wavefront snapshots, far-field and range-focused beam scanning.

Paper figures produced: Fig. 16 (FI_rx_aperture)
Outputs: results/exp06_rx_aperture/ (data, tables, extra figures), results/figdata/ (figure data),
         paper/figures/ (paper figures)
Run    : python experiments/exp06_rx_aperture.py [--quick]
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
from figures.fig16_rx_aperture import fig_aperture
from experiments import exp02_estimation_crb as p2

logging.getLogger('fontTools').setLevel(logging.ERROR)
N_REF = 16


def snr_for(n):
    return mc.SNR1_DB + 10 * np.log10(n / N_REF)


def path_errors(n, ntr, batch, p_rx, seed):
    nu = mc.tone_offsets()
    out, done, b = [], 0, 0
    while done < ntr:
        B = min(batch, ntr - done)
        Y, L, _ = p2.simulate_physical(p2.child_rng(seed, n, b), B, snr_for(n), mc.NSW, nu, p_rx)
        o = p2.est_coherent(Y, nu, want=('ml',))
        out.append(p2.wrap_err(o['L_ml'], L))
        done += B
        b += 1
    return np.concatenate(out)


def steer(n, phi, r=None):
    """Steering vectors (len(phi) x n).  Far field if r is None, else exact
    spherical wavefront from a source at distance r and angle phi."""
    x = (np.arange(n) - (n - 1) / 2) * mc.D_RX
    phi = np.atleast_1d(phi)
    if r is None:
        return np.exp(1j * mc.K_C * x[None, :] * np.sin(phi)[:, None])
    r = np.atleast_1d(r)
    if r.size == 1:
        r = np.full(phi.size, r[0])
    d = np.sqrt(r[:, None] ** 2 + x[None, :] ** 2 - 2 * r[:, None] * x[None, :] * np.sin(phi)[:, None])
    return np.exp(-1j * mc.K_C * (d - r[:, None]))


def scan(x, n, r_hat, crb, span_deg=1.5):
    """Two-stage beam scan (coarse, then fine at ~CRB/10) + parabolic peak."""
    B = x.shape[0]
    est = np.empty(B)
    g1 = np.deg2rad(np.arange(-span_deg, span_deg + 1e-9, 0.02))
    step = max(crb / 10, 1e-7)
    g2 = np.arange(-30, 31) * step
    for b in range(B):
        rr = None if r_hat is None else r_hat[b]
        A = steer(n, g1, rr)
        p = np.abs(A.conj() @ x[b]) ** 2
        c = g1[np.argmax(p)]
        g = c + g2
        A = steer(n, g, rr)
        p = np.abs(A.conj() @ x[b]) ** 2
        i = int(np.clip(np.argmax(p), 1, g.size - 2))
        den = p[i - 1] - 2 * p[i] + p[i + 1]
        d = 0.5 * (p[i - 1] - p[i + 1]) / den if den != 0 else 0.0
        est[b] = g[i] + np.clip(d, -0.5, 0.5) * step
    return est


def angle_errors(n, ntr, r, rng, sigma_r=0.03):
    phi = np.deg2rad(rng.uniform(-1, 1, ntr))
    snr_el = mc.snr_snapshot_lin(snr_for(n)) / n
    a = steer(n, phi, r)
    x = (np.sqrt(snr_el) * np.exp(1j * rng.uniform(0, 2 * np.pi, ntr))[:, None] * a
         + (rng.standard_normal((ntr, n)) + 1j * rng.standard_normal((ntr, n))) / np.sqrt(2))
    crb = float(mc.crb_angle_ula(mc.snr_snapshot_lin(snr_for(n)), nrx=n))
    span = max(1.5, 1.0 + 6 * np.degrees(crb))          # never clip the estimate
    e_ff = scan(x, n, None, crb, span) - phi
    e_nf = scan(x, n, r + sigma_r * rng.standard_normal(ntr), crb, span) - phi
    return e_ff, e_nf, crb


def defocus_loss_db(n, r):
    a_true = steer(n, 0.0, r)[0]
    a_ff = steer(n, 0.0)[0]
    return float(-20 * np.log10(np.abs(np.vdot(a_ff, a_true)) / n))


def pos_rmse(eL, eth, p_rx):
    L0 = mc.bistatic_path(mc.P_TARGET, p_rx)
    t0 = mc.bearing(p_rx, mc.P_TARGET)
    p = mc.invert_position(L0 + eL, t0 + eth, p_rx)
    return float(np.sqrt(np.mean(np.sum((p - mc.P_TARGET) ** 2, axis=1))))




def table_aperture(out, rows):
    body = "\n".join(
        f"{r['n']} & {r['aperture_cm']:.1f} & {r['grx_dbi']:.0f} & {r['fraunhofer_m']:.1f} & {r['crb_L']*100:.2f} & "
        f"{np.degrees(r['crb_th']):.4f} & {r['crb_pos']*100:.2f} & {r['pos_nf']*100:.2f} & {r['pos_ff']*100:.2f} \\\\" for r in rows)
    tex = r"""\begin{table}[!t]
\centering
\caption{Positioning Versus Receive Aperture ($\beta=60^\circ$, Rx--Target Distance 8\,m)}
\label{tab:rx_aperture}
\footnotesize
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{@{}ccccccccc@{}}
\hline\hline
$N_{\rm rx}$ & $D$ & $G_{\rm rx}$ & $2D^2/\lambda$ & $\sqrt{\mathrm{CRB}_L}$ & $\sqrt{\mathrm{CRB}_\theta}$ & $\sqrt{\mathrm{CRB}_{\rm pos}}$ & RMSE & RMSE \\
 & (cm) & (dBi) & (m) & (cm) & (deg) & (cm) & focused & far field \\
\hline
""" + body + r"""
\hline\hline
\multicolumn{9}{@{}p{0.97\columnwidth}@{}}{\scriptsize Half-wavelength spacing; $G_{\rm rx}$ and $\mathrm{SNR}_1$ scale with $N_{\rm rx}$ from the 16-element values of Table~\ref{tab:sys_params}. RMSE in cm (Monte Carlo); ``focused'' scans with the Rx--target distance known from the path estimate.}
\end{tabular}
\end{table}
"""
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'TI_rx_aperture.tex').write_text(tex, encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description='receive aperture scaling')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'exp06_rx_aperture'))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--trials', type=int, default=None)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--beta', type=float, default=mc.BETA_DEFAULT_DEG)
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
    ntr = args.trials or (200 if args.quick else 1000)
    p_rx = mc.rx_position(args.beta)
    r3 = float(np.linalg.norm(mc.P_TARGET - p_rx))
    J = mc.position_jacobian(mc.P_TARGET, p_rx)
    jL, jT = float(np.linalg.norm(J[:, 0])), float(np.linalg.norm(J[:, 1]))
    log = []

    def say(s=''):
        print(s, flush=True)
        log.append(s)

    say('=' * 74)
    say(' Receive aperture scaling')
    say('=' * 74)
    say(f' SNR1(16) = {mc.SNR1_DB:.1f} dB   trials = {ntr}   beta = {args.beta:.0f} deg   Rx-target {r3:.2f} m')
    say(f' Jacobian: |dp/dL| = {jL:.2f}, |dp/dtheta| = {jT:.2f} m/rad')
    Ns = [4, 8, 16, 32, 64, 128]
    rows = []
    rng = p2.child_rng(60, 0)
    say('\n  N_rx   D(cm)  2D^2/l(m)  defocus(dB)  sL(cm) CRB/MC   sth(deg) CRB/MC(nf)/MC(ff)      pos CRB / nf / ff (cm)')
    for n in Ns:
        eL = path_errors(n, ntr, args.batch, p_rx, 61)
        e_ff, e_nf, crb_th = angle_errors(n, ntr, r3, rng)
        crb_L = float(mc.crb_path(snr_for(n)))
        D = (n - 1) * mc.D_RX
        r = dict(n=n, aperture_cm=D * 100, grx_dbi=18 + 10 * np.log10(n / N_REF), fraunhofer_m=2 * D ** 2 / mc.LAMBDA,
                 defocus_db=defocus_loss_db(n, r3), snr1_db=float(snr_for(n)), crb_L=crb_L, crb_th=crb_th,
                 rmse_L=float(np.sqrt(np.mean(eL ** 2))), rmse_th_nf=float(np.sqrt(np.mean(e_nf ** 2))),
                 rmse_th_ff=float(np.sqrt(np.mean(e_ff ** 2))), bias_th_ff=float(np.mean(e_ff)),
                 crb_pos=float(mc.crb_position(crb_L, crb_th, mc.P_TARGET, p_rx)),
                 crb_range_part=jL * crb_L, crb_angle_part=jT * crb_th,
                 pos_nf=pos_rmse(eL, e_nf, p_rx), pos_ff=pos_rmse(eL, e_ff, p_rx))
        rows.append(r)
        say(f"  {n:4d}  {r['aperture_cm']:6.1f}  {r['fraunhofer_m']:8.2f}  {r['defocus_db']:10.3f}   "
            f"{crb_L*100:5.2f} {r['rmse_L']/crb_L:5.2f}   {np.degrees(crb_th):.4f} {r['rmse_th_nf']/crb_th:5.2f} {r['rmse_th_ff']/crb_th:6.2f}"
            f"   {r['crb_pos']*100:6.2f} / {r['pos_nf']*100:6.2f} / {r['pos_ff']*100:6.2f}")
    # balance point: range and angular contributions equal (log-log interpolation)
    rr = np.log(np.array([r['crb_range_part'] / r['crb_angle_part'] for r in rows]))
    ln = np.log(np.array(Ns, float))
    i = int(np.where(np.diff(np.sign(rr)) != 0)[0][0])
    n_star = float(np.exp(np.interp(0, [rr[i], rr[i + 1]], [ln[i], ln[i + 1]])))
    slope_th = float(np.polyfit(np.log(Ns[2:]), np.log([r['crb_th'] for r in rows[2:]]), 1)[0])
    slope_L = float(np.polyfit(np.log(Ns), np.log([r['rmse_L'] for r in rows]), 1)[0])
    r16 = next(r for r in rows if r['n'] == 16)
    say(f'\n  balance point N* = {n_star:.1f};  CRB_theta slope {slope_th:.3f} (theory -1.5);  MC path slope {slope_L:.3f} (theory -0.5)')
    for n in (32, 64):
        rn = next(r for r in rows if r['n'] == n)
        say(f"  N = {n}: position {rn['pos_nf']*100:.2f} cm vs {r16['pos_nf']*100:.2f} cm at N = 16 "
            f"({r16['pos_nf']/rn['pos_nf']:.1f}x better)")

    say('\n[out]')
    p = fig_aperture(fdir, rows, n_star, round(r3))
    table_aperture(tdir, rows)
    say(f'    {p}\n    {tdir / "TI_rx_aperture.tex"}')

    say('\n[checks] Validation')
    checks = [
        ('range-focused position RMSE within 1.15x CRB for every N', all(r['pos_nf'] / r['crb_pos'] < 1.15 for r in rows)),
        ('range-focused angle RMSE within 1.15x CRB for every N', all(r['rmse_th_nf'] / r['crb_th'] < 1.15 for r in rows)),
        ('angle CRB scales as N^-1.5 (slope within 0.1)', abs(slope_th + 1.5) < 0.1),
        ('path RMSE scales as N^-0.5 with the array gain (slope within 0.1)', abs(slope_L + 0.5) < 0.1),
        ('far-field scan: negligible defocus for N <= 16 (< 0.1 dB), > 1 dB at N = 128',
         all(r['defocus_db'] < 0.1 for r in rows if r['n'] <= 16) and rows[-1]['defocus_db'] > 1.0),
        ('range and angular contributions balance between N = 16 and 64', 16 < n_star < 64),
    ]
    for n_, ok in checks:
        say(f"    [{'PASS' if ok else 'FAIL'}] {n_}")
    say(f'    time {time.time()-t0:.0f} s')
    res = dict(snr1_db=mc.SNR1_DB, beta_deg=args.beta, trials=ntr, r3_m=r3, jac=[jL, jT], rows=rows, n_star=n_star,
               slope_theta=slope_th, slope_path=slope_L, gate6={n_: bool(o) for n_, o in checks})
    (ddir / 'exp06_results.json').write_text(json.dumps(res, indent=2, default=float), encoding='utf-8')
    (ddir / 'exp06_summary.txt').write_text('\n'.join(log), encoding='utf-8')
    return 0 if all(o for _, o in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
