"""
exp02_estimation_crb.py -- Estimators and Cramer-Rao bounds
===========================================================
Monte Carlo RMSE versus SNR for bistatic path, velocity, angle and position
against the exact bounds; estimator-efficiency decomposition (bias,
variance, outliers); zero-padding study; hardware mechanisms (phase
quantisation, element spread, pointing) with common random numbers.

Paper figures produced: Fig. 7 (F7rev_rmse_path_position), Fig. 8 (F8rev_rmse_velocity), Fig. 9 (FA_estimator_efficiency)
Outputs: results/exp02_estimation_crb/ (data, tables, extra figures), results/figdata/ (figure data),
         paper/figures/ (paper figures)
Run    : python experiments/exp02_estimation_crb.py [--quick]
Ends with self-validation checks; exit code 0 means all passed.
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging

from mfsk import common as mc
from figures.fig07_rmse_path_position import fig_rmse_path_position
from figures.fig08_rmse_velocity import fig_rmse_velocity
from figures.fig09_estimator_efficiency import fig_efficiency

logging.getLogger('fontTools').setLevel(logging.ERROR)

OUTLIER_M = 0.5            # |path error| > half the 1 m Rayleigh cell = threshold outlier
NFFT_R_FACTOR = 8          # submitted range zero-padding (NFFT = 8M)
DOPPLER_PAD = 4


# =============================================================================
# 1. Signal models
# =============================================================================
def legacy_truth():
    """Signal model of the submitted estimation_crb.py (for L0/L1 only)."""
    L = mc.R2_RIS_T                                             # tau = R/c, R = 10 m
    fd = 2.0 * mc.V_TARGET / mc.LAMBDA * np.cos(np.deg2rad(mc.THETA1_DEG - mc.HEADING_DEG))
    return L, fd


def sample_times(m, n):
    """Centred sample instants t_kp = (p-(N-1)/2) Tsw + (k-(M+1)/2) Tsym."""
    tp = (np.arange(n) - (n - 1) / 2.0) * mc.TSW
    tk = (np.arange(1, m + 1) - (m + 1) / 2.0) * mc.TSYM
    return tp, tk


def simulate_legacy(rng, B, snr1_db, nsw):
    m = mc.M_TONES
    L, fd = legacy_truth()
    k = np.arange(1, m + 1)
    p = np.arange(nsw)
    s = (np.exp(-1j * 2 * np.pi * k * mc.DELTA_F * L / mc.C_LIGHT)[:, None]
         * np.exp(1j * 2 * np.pi * fd * p * mc.TSW)[None, :])
    w = (rng.standard_normal((B, m, nsw)) + 1j * rng.standard_normal((B, m, nsw))) / np.sqrt(2)
    Y = np.sqrt(10 ** (snr1_db / 10)) * s[None] + w
    return Y, np.full(B, L), np.full(B, fd * mc.LAMBDA / 2)


def simulate_physical(rng, B, snr1_db, nsw, nu, p_rx, gains=None, cell_halfwidth=None):
    """Physical bistatic model.  Truth is the mid-CPI path L and the carrier
    Doppler; each trial draws the path uniformly within one submitted range
    grid cell and the path rate uniformly within one Doppler bin, so grid
    effects are averaged honestly rather than evaluated at a lucky point."""
    m = nu.size
    L0 = mc.bistatic_path(mc.P_TARGET, p_rx)
    Ld0 = mc.bistatic_path_rate(mc.P_TARGET, p_rx)
    if cell_halfwidth is None:
        cell_halfwidth = mc.C_LIGHT / (NFFT_R_FACTOR * m * mc.DELTA_F) / 2
    L = L0 + rng.uniform(-cell_halfwidth, cell_halfwidth, B)
    Ld = Ld0 + rng.uniform(-0.5, 0.5, B) * mc.LAMBDA / (nsw * mc.TSW)
    tp, tk = sample_times(m, nsw)
    t = tk[:, None] + tp[None, :]                                          # (M, N)
    phi0 = rng.uniform(0, 2 * np.pi, B)
    ph = (phi0[:, None, None]
          - 2 * np.pi * nu[None, :, None] * L[:, None, None] / mc.C_LIGHT
          - 2 * np.pi * (mc.FC + nu[None, :, None]) * Ld[:, None, None] * t[None] / mc.C_LIGHT)
    amp = np.sqrt(10 ** (snr1_db / 10))
    s = amp * np.exp(1j * ph)
    if gains is not None:
        s = s * gains[:, :, None]
    w = (rng.standard_normal((B, m, nsw)) + 1j * rng.standard_normal((B, m, nsw))) / np.sqrt(2)
    v_true = -Ld / 2.0                    # paper convention v = lambda f_D / 2
    return s + w, L, v_true


# =============================================================================
# 2. Estimators
# =============================================================================
def _parabolic(y_m1, y0, y_p1):
    den = y_m1 - 2 * y0 + y_p1
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.where(np.abs(den) > 0, 0.5 * (y_m1 - y_p1) / den, 0.0)
    return np.clip(d, -0.5, 0.5)


def est_legacy(Y, nfft_r):
    """Exact algorithm of the submitted monte_carlo_estimation()."""
    B, m, n = Y.shape
    rp = np.fft.ifft(Y, n=nfft_r, axis=1)                              # (B, NFFT, N)
    pk = np.argmax(np.mean(np.abs(rp) ** 2, axis=2), axis=1)
    L_hat = pk * mc.C_LIGHT / (nfft_r * mc.DELTA_F)
    seq = rp[np.arange(B), pk, :]
    D = np.fft.fftshift(np.fft.fft(seq, n=n, axis=1), axes=1)
    iv = np.argmax(np.abs(D), axis=1)
    dv = mc.LAMBDA / (2 * n * mc.TSW)
    v_hat = (iv - n // 2) * dv
    return L_hat, v_hat


def _doppler_newton(Y, t, fk_scale, f0, iters=4, max_step=None):
    """Newton ascent of S(f) = sum_k |sum_p y_kp exp(-j2pi f_k t_kp)|^2,
    f_k = f (fc + nu_k)/fc.  Falls back to the start value where S'' >= 0."""
    f = f0.copy()
    tt = t[None, :, :] * fk_scale[None, :, None]                         # (1, M, N)
    for _ in range(iters):
        e = np.exp(-1j * 2 * np.pi * f[:, None, None] * tt)
        z0 = np.sum(Y * e, axis=2)
        z1 = np.sum(Y * e * (-1j * 2 * np.pi * tt), axis=2)
        z2 = np.sum(Y * e * (-(2 * np.pi * tt) ** 2), axis=2)
        d1 = np.sum(2 * np.real(np.conj(z0) * z1), axis=1)
        d2 = np.sum(2 * np.real(np.conj(z1) * z1 + np.conj(z0) * z2), axis=1)
        step = np.where(d2 < 0, -d1 / np.where(d2 < 0, d2, -1.0), 0.0)
        if max_step is not None:
            step = np.clip(step, -max_step, max_step)
        f = f + step
    return f


def _range_newton(c, nu, L0, iters=4, max_step=None):
    """Newton ascent of |sum_k c_k exp(+j2pi nu_k L/c)|^2."""
    L = L0.copy()
    a = 2 * np.pi * nu / mc.C_LIGHT
    for _ in range(iters):
        e = np.exp(1j * a[None, :] * L[:, None])
        z0 = np.sum(c * e, axis=1)
        z1 = np.sum(c * e * (1j * a[None, :]), axis=1)
        z2 = np.sum(c * e * (-(a[None, :]) ** 2), axis=1)
        d1 = 2 * np.real(np.conj(z0) * z1)
        d2 = 2 * np.real(np.conj(z1) * z1 + np.conj(z0) * z2)
        step = np.where(d2 < 0, -d1 / np.where(d2 < 0, d2, -1.0), 0.0)
        if max_step is not None:
            step = np.clip(step, -max_step, max_step)
        L = L + step
    return L


def est_coherent(Y, nu, nfft_r=None, want=('argmax', 'parabolic', 'ml')):
    """Coherent 2-D processing: (i) Doppler from the tone-averaged slow-time
    spectrum (Eq. 38), (ii) Doppler-compensated coherent sum per tone with the
    centred sample times (compensates intra-sweep timing and tone-dependent
    Doppler), (iii) range from the tone phasors.  Returns dict of estimates at
    each refinement level, plus the tone phasors of the ML level."""
    B, m, n = Y.shape
    nfft_r = NFFT_R_FACTOR * m if nfft_r is None else nfft_r
    tp, tk = sample_times(m, n)
    t = tk[:, None] + tp[None, :]
    fk_scale = (mc.FC + nu) / mc.FC

    nd = DOPPLER_PAD * n
    D = np.fft.fft(Y, n=nd, axis=2)
    S = np.sum(np.abs(D) ** 2, axis=1)                                # (B, nd)
    fgrid = np.fft.fftfreq(nd, mc.TSW)
    dfg = 1.0 / (nd * mc.TSW)
    i0 = np.argmax(S, axis=1)
    ib = np.arange(B)
    out = {}
    # plain N-point FFT (every DOPPLER_PAD-th bin) -- the submitted resolution
    iN = np.argmax(S[:, ::DOPPLER_PAD], axis=1) * DOPPLER_PAD
    out['v_fft'] = fgrid[iN] * mc.LAMBDA / 2
    f_arg = fgrid[i0]
    dpar = _parabolic(S[ib, (i0 - 1) % nd], S[ib, i0], S[ib, (i0 + 1) % nd])
    f_par = f_arg + dpar * dfg
    f_ml = _doppler_newton(Y, t, fk_scale, f_par, iters=4, max_step=dfg)
    fsel = {'argmax': f_arg, 'parabolic': f_par, 'ml': f_ml}

    cell = mc.C_LIGHT / (nfft_r * mc.DELTA_F)
    for lev in want:
        f = fsel[lev]
        e = np.exp(-1j * 2 * np.pi * f[:, None, None] * fk_scale[None, :, None] * t[None])
        c = np.mean(Y * e, axis=2)                                     # (B, M)
        P = np.abs(np.fft.ifft(c, n=nfft_r, axis=1)) ** 2
        k0 = np.argmax(P, axis=1)
        L_arg = k0 * cell
        if lev == 'argmax':
            L_hat = L_arg
        else:
            d = _parabolic(P[ib, (k0 - 1) % nfft_r], P[ib, k0], P[ib, (k0 + 1) % nfft_r])
            L_hat = L_arg + d * cell
            if lev == 'ml':
                L_hat = _range_newton(c, nu, L_hat, iters=4, max_step=cell)
        out[f'L_{lev}'] = L_hat
        out[f'v_{lev}'] = f * mc.LAMBDA / 2
    return out


def range_zero_pad_sweep(Y, nu, factors):
    """Coherent (ML-Doppler) tone phasors, then range by argmax and by
    parabolic interpolation for each IFFT length NFFT = factor * M."""
    B, m, n = Y.shape
    tp, tk = sample_times(m, n)
    t = tk[:, None] + tp[None, :]
    fk_scale = (mc.FC + nu) / mc.FC
    nd = DOPPLER_PAD * n
    S = np.sum(np.abs(np.fft.fft(Y, n=nd, axis=2)) ** 2, axis=1)
    fgrid = np.fft.fftfreq(nd, mc.TSW)
    dfg = 1.0 / (nd * mc.TSW)
    i0 = np.argmax(S, axis=1)
    ib = np.arange(B)
    f = fgrid[i0] + _parabolic(S[ib, (i0 - 1) % nd], S[ib, i0], S[ib, (i0 + 1) % nd]) * dfg
    f = _doppler_newton(Y, t, fk_scale, f, iters=4, max_step=dfg)
    e = np.exp(-1j * 2 * np.pi * f[:, None, None] * fk_scale[None, :, None] * t[None])
    c = np.mean(Y * e, axis=2)
    res = {}
    for fac in factors:
        nf = int(fac * m)
        cell = mc.C_LIGHT / (nf * mc.DELTA_F)
        P = np.abs(np.fft.ifft(c, n=nf, axis=1)) ** 2
        k0 = np.argmax(P, axis=1)
        d = _parabolic(P[ib, (k0 - 1) % nf], P[ib, k0], P[ib, (k0 + 1) % nf])
        res[fac] = (k0 * cell, (k0 + d) * cell)
    return res


def angle_mc(rng, n, snr1_db, nrx=mc.NRX, grid_deg=0.5, span_deg=1.0):
    """Rx-ULA AoA (relative to broadside).  Snapshot = phase-aligned sum over
    all M tones and N sweeps (Eq. 30), i.e. per-element SNR = SNR1 M N / N_rx.
    Returns errors [rad] for grid beam scan, + parabolic, + Newton ML."""
    snr_el = mc.snr_snapshot_lin(snr1_db) / nrx
    phi = np.deg2rad(rng.uniform(-span_deg, span_deg, n))
    l = np.arange(nrx)
    x = (np.sqrt(snr_el) * np.exp(1j * rng.uniform(0, 2 * np.pi, n))[:, None]
         * np.exp(1j * np.pi * l[None, :] * np.sin(phi)[:, None])
         + (rng.standard_normal((n, nrx)) + 1j * rng.standard_normal((n, nrx))) / np.sqrt(2))
    g = np.deg2rad(np.arange(-60.0, 60.0 + 1e-9, grid_deg))
    A = np.exp(1j * np.pi * l[:, None] * np.sin(g)[None, :])
    P = np.abs(np.conj(x) @ A) ** 2
    i0 = np.argmax(P, axis=1)
    ib = np.arange(n)
    phi_g = g[i0]
    im, ip = np.clip(i0 - 1, 0, g.size - 1), np.clip(i0 + 1, 0, g.size - 1)
    phi_p = phi_g + _parabolic(P[ib, im], P[ib, i0], P[ib, ip]) * np.deg2rad(grid_deg)
    th = phi_p.copy()
    for _ in range(4):
        a = np.pi * l[None, :] * np.cos(th)[:, None]
        e = np.exp(-1j * np.pi * l[None, :] * np.sin(th)[:, None])
        z0 = np.sum(x * e, axis=1)
        d_ph = -1j * a
        dd_ph = 1j * np.pi * l[None, :] * np.sin(th)[:, None]
        z1 = np.sum(x * e * d_ph, axis=1)
        z2 = np.sum(x * e * (d_ph ** 2 + dd_ph), axis=1)
        d1 = 2 * np.real(np.conj(z0) * z1)
        d2 = 2 * np.real(np.conj(z1) * z1 + np.conj(z0) * z2)
        step = np.where(d2 < 0, -d1 / np.where(d2 < 0, d2, -1.0), 0.0)
        th = th + np.clip(step, -np.deg2rad(grid_deg), np.deg2rad(grid_deg))
    return {'grid': phi_g - phi, 'parabolic': phi_p - phi, 'ml': th - phi}


# =============================================================================
# 3. Statistics
# =============================================================================
def wrap_err(L_hat, L_true):
    P = mc.path_ambiguity()
    return (L_hat - L_true + P / 2) % P - P / 2


def decompose(err, thr=OUTLIER_M):
    """MSE = p_in*bias_in^2 + p_in*var_in + p_out*E[e^2 | outlier]."""
    err = np.asarray(err)
    mse = float(np.mean(err ** 2))
    inl = np.abs(err) <= thr
    p_in = float(np.mean(inl))
    if inl.any():
        b = float(np.mean(err[inl]))
        v = float(np.var(err[inl]))
    else:
        b = v = 0.0
    out_c = float(np.sum(err[~inl] ** 2) / err.size)
    return {'rmse': np.sqrt(mse), 'bias': float(np.mean(err)), 'std': float(np.std(err)),
            'p_out': 1 - p_in, 'mse': mse, 'mse_bias': p_in * b ** 2,
            'mse_var': p_in * v, 'mse_out': out_c}


def run_batches(n_trials, batch, fn):
    """Accumulate per-trial arrays returned by fn(B) over batches."""
    acc = {}
    done = 0
    while done < n_trials:
        B = min(batch, n_trials - done)
        for k, v in fn(B).items():
            acc.setdefault(k, []).append(np.asarray(v))
        done += B
    return {k: np.concatenate(v) for k, v in acc.items()}


def child_rng(*keys):
    return np.random.default_rng([mc.RANDOM_SEED, *[int(round(k * 100)) + 100000 for k in keys]])


# =============================================================================
# 4. Figures
# =============================================================================




def _ladder_axes(ax, rows, labels, title):
    from matplotlib.patches import Patch
    y = np.arange(len(rows))[::-1]
    cb, cv, co = '#c0392b', '#1f5fa8', '#9a9a9a'
    for yi, r in zip(y, rows):
        tot = r['ratio']
        fr = np.array([r['mse_bias'], r['mse_var'], r['mse_out']]) / max(r['mse'], 1e-30)
        left = 0.0
        for f_, c_ in zip(fr, (cb, cv, co)):
            ax.barh(yi, tot * f_, left=left, color=c_, height=0.62, edgecolor='w', lw=0.3)
            left += tot * f_
        ax.text(tot + 0.08, yi, '%.2f cm (%.2f$\\times$)' % (r['rmse'] * 100, tot), va='center', fontsize=5.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.axvline(1.0, color='k', lw=0.6, ls='--')
    ax.set_xlabel('RMSE / bound')
    ax.set_xlim(0, max(r['ratio'] for r in rows) * 1.62)
    ax.legend(handles=[Patch(color=cb, label='bias$^2$'), Patch(color=cv, label='variance'),
                       Patch(color=co, label='outliers')], fontsize=5.6, loc='center right', ncol=1)
    ax.set_title(title, fontsize=7)
    ax.grid(True, axis='x')




def fig_gap_attribution(out, ladder):
    """Response-letter figure: the full ladder L0-L6 from the submitted estimator."""
    labels = ['L%d  %s' % (i, r['label']) for i, r in enumerate(ladder)]
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 2.5))
    _ladder_axes(ax, ladder, labels, 'Attribution of the submitted RMSE/CRB gap (one change per row)')
    return mc.save_fig(fig, out, 'FR1_gap_attribution')


def fig_geometry(out, beta_sel, sig_L, sig_th):
    betas = np.linspace(5, 172, 300)
    crb, dpdl = [], []
    for b in betas:
        rx = mc.rx_position(b)
        crb.append(mc.crb_position(sig_L, sig_th, mc.P_TARGET, rx))
        J = mc.position_jacobian(mc.P_TARGET, rx)
        dpdl.append(np.linalg.norm(J[:, 0]))
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 2.0))
    ax.semilogy(betas, np.array(crb) * 100, color=mc.COLORS['crb'], label='position CRB')
    ax2 = ax.twinx()
    ax2.semilogy(betas, dpdl, color='0.5', ls='--', lw=0.8, label='$|\\partial \\mathbf{p}/\\partial L|$')
    for b, c, t in ((mc.BETA_LEGACY_DEG, mc.COLORS['accent'], 'submitted'), (beta_sel, mc.COLORS['green'], 'revised')):
        v = mc.crb_position(sig_L, sig_th, mc.P_TARGET, mc.rx_position(b)) * 100
        ax.plot(b, v, 'o', color=c, ms=4)
        ax.annotate('%s\n$\\beta$=%.0f$^\\circ$: %.1f cm' % (t, b, v), (b, v),
                    xytext=(-62, 6) if b > 100 else (6, -24),
                    textcoords='offset points', fontsize=6, color=c)
    ax.set_xlabel('Bistatic angle at target $\\beta$ (deg)')
    ax.set_ylabel('Position CRB (cm)')
    ax2.set_ylabel('$|\\partial\\mathbf{p}/\\partial L|$', fontsize=7)
    ax.grid(True, which='both')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5.8, loc='upper left')
    return mc.save_fig(fig, out, 'FA2_bistatic_geometry')


# =============================================================================
# 5. Tables
# =============================================================================
def _sb(x):
    """Signed 2-dp value with a clean zero (no '-0.00')."""
    return '0.00' if abs(x) < 0.005 else f'{x:+.2f}'


def _ladder_rows(rows, key, prefix):
    out = []
    for i, r in rows:
        out.append(f"{prefix}{i} & {r[key]} & {r['rmse']*100:.2f} & {_sb(r['bias']*100)} & "
                   f"{r['std']*100:.2f} & {r['p_out']*100:.1f} & {r['bound']*100:.2f} & {r['ratio']:.2f} \\\\")
    return out


def table_error_budget(out, ladder, mech):
    """Paper Table A: estimator variants (L2-L6) + hardware mechanisms."""
    L = _ladder_rows([(i + 1, r) for i, r in enumerate(ladder[2:])], 'paper_tex', 'E')
    Mr = []
    for r in mech:
        Mr.append(f"& {r['label_tex']} & {r['rmse']*100:.2f} & {_sb(r['bias']*100)} & "
                  f"{r['gain_loss_db']:.2f} & -- & {r['bound']*100:.2f} & {r['ratio']:.2f} \\\\")
    tex = r"""\begin{table}[!t]
\centering
\caption{Error Budget of the Bistatic Path Estimate at $\mathrm{SNR}_1=OPSNR$\,dB}
\label{tab:error_budget}
\footnotesize
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{@{}clcccccc@{}}
\hline\hline
 & Configuration & RMSE & Bias & Std & Out. & Bound & Ratio \\
 &  & (cm) & (cm) & (cm) & (\%) & (cm) &  \\
\hline
\multicolumn{8}{@{}l}{\emph{(a) Estimator (ideal hardware)}}\\
""" + "\n".join(L) + r"""
\hline
\multicolumn{8}{@{}l}{\emph{(b) Hardware, estimator E5; Std column = gain loss (dB)}}\\
""" + "\n".join(Mr) + r"""
\hline\hline
\multicolumn{8}{@{}p{0.97\columnwidth}@{}}{\scriptsize Bound: \eqref{eq:crb_R}, recomputed in (b) with the tone-resolved R-RIS gains. Rows of (b) share one noise stream (common random numbers), so they differ only in the hardware; that stream differs from (a), so the ideal row of (b) and E5 differ by Monte Carlo variation ($\pm$2.2\,\% standard error). Outliers: $|e|>0.5$\,m.}
\end{tabular}
\end{table}
"""
    Path(out).mkdir(parents=True, exist_ok=True)
    tex = tex.replace('OPSNR', f'{mc.SNR1_DB:.1f}')
    (Path(out) / 'TA_error_budget.tex').write_text(tex, encoding='utf-8')


def table_gap_attribution(out, ladder):
    """Response-letter table: the full ladder L0-L6."""
    L = _ladder_rows(list(enumerate(ladder)), 'label_tex', 'L')
    tex = r"""\begin{table}[!t]
\centering
\caption{Attribution of the submitted RMSE/CRB gap (response to R1.1)}
\footnotesize
\setlength{\tabcolsep}{1.5pt}
\begin{tabular}{@{}clcccccc@{}}
\hline\hline
 & Change & RMSE & Bias & Std & Out. & Bound & Ratio \\
 &  & (cm) & (cm) & (cm) & (\%) & (cm) &  \\
\hline
""" + "\n".join(L) + r"""
\hline\hline
\end{tabular}
\end{table}
"""
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'TR1_gap_attribution.tex').write_text(tex, encoding='utf-8')


def table_estimation_perf(out, R):
    rows = [
        ('Operating single-symbol SNR', f"${mc.SNR1_DB:.1f}$", 'dB'),
        ('Post-CPI SNR ($N_{\\mathrm{sw}}$=1024)', f"${mc.SNR_CPI_DB:.1f}$", 'dB'),
        ('Path resolution $c/(M\\Delta f)$', f"{mc.path_resolution():.2f}", 'm'),
        ('Unambiguous path $c/\\Delta f$', f"{mc.path_ambiguity():.2f}", 'm'),
        ('Velocity resolution $\\lambda/(2T_{\\mathrm{CPI}})$', f"{mc.LAMBDA/(2*mc.TCPI)*100:.2f}", 'cm/s'),
        ('CRB path $\\sqrt{\\mathrm{CRB}_L}$, \\eqref{eq:crb_R}', f"{R['crb_L']*100:.2f}", 'cm'),
        ('CRB velocity', f"{R['crb_v']*100:.2f}", 'cm/s'),
        ('CRB angle', f"{np.degrees(R['crb_th']):.3f}", 'deg'),
        (f"CRB position ($\\beta={R['beta']:.0f}^\\circ$)", f"{R['crb_pos']*100:.2f}", 'cm'),
        ('RMSE path: argmax / refined', f"{R['L_argmax']*100:.2f} / {R['L_ml']*100:.2f}", 'cm'),
        ('RMSE velocity: FFT / refined', f"{R['v_fft']*100:.2f} / {R['v_ml']*100:.2f}", 'cm/s'),
        ('RMSE angle: 0.5$^\\circ$ scan / refined', f"{np.degrees(R['th_grid']):.3f} / {np.degrees(R['th_ml']):.3f}", 'deg'),
        ('RMSE position: coarse / refined', f"{R['pos_coarse']*100:.2f} / {R['pos_ml']*100:.2f}", 'cm'),
        ('Efficiency RMSE/CRB (path, refined)', f"{R['L_ml']/R['crb_L']:.2f}", '--'),
        ('MC trials per SNR point', f"{R['n_trials']:d}", '--'),
    ]
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
    tex = r"""\begin{table}[!t]
\centering
\caption{Estimation Performance Summary at the Operating Point}
\label{tab:estimation_perf}
\footnotesize
\begin{tabular}{@{}lrl@{}}
\hline\hline
Metric & Value & Units \\
\hline
""" + body + r"""
\hline\hline
\multicolumn{3}{@{}p{0.95\columnwidth}@{}}{\scriptsize Operating point $\mathrm{SNR}_1=OPSNR$\,dB, $N_{\mathrm{sw}}=1024$, $M=12$; refined = coherent 2-D processing with ML peak refinement (Section~\ref{sec:efficiency}).}
\end{tabular}
\end{table}
"""
    Path(out).mkdir(parents=True, exist_ok=True)
    tex = tex.replace('OPSNR', f'{mc.SNR1_DB:.1f}')
    (Path(out) / 'T4rev_estimation_performance.tex').write_text(tex, encoding='utf-8')


# =============================================================================
# 6. Main
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description='estimator efficiency')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'exp02_estimation_crb'))
    ap.add_argument('--quick', action='store_true', help='200 trials, coarse SNR grid')
    ap.add_argument('--trials', type=int, default=None)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--beta', type=float, default=mc.BETA_DEFAULT_DEG, help='Rx bistatic angle (deg)')
    ap.add_argument('--f-off', type=float, default=0.0, help='tone offset in MHz (Phase 1 decision D1)')
    ap.add_argument('--skip-legacy-check', action='store_true')
    ap.add_argument('--snr1', type=float, default=None,
                    help='override the operating single-symbol SNR in dB (gate decision D0)')
    args = ap.parse_args(argv)
    if args.snr1 is not None:
        mc.SNR1_DB = float(args.snr1)
        mc.SNR_CPI_DB = mc.SNR1_DB + 10.0 * np.log10(mc.NSW)

    t0 = time.time()
    mc.set_ieee_style()
    out = Path(args.out)
    fdir, tdir, ddir = out / 'figures', out / 'tables', out / 'data'
    for d in (fdir, tdir, ddir):
        d.mkdir(parents=True, exist_ok=True)
    ntr = args.trials or (200 if args.quick else 1000)
    B = args.batch
    nu = mc.tone_offsets(args.f_off * 1e6)
    p_rx = mc.rx_position(args.beta)
    op = mc.SNR1_DB
    log = []

    def say(s=''):
        print(s, flush=True)
        log.append(s)

    say('=' * 74)
    say(' Estimator efficiency and the RMSE/CRB gap')
    say('=' * 74)
    legacy_dir = mc.find_legacy_dir(__file__)
    mc.check_against_legacy_config(legacy_dir=legacy_dir)
    say(f' SNR1 = {op:.1f} dB   trials/point = {ntr}   beta = {args.beta:.1f} deg   f_off = {args.f_off:.0f} MHz'
        f'   Rx = ({p_rx[0]:.2f}, {p_rx[1]:.2f}) m')

    # ---- 0. exact reproduction of the submitted Table IV (if legacy code present)
    legacy_exact = None
    if legacy_dir is not None and not args.skip_legacy_check:
        say('\n[0] Exact reproduction with the submitted estimation_crb.py')
        with mc._InDir(legacy_dir):
            import config as lcfg
            from estimation_crb import compute_crb, monte_carlo_estimation
            s = lcfg.SNR_SWEEP_DB
            cr = compute_crb(s, n_sweeps=lcfg.NSw)
            i = cr['snr_op_idx']
            mm = monte_carlo_estimation(s, n_trials=lcfg.MC_TRIALS, n_sweeps=max(256, lcfg.NSw // 4))
        legacy_exact = {'snr_used_db': float(s[i]), 'crb_r_cm': float(cr['crb_r_m'][i] * 100),
                        'rmse_r_cm': float(mm['rmse_r_m'][i] * 100), 'rmse_v_cms': float(mm['rmse_v_ms'][i] * 100),
                        'crb_pos_cm': float(cr['crb_pos_m'][i] * 100), 'rmse_pos_cm': float(mm['rmse_pos_m'][i] * 100)}
        e = legacy_exact
        say(f"    op-point grid value used for Table IV : {e['snr_used_db']:.1f} dB (text states -14.5 dB)")
        say(f"    CRB_R {e['crb_r_cm']:.3f} cm | RMSE_R {e['rmse_r_cm']:.3f} cm | ratio {e['rmse_r_cm']/e['crb_r_cm']:.3f}")
        say(f"    RMSE_v {e['rmse_v_cms']:.2f} cm/s | CRB_pos {e['crb_pos_cm']:.3f} | RMSE_pos {e['rmse_pos_cm']:.3f} cm")
        mc.set_ieee_style()        # the legacy module overrides rcParams on import
    else:
        say('\n[0] submitted estimation_crb.py not found -- exact reproduction skipped')

    # ---- 1. bounds -----------------------------------------------------------
    crb_L = float(mc.crb_path(op, nu=nu))
    crb_v = float(mc.crb_radial_velocity(op))
    crb_th = float(mc.crb_angle_ula(mc.snr_snapshot_lin(op)))
    crb_pos = float(mc.crb_position(crb_L, crb_th, mc.P_TARGET, p_rx))
    crb_L_leg = float(mc.crb_path_legacy(-14.0))
    say('\n[1] Bounds at the operating point')
    say(f'    path  : Eq.(37) {crb_L*100:.3f} cm   (submitted implementation: {mc.crb_path_legacy(op)*100:.3f} cm'
        f' at {op:.1f} dB, {crb_L_leg*100:.3f} cm at -14 dB)')
    say(f'    vel.  : {crb_v*100:.3f} cm/s  (submitted: {mc.crb_radial_velocity_legacy(-14)*100:.3f} cm/s)')
    say(f'    angle : {np.degrees(crb_th):.4f} deg  (submitted table: 0.1895 deg)')
    J = mc.position_jacobian(mc.P_TARGET, p_rx)
    say(f'    pos.  : {crb_pos*100:.3f} cm via Jacobian |dp/dL| = {np.linalg.norm(J[:,0]):.2f},'
        f' |dp/dtheta| = {np.linalg.norm(J[:,1]):.2f} m/rad')
    crb_pos_leg = float(mc.crb_position(crb_L, crb_th, mc.P_TARGET, mc.rx_position(mc.BETA_LEGACY_DEG)))
    say(f'    pos. at the submitted Rx placement (beta = {mc.BETA_LEGACY_DEG} deg): {crb_pos_leg*100:.2f} cm')

    # ---- 2. ladder -------------------------------------------------------------
    say('\n[2] Ladder at the operating point (one change per step)')
    ladder = []

    def add(label, label_tex, err_L, err_v, bound, paper=None):
        d = decompose(err_L)
        d.update(label=label, label_tex=label_tex, bound=bound, ratio=d['rmse'] / bound,
                 paper=paper or '', paper_tex=(paper or '').replace('N_sw', '$N_{\\mathrm{sw}}$'),
                 v_rmse=float(np.sqrt(np.mean(np.asarray(err_v) ** 2))))
        ladder.append(d)
        say(f"    L{len(ladder)-1}  {label:30s} RMSE {d['rmse']*100:6.2f} cm  bias {d['bias']*100:+6.2f}"
            f"  std {d['std']*100:5.2f}  out {d['p_out']*100:4.1f}%  bound {bound*100:5.2f}"
            f"  ratio {d['ratio']:5.2f}  | v-RMSE {d['v_rmse']*100:6.2f} cm/s")

    def legacy_fn(snr, nsw, seedkey):
        rng = child_rng(*seedkey)

        def f(Bn):
            Y, L, v = simulate_legacy(rng, Bn, snr, nsw)
            Lh, vh = est_legacy(Y, NFFT_R_FACTOR * mc.M_TONES)
            return {'eL': Lh - L, 'ev': vh - v}
        return f

    r = run_batches(ntr, B, legacy_fn(-14.0, 256, (1, 0)))
    add('as submitted', 'As submitted', r['eL'], r['ev'], crb_L_leg)
    r = run_batches(ntr, B, legacy_fn(op, 256, (1, 1)))
    add('+ corrected bound, -14.5 dB', '+ Eq.~(37) bound, $-14.5$\\,dB', r['eL'], r['ev'], crb_L)

    def phys_fn(snr, nsw, seedkey, estimator, gains_fn=None, hw_seedkey=None):
        rng = child_rng(*seedkey)
        # separate stream for hardware draws -> noise/truth streams are identical
        # across mechanisms (common random numbers), isolating the mechanism effect
        hw_rng = child_rng(*(hw_seedkey or (99, *seedkey)))

        def f(Bn):
            g = gains_fn(hw_rng, Bn) if gains_fn else None
            Y, L, v = simulate_physical(rng, Bn, snr, nsw, nu, p_rx, gains=g)
            res = {}
            if estimator == 'legacy':
                Lh, vh = est_legacy(Y, NFFT_R_FACTOR * mc.M_TONES)
                res.update(eL=wrap_err(Lh, L), ev=vh - v)
            else:
                o = est_coherent(Y, nu)
                for lev in ('argmax', 'parabolic', 'ml'):
                    res[f'eL_{lev}'] = wrap_err(o[f'L_{lev}'], L)
                    res[f'ev_{lev}'] = o[f'v_{lev}'] - v
                res['ev_fft'] = o['v_fft'] - v
            if g is not None:
                w = np.abs(g) ** 2
                res['crb2'] = np.array([mc.crb_path(snr, nu=nu, weights=wi) ** 2 for wi in w])
                res['gain'] = np.mean(w, axis=1)
            return res
        return f

    r = run_batches(ntr, B, phys_fn(op, 256, (2, 0), 'legacy'))
    add('+ physical signal model', '+ physical signal model', r['eL'], r['ev'], crb_L,
        paper='sweep-power avg., N_sw=256')
    r = run_batches(ntr, B, phys_fn(op, mc.NSW, (2, 1), 'legacy'))
    add('+ full CPI (1024 sweeps)', '+ full CPI ($N_{\\mathrm{sw}}$=1024)', r['eL'], r['ev'], crb_L,
        paper='sweep-power avg., N_sw=1024')
    r = run_batches(ntr, B, phys_fn(op, mc.NSW, (2, 2), 'coherent'))
    add('+ coherent 2-D processing', '+ coherent 2-D processing', r['eL_argmax'], r['ev_argmax'], crb_L,
        paper='coherent 2-D, argmax')
    add('+ parabolic interpolation', '+ peak interpolation', r['eL_parabolic'], r['ev_parabolic'], crb_L,
        paper='coherent 2-D + interpolation')
    add('+ ML refinement', '+ ML refinement', r['eL_ml'], r['ev_ml'], crb_L,
        paper='coherent 2-D + ML refinement')
    op_coh = r

    # ---- 3. hardware mechanisms --------------------------------------------------
    say('\n[3] Hardware mechanisms with the L6 estimator (tone-resolved R-RIS gains)')
    th1 = np.deg2rad(mc.THETA1_DEG)
    fabs = mc.FC + nu

    def gains_factory(bits, s_amp, s_ph_deg, s_point_deg):
        def g(rng, Bn):
            out = np.empty((Bn, nu.size), complex)
            for b in range(Bn):
                st = th1 + np.deg2rad(s_point_deg) * rng.standard_normal() if s_point_deg > 0 else th1
                out[b] = mc.rris_tone_gains(fabs, th1, st, bits=bits, sigma_amp=s_amp,
                                            sigma_phase_rad=np.deg2rad(s_ph_deg), rng=rng)
            return out
        return g

    # (label, LaTeX label, bits, sigma_amp, sigma_phase_deg, sigma_pointing_deg, stressed)
    mechs = [('ideal (continuous phase)', 'Ideal (continuous phase)', None, 0, 0, 0, False),
             ('1-bit', '1-bit phase quantisation', 1, 0, 0, 0, True),
             ('2-bit', '2-bit', 2, 0, 0, 0, False),
             ('3-bit (nominal)', '3-bit (nominal)', 3, 0, 0, 0, False),
             ('3-bit + spread 2%/3deg', '+ spread 2\\,\\%/3$^\\circ$', 3, 0.02, 3.0, 0, False),
             ('3-bit + spread 10%/15deg', '+ spread 10\\,\\%/15$^\\circ$', 3, 0.10, 15.0, 0, True),
             ('3-bit + pointing 0.5deg', '+ pointing 0.5$^\\circ$', 3, 0, 0, 0.5, False),
             ('3-bit + pointing 2deg', '+ pointing 2$^\\circ$', 3, 0, 0, 2.0, True),
             ('combined nominal', 'Combined nominal', 3, 0.02, 3.0, 0.5, False)]
    mech = []
    for i, (lab, labt, bits, sa, sp, spt, stressed) in enumerate(mechs):
        r = run_batches(ntr, B, phys_fn(op, mc.NSW, (3, 0), 'coherent', gains_factory(bits, sa, sp, spt),
                                        hw_seedkey=(30, i)))
        d = decompose(r['eL_ml'])
        bnd = float(np.sqrt(np.mean(r['crb2'])))
        d.update(label=lab, label_tex=labt, bound=bnd, ratio=d['rmse'] / bnd, stressed=stressed,
                 gain_loss_db=float(-10 * np.log10(np.mean(r['gain']))))
        mech.append(d)
        say(f"    {lab:26s} gain loss {d['gain_loss_db']:6.2f} dB  RMSE {d['rmse']*100:5.2f} cm"
            f"  bias {d['bias']*100:+5.2f}  bound {bnd*100:5.2f}  ratio {d['ratio']:.3f}"
            f"{'  [stress]' if stressed else ''}")

    # ---- 4. zero padding -----------------------------------------------------------
    say('\n[4] Zero-padding sweep (coherent processing, ML Doppler)')
    factors = [1, 2, 4, 8, 16, 32, 64]
    rng = child_rng(4, 0)

    # truth uniform over the COARSEST grid cell (N_FFT = M -> 1 m), so every
    # IFFT length sees uniformly distributed sub-bin offsets (otherwise
    # interpolation bias toward bin centres can masquerade as sub-CRB accuracy)
    half_1m = mc.path_resolution() / 2

    def zp_fn(Bn):
        Y, L, v = simulate_physical(rng, Bn, op, mc.NSW, nu, p_rx, cell_halfwidth=half_1m)
        res = range_zero_pad_sweep(Y, nu, factors)
        o = {}
        for fac, (la, lp) in res.items():
            o[f'a{fac}'] = wrap_err(la, L)
            o[f'p{fac}'] = wrap_err(lp, L)
        return o

    r = run_batches(ntr, B, zp_fn)
    zp = {fac: {'argmax': float(np.sqrt(np.mean(r[f'a{fac}'] ** 2))),
                'parabolic': float(np.sqrt(np.mean(r[f'p{fac}'] ** 2)))} for fac in factors}
    for fac in factors:
        say(f"    N_FFT = {fac:2d}M   argmax {zp[fac]['argmax']*100:7.2f} cm   parabolic {zp[fac]['parabolic']*100:6.2f} cm"
            f"   (CRB {crb_L*100:.2f})")

    # ---- 5. angle at the operating point ---------------------------------------------
    say('\n[5] Angle (Rx ULA, N_rx = 16, broadside toward the surveillance region)')
    ea = angle_mc(child_rng(5, 0), ntr, op)
    th_rm = {k: float(np.sqrt(np.mean(v ** 2))) for k, v in ea.items()}
    say(f"    0.5deg scan {np.degrees(th_rm['grid']):.4f}  + parabolic {np.degrees(th_rm['parabolic']):.4f}"
        f"  + ML {np.degrees(th_rm['ml']):.4f} deg   (CRB {np.degrees(crb_th):.4f})")

    # ---- 6. position at the operating point --------------------------------------------
    say('\n[6] Position through the bistatic inversion, Eqs. (40)-(42)')

    def pos_rmse(eL, eth, rx):
        L0 = mc.bistatic_path(mc.P_TARGET, rx)
        t0_ = mc.bearing(rx, mc.P_TARGET)
        p = mc.invert_position(L0 + eL, t0_ + eth, rx)
        return float(np.sqrt(np.mean(np.sum((p - mc.P_TARGET) ** 2, axis=1))))

    pos_c = pos_rmse(op_coh['eL_argmax'], ea['grid'], p_rx)
    pos_r = pos_rmse(op_coh['eL_ml'], ea['ml'], p_rx)
    rx_leg = mc.rx_position(mc.BETA_LEGACY_DEG)
    pos_r_leg = pos_rmse(op_coh['eL_ml'], ea['ml'], rx_leg)
    say(f'    beta = {args.beta:.0f} deg : coarse {pos_c*100:.2f} cm   refined {pos_r*100:.2f} cm   CRB {crb_pos*100:.2f} cm')
    say(f'    beta = {mc.BETA_LEGACY_DEG} deg (submitted placement): refined {pos_r_leg*100:.2f} cm   CRB {crb_pos_leg*100:.2f} cm')

    # ---- 7. SNR sweeps ----------------------------------------------------------------
    snr_grid = np.arange(-30, 21, 4 if args.quick else 2).astype(float)
    say(f'\n[7] SNR sweeps ({snr_grid.size} points x {ntr} trials)')
    cols = {k: [] for k in ('snr_db', 'crb_L', 'crb_v', 'crb_th', 'crb_pos', 'L_argmax', 'L_ml',
                            'v_fft', 'v_ml', 'th_grid', 'th_ml', 'pos_coarse', 'pos_ml')}
    for j, sdb in enumerate(snr_grid):
        r = run_batches(ntr, B, phys_fn(sdb, mc.NSW, (7, j), 'coherent'))
        a = angle_mc(child_rng(8, j), ntr, sdb)
        cL, cth = float(mc.crb_path(sdb, nu=nu)), float(mc.crb_angle_ula(mc.snr_snapshot_lin(sdb)))
        vals = dict(snr_db=sdb, crb_L=cL, crb_v=float(mc.crb_radial_velocity(sdb)), crb_th=cth,
                    crb_pos=float(mc.crb_position(cL, cth, mc.P_TARGET, p_rx)),
                    L_argmax=float(np.sqrt(np.mean(r['eL_argmax'] ** 2))),
                    L_ml=float(np.sqrt(np.mean(r['eL_ml'] ** 2))),
                    v_fft=float(np.sqrt(np.mean(r['ev_fft'] ** 2))),
                    v_ml=float(np.sqrt(np.mean(r['ev_ml'] ** 2))),
                    th_grid=float(np.sqrt(np.mean(a['grid'] ** 2))),
                    th_ml=float(np.sqrt(np.mean(a['ml'] ** 2))),
                    pos_coarse=pos_rmse(r['eL_argmax'], a['grid'], p_rx),
                    pos_ml=pos_rmse(r['eL_ml'], a['ml'], p_rx))
        for k, v in vals.items():
            cols[k].append(v)
        say(f"    {sdb:+5.0f} dB  path {vals['L_argmax']*100:8.2f}/{vals['L_ml']*100:7.2f} (CRB {cL*100:6.2f}) cm"
            f"   vel {vals['v_fft']*100:6.2f}/{vals['v_ml']*100:6.2f} cm/s   pos {vals['pos_ml']*100:8.2f} cm")
    sw = {k: np.array(v) for k, v in cols.items()}

    # ---- 8. outputs ------------------------------------------------------------------
    say('\n[8] Writing figures and tables')
    paths = [fig_rmse_path_position(fdir, sw, op, args.beta), fig_rmse_velocity(fdir, sw, op),
             fig_efficiency(fdir, ladder, zp, crb_L), fig_gap_attribution(fdir, ladder),
             fig_geometry(fdir, args.beta, crb_L, crb_th)]
    L6 = ladder[-1]
    R = dict(crb_L=crb_L, crb_v=crb_v, crb_th=crb_th, crb_pos=crb_pos, beta=args.beta,
             L_argmax=ladder[4]['rmse'], L_ml=L6['rmse'],
             v_fft=float(np.sqrt(np.mean(op_coh['ev_fft'] ** 2))), v_ml=L6['v_rmse'],
             th_grid=th_rm['grid'], th_ml=th_rm['ml'], pos_coarse=pos_c, pos_ml=pos_r, n_trials=ntr)
    table_error_budget(tdir, ladder, mech)
    table_gap_attribution(tdir, ladder)
    table_estimation_perf(tdir, R)
    for p in paths:
        say(f'    {p}')
    for tn in ('TA_error_budget.tex', 'TR1_gap_attribution.tex', 'T4rev_estimation_performance.tex'):
        say(f'    {tdir / tn}')
    with open(ddir / 'exp02_sweeps.csv', 'w', newline='') as fh:
        wtr = csv.writer(fh)
        wtr.writerow(list(sw.keys()))
        for row in zip(*sw.values()):
            wtr.writerow([f'{x:.6g}' for x in row])

    # ---- 9. gate -------------------------------------------------------------------
    say('\n[checks] Validation')
    hi = int(np.argmin(np.abs(sw['snr_db'] - 10)))
    mech_ratios = [m_['ratio'] for m_ in mech if not m_['stressed']]
    checks = [
        ('L0 replica within 10% of the submitted 7.14 cm', abs(ladder[0]['rmse'] * 100 - 7.14) / 7.14 < 0.10),
        ('refined path estimator efficient at +10 dB (RMSE/CRB < 1.15)', sw['L_ml'][hi] / sw['crb_L'][hi] < 1.15),
        ('refined path estimator at SNR1 within 1.25x of Eq. (37)', L6['ratio'] < 1.25),
        ('refined angle within 1.15x of exact CRB at SNR1', th_rm['ml'] / crb_th < 1.15),
        ('refined velocity within 1.25x of CRB at SNR1', L6['v_rmse'] / crb_v < 1.25),
        ('refined position within 1.25x of Jacobian CRB at SNR1', pos_r / crb_pos < 1.25),
        ('interpolated range reaches CRB (< 1.1x) for N_FFT >= 4M',
         all(zp[f]['parabolic'] / crb_L < 1.10 for f in factors if f >= 4)),
        ('non-stressed hardware changes RMSE/bound by < 10% vs ideal (common random numbers)',
         max(abs(x / mech[0]['ratio'] - 1) for x in mech_ratios) < 0.10),
    ]
    for n_, ok in checks:
        say(f"    [{'PASS' if ok else 'FAIL'}] {n_}")
    say(f'    time {time.time() - t0:.0f} s')

    res = dict(n_trials=ntr, beta_deg=args.beta, f_off_mhz=args.f_off, p_rx=p_rx.tolist(),
               snr1_db=float(op), snr_cpi_db=float(mc.SNR_CPI_DB),
               legacy_exact=legacy_exact,
               bounds=dict(crb_L_cm=crb_L * 100, crb_L_legacy_m14_cm=crb_L_leg * 100, crb_v_cms=crb_v * 100,
                           crb_theta_deg=float(np.degrees(crb_th)), crb_pos_cm=crb_pos * 100,
                           crb_pos_legacy_geometry_cm=crb_pos_leg * 100,
                           jacobian=J.tolist()),
               ladder=[{k: (v if isinstance(v, str) else float(v)) for k, v in d.items()} for d in ladder],
               mechanisms=[{k: (v if isinstance(v, (str, bool)) else float(v)) for k, v in d.items()} for d in mech],
               zero_padding=zp,
               angle_rmse_deg={k: float(np.degrees(v)) for k, v in th_rm.items()},
               position_cm=dict(coarse=pos_c * 100, refined=pos_r * 100, refined_legacy_geometry=pos_r_leg * 100),
               operating_point=R, gate2={n_: bool(o) for n_, o in checks})
    (ddir / 'exp02_results.json').write_text(json.dumps(res, indent=2, default=float), encoding='utf-8')
    (ddir / 'exp02_summary.txt').write_text('\n'.join(log), encoding='utf-8')
    return 0 if all(o for _, o in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
