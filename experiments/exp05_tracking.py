"""
exp05_tracking.py -- Beam tracking under manoeuvre
==================================================
Closed-loop tracking with the R-RIS illumination gain in the loop:
alpha-smoother, alpha-beta, CV/CA EKFs fusing the Doppler path rate, and an
IMM; straight-line, turning, accelerating and S-manoeuvre trajectories.

Paper figures produced: Fig. 10 (F9rev_tracking_cv), Fig. 11 (FG_tracking_manoeuvre)
Outputs: results/exp05_tracking/ (data, tables, extra figures), results/figdata/ (figure data),
         paper/figures/ (paper figures)
Run    : python experiments/exp05_tracking.py [--quick]
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
from figures.fig10_tracking_cv import fig_cv
from figures.fig11_tracking_manoeuvre import fig_manoeuvre
from experiments import exp02_estimation_crb as p2

logging.getLogger('fontTools').setLevel(logging.ERROR)

T = mc.TCPI
X16 = (np.arange(mc.N_X) - (mc.N_X - 1) / 2) * mc.D_ELEM
SNR_THRESHOLD_DB = -24.0     # refined-estimator threshold, Phase 2 (Fig. 7)
NAMES = ['alpha-smoother (submitted)', 'alpha-beta', 'CV-EKF', 'CA-EKF', 'IMM (CV+CA)']
SHORT = ['T0', 'T1', 'T2', 'T3', 'T4']


# =============================================================================
# Trajectories
# =============================================================================
def trajectory(kind, param, n_frames, v0=mc.V_TARGET, heading_deg=mc.HEADING_DEG, t_on=0.5, t_off=1.25, sub=20):
    """Position/velocity at the CPI centres.  Heading from +x (config convention)."""
    dt = T / sub
    p = mc.P_TARGET.astype(float).copy()
    h = np.deg2rad(heading_deg)
    v = v0
    P, V = [], []
    for k in range(n_frames):
        for s in range(sub):
            t = (k * sub + s) * dt
            a_t, w = 0.0, 0.0
            if kind == 'acc' and t_on <= t < t_off:
                a_t = param
            elif kind == 'turn' and t >= t_on:
                w = np.deg2rad(param)
            elif kind == 's' and t >= t_on:
                w = np.deg2rad(param) * np.sign(np.sin(2 * np.pi * (t - t_on) / 1.5) + 1e-12)
            v = v + a_t * dt
            h = h + w * dt
            p = p + v * np.array([np.cos(h), np.sin(h)]) * dt
            if s == sub // 2 - 1:
                P.append(p.copy())
                V.append(v * np.array([np.cos(h), np.sin(h)]))
    return np.array(P), np.array(V)


# =============================================================================
# Measurement model
# =============================================================================
def pointing_gain(th_true, th_cmd):
    """R-RIS illumination power gain toward th_true for beam(s) steered to th_cmd."""
    th_cmd = np.atleast_1d(th_cmd)
    ph = mc.K_C * X16[None, :] * (np.sin(th_cmd)[:, None] - np.sin(th_true))
    return np.abs(np.mean(np.exp(1j * ph), axis=1)) ** 2


class Geometry:
    def __init__(self, beta):
        self.p_rx = mc.rx_position(beta)
        self.th_b = mc.bearing(self.p_rx, mc.P_TARGET)            # Rx broadside
        self.R2_0 = np.linalg.norm(mc.P_TARGET)
        self.R3_0 = np.linalg.norm(mc.P_TARGET - self.p_rx)
        self.c0 = np.cos(np.deg2rad(mc.THETA1_DEG))

    def snr(self, p, g):
        R2, R3 = np.linalg.norm(p), np.linalg.norm(p - self.p_rx)
        th = mc.bearing(mc.P_RIS, p)
        return (mc.SNR1_DB + 10 * np.log10(np.maximum(g, 1e-12))
                + 20 * np.log10(self.R2_0 * self.R3_0 / (R2 * R3)) + 10 * np.log10(np.cos(th) / self.c0))

    def sigmas(self, p, snr):
        phi = mc.bearing(self.p_rx, p) - self.th_b
        sL = mc.crb_path(snr)
        sT = mc.crb_angle_ula(mc.snr_snapshot_lin(snr), phi_rad=phi)
        sLd = 2 * mc.crb_radial_velocity(snr)                     # path rate = 2 x radial velocity
        return np.atleast_1d(sL), np.atleast_1d(sT), np.atleast_1d(sLd)

    def measure(self, rng, p, v, th_cmd):
        B = th_cmd.size
        g = pointing_gain(mc.bearing(mc.P_RIS, p), th_cmd)
        snr = self.snr(p, g)
        sL, sT, sLd = self.sigmas(p, snr)
        L = mc.bistatic_path(p, self.p_rx)
        Ld = mc.bistatic_path_rate(p, self.p_rx, v)
        thr = mc.bearing(self.p_rx, p)
        Lh = L + sL * rng.standard_normal(B)
        Th = thr + sT * rng.standard_normal(B)
        Ldh = Ld + sLd * rng.standard_normal(B)
        fix = mc.invert_position(Lh, Th, self.p_rx)
        J = mc.position_jacobian(p, self.p_rx)
        Rp = (np.einsum('ij,bj,kj->bik', J, np.stack([sL ** 2, sT ** 2], 1), J))
        return fix, Ldh, Rp, sLd ** 2, g, snr


def ldot_jac(x, p_rx):
    """h3(x) = (u_rt - u_rx,t).v and its gradient wrt position and velocity."""
    p, v = x[:, 0:2], x[:, 2:4]
    d1 = p - mc.P_RIS
    r1 = np.linalg.norm(d1, axis=1, keepdims=True)
    u1 = d1 / r1
    d2 = p_rx[None, :] - p
    r2 = np.linalg.norm(d2, axis=1, keepdims=True)
    u2 = d2 / r2
    h = np.sum((u1 - u2) * v, axis=1)
    dp = (v - u1 * np.sum(u1 * v, 1, keepdims=True)) / r1 + (v - u2 * np.sum(u2 * v, 1, keepdims=True)) / r2
    return h, dp, (u1 - u2)


# =============================================================================
# Filters (vectorised over trials)
# =============================================================================
def mats(model, n, sig):
    I2 = np.eye(2)
    if model == 'cv':
        F = np.zeros((n, n))
        F[0:2, 0:2] = I2
        F[0:2, 2:4] = T * I2
        F[2:4, 2:4] = I2
        G = np.zeros((n, 2))
        G[0:2] = T ** 2 / 2 * I2
        G[2:4] = T * I2
        Q = sig ** 2 * G @ G.T
        if n == 6:
            Q[4:6, 4:6] = 1e-6 * np.eye(2)
    else:
        F = np.eye(6)
        F[0:2, 2:4] = T * I2
        F[0:2, 4:6] = T ** 2 / 2 * I2
        F[2:4, 4:6] = T * I2
        G = np.vstack([T ** 3 / 6 * I2, T ** 2 / 2 * I2, T * I2])
        Q = sig ** 2 * G @ G.T
    return F, Q


def ekf_update(x, P, fix, ldh, Rp, rld, p_rx):
    B, n = x.shape
    h3, dp, dv = ldot_jac(x, p_rx)
    H = np.zeros((B, 3, n))
    H[:, 0, 0] = 1
    H[:, 1, 1] = 1
    H[:, 2, 0:2] = dp
    H[:, 2, 2:4] = dv
    R = np.zeros((B, 3, 3))
    R[:, 0:2, 0:2] = Rp
    R[:, 2, 2] = rld
    y = np.concatenate([fix - x[:, 0:2], (ldh - h3)[:, None]], axis=1)
    S = H @ P @ np.transpose(H, (0, 2, 1)) + R
    K = np.transpose(np.linalg.solve(S, H @ P), (0, 2, 1))
    x = x + np.einsum('bnm,bm->bn', K, y)
    P = (np.eye(n)[None] - K @ H) @ P
    P = 0.5 * (P + np.transpose(P, (0, 2, 1)))
    sign, logdet = np.linalg.slogdet(S)
    maha = np.einsum('bi,bi->b', y, np.linalg.solve(S, y[..., None])[..., 0])
    lik = np.exp(-0.5 * (maha + logdet + 3 * np.log(2 * np.pi)))
    return x, P, np.maximum(lik, 1e-300)


class KF:
    def __init__(self, model, sig, x0, P0, p_rx):
        n = x0.shape[1]
        self.F, self.Q = mats(model, n, sig)
        self.x, self.P, self.p_rx = x0.copy(), P0.copy(), p_rx

    def step(self, fix, ldh, Rp, rld):
        self.x = self.x @ self.F.T
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x, self.P, _ = ekf_update(self.x, self.P, fix, ldh, Rp, rld, self.p_rx)
        return self.x[:, 0:2], self.x[:, 2:4], (self.x @ self.F.T)[:, 0:2]


class IMM:
    def __init__(self, sig_cv, sig_ca, x0, P0, p_rx, pi_stay=0.95):
        self.m = [mats('cv', 6, sig_cv), mats('ca', 6, sig_ca)]
        self.x = [x0.copy(), x0.copy()]
        self.P = [P0.copy(), P0.copy()]
        self.mu = np.tile([0.9, 0.1], (x0.shape[0], 1))
        self.Pi = np.array([[pi_stay, 1 - pi_stay], [1 - pi_stay, pi_stay]])
        self.p_rx = p_rx

    def step(self, fix, ldh, Rp, rld):
        c = self.mu @ self.Pi                                   # (B,2) predicted mode probs
        w = (self.mu[:, :, None] * self.Pi[None]) / c[:, None, :]  # w[b,i,j] = P(i | j)
        xs, Ps = [], []
        for j in range(2):
            x0 = sum(w[:, i, j][:, None] * self.x[i] for i in range(2))
            P0 = sum(w[:, i, j][:, None, None] * (self.P[i] + np.einsum('bi,bj->bij', self.x[i] - x0, self.x[i] - x0))
                     for i in range(2))
            xs.append(x0)
            Ps.append(P0)
        lik = []
        for j, (F, Q) in enumerate(self.m):
            x = xs[j] @ F.T
            P = F @ Ps[j] @ F.T + Q
            x, P, L = ekf_update(x, P, fix, ldh, Rp, rld, self.p_rx)
            self.x[j], self.P[j] = x, P
            lik.append(L)
        mu = c * np.stack(lik, 1)
        self.mu = mu / mu.sum(1, keepdims=True)
        xc = sum(self.mu[:, j][:, None] * self.x[j] for j in range(2))
        xp = sum(self.mu[:, j][:, None] * (self.x[j] @ self.m[j][0].T) for j in range(2))
        return xc[:, 0:2], xc[:, 2:4], xp[:, 0:2]


class AlphaSmoother:
    def __init__(self, fix0, alpha=0.3):
        self.a = alpha
        self.th = np.arctan2(fix0[:, 0], fix0[:, 1])

    def step(self, fix, *_):
        th_hat = np.arctan2(fix[:, 0], fix[:, 1])
        self.th = self.a * th_hat + (1 - self.a) * self.th
        r = np.linalg.norm(fix, axis=1)
        pos = np.stack([r * np.sin(self.th), r * np.cos(self.th)], 1)
        return pos, None, pos                                   # command = smoothed bearing


class AlphaBeta:
    def __init__(self, p0, v0, a=0.5):
        self.a, self.b = a, a ** 2 / (2 - a)
        self.p, self.v = p0.copy(), v0.copy()

    def step(self, fix, *_):
        pp = self.p + T * self.v
        r = fix - pp
        self.p = pp + self.a * r
        self.v = self.v + self.b / T * r
        return self.p, self.v, self.p + T * self.v


# =============================================================================
# Closed-loop simulation
# =============================================================================
def run_scenario(geo, P_true, V_true, B, seed, trackers=('raw', 'T0', 'T1', 'T2', 'T3', 'T4'), keep_trial=True):
    rng = np.random.default_rng(seed)
    n = P_true.shape[0]
    th_true = np.array([mc.bearing(mc.P_RIS, p) for p in P_true])
    cmd = {k: np.full(B, th_true[0]) for k in trackers}        # acquisition: beam on the target
    est = {k: np.zeros((n, B, 2)) for k in trackers}
    vel = {k: np.full((n, B, 2), np.nan) for k in trackers}
    perr = {k: np.zeros((n, B)) for k in trackers}
    gain = {k: np.zeros((n, B)) for k in trackers}
    snr_min = np.inf
    fixes = {}
    flt = {}
    mrng = {k: np.random.default_rng([seed, 0]) for k in trackers}   # common random numbers
    for k in range(n):
        for name in trackers:
            fix, ldh, Rp, rld, g, snr = geo.measure(mrng[name], P_true[k], V_true[k], cmd[name])
            snr_min = min(snr_min, float(np.min(snr)))
            if name == 'raw' and k == 0:
                pass
            if k == 0:
                v0 = V_true[0][None, :] + 0.1 * mrng[name].standard_normal((B, 2))
                x4 = np.concatenate([fix, v0], 1)
                x6 = np.concatenate([fix, v0, np.zeros((B, 2))], 1)
                P4 = np.tile(np.diag([1e-3, 1e-3, 0.01, 0.01]), (B, 1, 1))
                P6 = np.tile(np.diag([1e-3, 1e-3, 0.01, 0.01, 1.0, 1.0]), (B, 1, 1))
                flt[name] = {'raw': None, 'T0': AlphaSmoother(fix), 'T1': AlphaBeta(fix, v0),
                             'T2': KF('cv', 0.5, x4, P4, geo.p_rx), 'T3': KF('ca', 20.0, x6, P6, geo.p_rx),
                             'T4': IMM(0.2, 30.0, x6, P6, geo.p_rx)}[name]
                pe, ve, pp = fix, v0, fix + T * v0
            elif name == 'raw':
                pe, ve, pp = fix, None, fix
            else:
                pe, ve, pp = flt[name].step(fix, ldh, Rp, rld)
            est[name][k] = pe
            if ve is not None:
                vel[name][k] = ve
            perr[name][k] = cmd[name] - th_true[k]
            gain[name][k] = g
            if name == 'T0':
                cmd[name] = flt[name].th.copy() if k > 0 else np.arctan2(fix[:, 0], fix[:, 1])
            else:
                cmd[name] = np.arctan2(pp[:, 0], pp[:, 1])
            if keep_trial and name == 'raw':
                fixes[k] = fix[0].copy()
    return dict(est=est, vel=vel, perr=perr, gain=gain, th_true=th_true, fixes=fixes, snr_min=snr_min)


def metrics(res, P_true, V_true, skip=1):
    hp = np.deg2rad(mc.hpbw_deg())
    out = {}
    for k, E in res['est'].items():
        e = np.linalg.norm(E[skip:] - P_true[skip:, None, :], axis=2)
        pe = np.abs(res['perr'][k][skip:])
        g = res['gain'][k][skip:]
        V = res['vel'][k][skip:]
        ve = np.linalg.norm(V - V_true[skip:, None, :], axis=2)
        out[k] = dict(pos_rmse=float(np.sqrt(np.mean(e ** 2))), pos_p95=float(np.percentile(e, 95)),
                      vel_rmse=float(np.sqrt(np.nanmean(ve ** 2))) if np.isfinite(ve).any() else None,
                      point_rms_deg=float(np.degrees(np.sqrt(np.mean(pe ** 2)))),
                      point_max_deg=float(np.degrees(pe.max())),
                      loss_mean_db=float(-10 * np.log10(np.mean(g))), loss_max_db=float(-10 * np.log10(g.min())),
                      track_loss_frac=float(np.mean(pe > hp / 2)))
    return out


def validate_measurement_model(geo, P_true, B, seed):
    """Signal-level Phase-2 estimator along the straight-line trajectory versus
    the CRB-based Gaussian model (perfect pointing)."""
    nu = mc.tone_offsets()
    keep = mc.P_TARGET.copy()
    rL, rLd = [], []
    try:
        for k in range(P_true.shape[0]):
            snr = float(geo.snr(P_true[k], np.array([1.0]))[0])
            mc.P_TARGET = P_true[k].copy()
            Y, L, v = p2.simulate_physical(np.random.default_rng([seed, k]), B, snr, mc.NSW, nu, geo.p_rx)
            o = p2.est_coherent(Y, nu, want=('ml',))
            sL, _, sLd = geo.sigmas(P_true[k], np.array([snr]))
            rL.append(np.mean(p2.wrap_err(o['L_ml'], L) ** 2) / sL[0] ** 2)
            rLd.append(np.mean((2 * (o['v_ml'] - v)) ** 2) / sLd[0] ** 2)
    finally:
        mc.P_TARGET = keep
    return float(np.sqrt(np.mean(rL))), float(np.sqrt(np.mean(rLd)))


# =============================================================================
# Figures / tables
# =============================================================================
CL = {'raw': '#bbbbbb', 'T0': '#d95f02', 'T1': '#7b3294', 'T2': '#1b9e77', 'T3': '#e7298a', 'T4': '#1f5fa8'}






def write_tables(out, cvm, cvinfo, sweep_rows):
    m0, m4, mr = cvm['T0'], cvm['T4'], cvm['raw']
    rows = [
        ('Initial range / angle', '$R_0$, $\\theta_0$', '10 m, 60$^\\circ$'),
        ('Target speed / heading', '$|\\bm v|$, $\\psi_v$', '2.0 m/s, 45$^\\circ$'),
        ('Final range / angle (frame %d)' % cvinfo['n'], '$R_{%d}$, $\\theta_{%d}$' % (cvinfo['n'], cvinfo['n']),
         '%.2f m, %.2f$^\\circ$' % (cvinfo['R_end'], cvinfo['th_end'])),
        ('CPIs / duration', '$N_{\\rm fr}$, $T_{\\rm tot}$', '%d, %.1f ms' % (cvinfo['n'], cvinfo['n'] * T * 1e3)),
        ('Rx bistatic angle', '$\\beta$', '%.0f$^\\circ$' % cvinfo['beta']),
    ]
    perf = [
        ('Position RMSE', '%.2f cm' % (mr['pos_rmse'] * 100), '%.2f cm' % (m0['pos_rmse'] * 100), '%.2f cm' % (m4['pos_rmse'] * 100)),
        ('95th-pct position error', '%.2f cm' % (mr['pos_p95'] * 100), '%.2f cm' % (m0['pos_p95'] * 100), '%.2f cm' % (m4['pos_p95'] * 100)),
        ('Velocity RMSE', '--', '--', '%.2f cm/s' % (m4['vel_rmse'] * 100)),
        ('Pointing error (rms)', '%.3f$^\\circ$' % mr['point_rms_deg'], '%.3f$^\\circ$' % m0['point_rms_deg'], '%.3f$^\\circ$' % m4['point_rms_deg']),
        ('Mean illumination loss', '%.3f dB' % mr['loss_mean_db'], '%.3f dB' % m0['loss_mean_db'], '%.3f dB' % m4['loss_mean_db']),
    ]
    tex = r"""\begin{table}[!t]
\centering
\caption{Tracking Scenario and Performance (Straight Line)}
\label{tab:tracking}
\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}lcc@{}}
\hline\hline
Parameter & Symbol & Value \\
\hline
""" + "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows) + r"""
\hline
\end{tabular}\\[2pt]
\begin{tabular}{@{}lccc@{}}
Metric & Per-CPI fix & $\alpha$-smoother & IMM \\
\hline
""" + "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in perf) + r"""
\hline\hline
\multicolumn{4}{@{}p{0.97\columnwidth}@{}}{\scriptsize %d Monte Carlo trials; $\mathrm{SNR}_1=OPSNR$\,dB at the start; measurement noise at the corrected bounds of the frame SNR, including the illumination gain of the beam commanded one CPI earlier.}
\end{tabular}
\end{table}
""" % cvinfo['trials']
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'T3rev_tracking.tex').write_text(tex.replace('OPSNR', f'{mc.SNR1_DB:.1f}'), encoding='utf-8')
    body = "\n".join(
        f"{r['label']} & {r['accel']:.2f} & " + " & ".join(f"{r['m'][k]['pos_rmse']*100:.2f}" for k in ('raw', 'T0', 'T1', 'T2', 'T3', 'T4'))
        + f" & {r['m']['T4']['loss_max_db']:.2f} \\\\" for r in sweep_rows)
    tex2 = r"""\begin{table}[!t]
\centering
\caption{Position RMSE (cm) of the Trackers Under Manoeuvre}
\label{tab:manoeuvre}
\footnotesize
\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{@{}lccccccccc@{}}
\hline\hline
Scenario & $|a|$ & Fix & $\alpha$-sm. & $\alpha$--$\beta$ & CV & CA & IMM & IMM loss \\
 & (m/s$^2$) & & & & & & & max (dB) \\
\hline
""" + body + r"""
\hline\hline
\multicolumn{9}{@{}p{0.97\columnwidth}@{}}{\scriptsize 2\,s trajectories starting at 2\,m/s; turns from 0.5\,s, acceleration from 0.5 to 1.25\,s; CV/CA/IMM fuse the Doppler path rate. ``IMM loss'': worst-frame illumination loss from pointing error.}
\end{tabular}
\end{table}
"""
    (Path(out) / 'TF_manoeuvre.tex').write_text(tex2, encoding='utf-8')


# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description='tracking under manoeuvre')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'exp05_tracking'))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--trials', type=int, default=None)
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
    B = args.trials or (50 if args.quick else 200)
    geo = Geometry(args.beta)
    log = []

    def say(s=''):
        print(s, flush=True)
        log.append(s)

    say('=' * 74)
    say(' Tracking under manoeuvre')
    say('=' * 74)
    say(f' SNR1 = {mc.SNR1_DB:.1f} dB   trials = {B}   beta = {args.beta:.0f} deg   T_CPI = {T*1e3:.1f} ms')

    # ---- 1. measurement-model validation --------------------------------------------------
    say('\n[1] Measurement model vs signal-level estimator (straight line, 50 CPIs)')
    Pcv50, Vcv50 = trajectory('cv', 0, 50)
    rL, rLd = validate_measurement_model(geo, Pcv50, 20 if args.quick else 100, 7)
    say(f'    RMS(error)/bound along the trajectory: path {rL:.3f}, path rate {rLd:.3f}')

    # ---- 2. straight line (Table III, Figs. 9-10) ------------------------------------------
    say('\n[2] Straight line, 50 CPIs (regenerates Table III and Figs. 9-10)')
    res_cv = run_scenario(geo, Pcv50, Vcv50, B, 11)
    m_cv = metrics(res_cv, Pcv50, Vcv50)
    for k in ('raw', 'T0', 'T1', 'T2', 'T3', 'T4'):
        m = m_cv[k]
        say(f"    {k:4s} pos {m['pos_rmse']*100:5.2f} cm (p95 {m['pos_p95']*100:5.2f})  vel "
            f"{(m['vel_rmse'] or float('nan'))*100:6.2f} cm/s  point {m['point_rms_deg']:.3f} deg rms  loss {m['loss_mean_db']:.4f} dB")
    cvinfo = dict(n=50, R_end=float(np.linalg.norm(Pcv50[-1])), th_end=float(np.degrees(mc.bearing(mc.P_RIS, Pcv50[-1]))),
                  beta=args.beta, trials=B)

    # ---- 3. manoeuvre sweep -------------------------------------------------------------
    nf = int(round(2.0 / T))
    say(f'\n[3] Manoeuvres ({nf} CPIs = 2 s, onset 0.5 s; acceleration phase 0.75 s)')
    cases = [('Straight line', 'cv', 0.0, 'cv0', 0.0)]
    turns = [15, 30, 60, 90] if not args.quick else [30, 90]
    accs = [0.5, 1.0, 2.0, 4.0] if not args.quick else [1.0, 4.0]
    for w in turns:
        cases.append((f'Turn {w}$^\\circ$/s', 'turn', w, 'turn', mc.V_TARGET * np.deg2rad(w)))
    for a in accs:
        cases.append((f'Accel. {a:g} m/s$^2$', 'acc', a, 'acc', a))
    cases.append(('S-manoeuvre $\\pm$60$^\\circ$/s', 's', 60, 's', mc.V_TARGET * np.deg2rad(60)))
    sweep = []
    ex = None
    for i, (lab, kind, par, fam, acc) in enumerate(cases):
        P, V = trajectory(kind, par, nf)
        res = run_scenario(geo, P, V, B, 100 + i)
        m = metrics(res, P, V, skip=1)
        sweep.append(dict(label=lab, kind=kind, param=par, family=fam, accel=float(acc), m=m, snr_min=res['snr_min']))
        plain = lab.replace('$^\\circ$', 'deg').replace('$^2$', '^2').replace('$\\pm$', '+/-')
        say(f"    {plain:26s} |a| {acc:4.2f}  pos RMSE: "
            + '  '.join(f"{k} {m[k]['pos_rmse']*100:6.2f}" for k in ('raw', 'T0', 'T1', 'T2', 'T3', 'T4'))
            + f"  | IMM loss max {m['T4']['loss_max_db']:.3f} dB, T0 {m['T0']['loss_max_db']:.3f} dB | min SNR {res['snr_min']:.1f} dB")
        if kind == 's':
            ex = dict(P=P, res=res, label='S-manoeuvre',
                      reversals=[0.5 + 0.75 * j for j in range(int((nf * T - 0.5) / 0.75) + 1)])
    for s in sweep:
        if s['family'] == 'cv0':
            s['family'] = 'turn'
    acc0 = dict(sweep[0])
    acc0['family'] = 'acc'
    sweep_plot = sweep + [acc0]

    # ---- 4. outputs -------------------------------------------------------------------
    say('\n[4] Writing figures and tables')
    paths = [fig_cv(fdir, Pcv50, res_cv, geo), fig_manoeuvre(fdir, ex, sweep_plot)]
    table_rows = [s for s in sweep if s['kind'] in ('cv', 's') or (s['kind'] == 'turn' and s['param'] in (30, 90))
                  or (s['kind'] == 'acc' and s['param'] in (1.0, 4.0))]
    write_tables(tdir, m_cv, cvinfo, table_rows)
    for p in paths:
        say(f'    {p}')
    say(f"    {tdir / 'T3rev_tracking.tex'}\n    {tdir / 'TF_manoeuvre.tex'}")

    # ---- 5. gate -------------------------------------------------------------------------
    say('\n[checks] Validation')
    best_single = [min(s['m'][k]['pos_rmse'] for k in ('T1', 'T2', 'T3')) for s in sweep]
    imm = [s['m']['T4']['pos_rmse'] for s in sweep]
    checks = [
        (f'every frame above the estimator threshold ({SNR_THRESHOLD_DB:.0f} dB) -> Gaussian model valid',
         min(s['snr_min'] for s in sweep) >= SNR_THRESHOLD_DB and res_cv['snr_min'] >= SNR_THRESHOLD_DB),
        ('measurement model matches signal-level estimator (path, rate within 15%)',
         abs(rL - 1) < 0.15 and abs(rLd - 1) < 0.15),
        ('straight line: all trackers keep illumination loss < 0.1 dB', all(m_cv[k]['loss_max_db'] < 0.1 for k in m_cv)),
        ('IMM within 15% of the best single-model filter in every scenario',
         all(i <= 1.15 * b for i, b in zip(imm, best_single))),
        ('IMM improves on the raw per-CPI fix on the straight line (> 1.5x)', m_cv['raw']['pos_rmse'] / m_cv['T4']['pos_rmse'] > 1.5),
        ('no track loss (pointing > HPBW/2) for the IMM in any scenario', all(s['m']['T4']['track_loss_frac'] == 0 for s in sweep)),
        ('IMM worst-frame illumination loss < 0.5 dB in every scenario', all(s['m']['T4']['loss_max_db'] < 0.5 for s in sweep)),
    ]
    for n_, ok in checks:
        say(f"    [{'PASS' if ok else 'FAIL'}] {n_}")
    say(f'    time {time.time()-t0:.0f} s')

    res_json = dict(snr1_db=mc.SNR1_DB, beta_deg=args.beta, trials=B, validation=dict(path=rL, rate=rLd),
                    straight=dict(info=cvinfo, metrics=m_cv), sweep=[{k: v for k, v in s.items()} for s in sweep],
                    gate5={n_: bool(o) for n_, o in checks})
    (ddir / 'exp05_results.json').write_text(json.dumps(res_json, indent=2, default=float), encoding='utf-8')
    (ddir / 'exp05_summary.txt').write_text('\n'.join(log), encoding='utf-8')
    return 0 if all(o for _, o in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
