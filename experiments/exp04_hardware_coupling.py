"""
exp04_hardware_coupling.py -- Element phase noise and inter-stage coupling
==========================================================================
Static element phase noise, PIN switching jitter and common-mode
oscillator phase noise (Monte Carlo with the Phase-2 estimator);
Rayleigh-Sommerfeld (angular-spectrum) inter-layer coupling versus gap,
lateral offset, rotation and tilt; tolerance budget.

Paper figures produced: Fig. 14 (FH_interlayer_coupling), Fig. 15 (FF_phase_noise)
Outputs: results/exp04_hardware_coupling/ (data, tables, extra figures), results/figdata/ (figure data),
         paper/figures/ (paper figures)
Run    : python experiments/exp04_hardware_coupling.py [--quick]
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
from figures.fig15_phase_noise import fig_phase_noise
from figures.fig14_interlayer_coupling import fig_coupling
from experiments import exp02_estimation_crb as p2

logging.getLogger('fontTools').setLevel(logging.ERROR)
TH1 = np.deg2rad(mc.THETA1_DEG)


# =============================================================================
# Part A -- phase noise
# =============================================================================
def af_cut(theta, eps_amp=None, eps_ph=None, bits=mc.PHASE_BITS):
    """x-cut array factor of all 256 R-RIS elements with per-element errors."""
    x = mc.rris_element_x()
    psi = mc.quantise_phase(mc.K_C * x * np.sin(TH1), bits)
    w = np.ones_like(x, dtype=complex)
    if eps_amp is not None:
        w = w * (1 + eps_amp)
    if eps_ph is not None:
        w = w * np.exp(1j * eps_ph)
    w = w * np.exp(1j * psi)
    return (w[None, :] * np.exp(-1j * mc.K_C * x[None, :] * np.sin(np.atleast_1d(theta))[:, None])).sum(1) / x.size


def static_pattern_stats(sig_deg, n_real, rng):
    """Gain toward target, sidelobe level and pointing bias over device draws."""
    th_fine = np.deg2rad(np.linspace(mc.THETA1_DEG - 4, mc.THETA1_DEG + 4, 1601))
    th_all = np.deg2rad(np.linspace(-89, 89, 3561))
    u0 = np.sin(TH1)
    main = np.abs(np.sin(th_all) - u0) < 2.0 / mc.N_X
    ref_peak = np.abs(af_cut(np.array([TH1]), bits=None))[0]
    g, sll, pb = [], [], []
    for _ in range(n_real):
        e = np.deg2rad(sig_deg) * rng.standard_normal(mc.N_ELEM) if sig_deg > 0 else None
        a_t = np.abs(af_cut(np.array([TH1]), eps_ph=e))[0]
        a_f = np.abs(af_cut(th_fine, eps_ph=e))
        a_a = np.abs(af_cut(th_all, eps_ph=e))
        g.append((a_t / ref_peak) ** 2)
        sll.append(20 * np.log10(a_a[~main].max() / a_a.max()))
        pb.append(np.degrees(th_fine[np.argmax(a_f)]) - mc.THETA1_DEG)
    g = np.array(g)
    return dict(gain_loss_db=float(-10 * np.log10(g.mean())), gain_loss_p95_db=float(-10 * np.log10(np.percentile(g, 5))),
                sll_db=float(np.mean(sll)), pointing_rms_deg=float(np.sqrt(np.mean(np.square(pb)))))


def wiener_loss(dnu, T=mc.TCPI):
    """Coherent-integration loss of a Wiener phase process of linewidth dnu
    over T:  eta = 2(x - 1 + e^-x)/x^2,  x = pi dnu T."""
    x = np.pi * np.asarray(dnu, dtype=float) * T
    with np.errstate(invalid='ignore', divide='ignore'):
        eta = np.where(x > 1e-9, 2 * (x - 1 + np.exp(-x)) / np.maximum(x, 1e-300) ** 2, 1.0)
    return eta


def wiener_phase(rng, B, dnu, m=mc.M_TONES, n=mc.NSW):
    """Common-mode phase over the CPI in transmission order (sweep-major)."""
    if dnu <= 0:
        return np.zeros((B, m, n))
    steps = np.sqrt(2 * np.pi * dnu * mc.TSYM) * rng.standard_normal((B, n * m))
    return np.cumsum(steps, axis=1).reshape(B, n, m).transpose(0, 2, 1)


def pos_rmse(eL, eth, p_rx):
    L0 = mc.bistatic_path(mc.P_TARGET, p_rx)
    t0 = mc.bearing(p_rx, mc.P_TARGET)
    p = mc.invert_position(L0 + eL, t0 + eth, p_rx)
    return float(np.sqrt(np.mean(np.sum((p - mc.P_TARGET) ** 2, axis=1))))


def sensing_mc(ntr, batch, p_rx, gains_fn=None, dnu=0.0, seed=(40,)):
    """Path/velocity errors with the Phase-2 estimator; common random numbers
    (the noise stream depends only on the batch index, not on the impairment)."""
    nu = mc.tone_offsets()
    eL, ev, gm = [], [], []
    done, b = 0, 0
    while done < ntr:
        Bn = min(batch, ntr - done)
        rng = p2.child_rng(*seed, b)
        hw = p2.child_rng(*seed, 1000 + b)
        g = gains_fn(hw, Bn) if gains_fn else None
        Y, L, v = p2.simulate_physical(rng, Bn, mc.SNR1_DB, mc.NSW, nu, p_rx, gains=g)
        if dnu > 0:
            Y = Y * np.exp(1j * wiener_phase(p2.child_rng(*seed, 2000 + b), Bn, dnu))
        o = p2.est_coherent(Y, nu, want=('ml',))
        eL.append(p2.wrap_err(o['L_ml'], L))
        ev.append(o['v_ml'] - v)
        gm.append(np.mean(np.abs(g) ** 2, axis=1) if g is not None else np.ones(Bn))
        done += Bn
        b += 1
    return np.concatenate(eL), np.concatenate(ev), np.concatenate(gm)


# =============================================================================
# Part B -- angular-spectrum (Rayleigh-Sommerfeld) inter-layer model
# =============================================================================
class ASModel:
    """Scalar angular-spectrum propagation between the T-RIS and R-RIS planes.
    Grid pitch d/16 keeps the element period an integer number of pixels."""

    def __init__(self, n=512, sub=16):
        self.d = mc.D_ELEM
        self.dx = self.d / sub
        self.n = n
        self.x = (np.arange(n) - n / 2 + 0.5) * self.dx     # element centres on pixel boundaries
        self._rx_idx = {}
        self.X, self.Y = np.meshgrid(self.x, self.x, indexing='xy')
        kx = 2 * np.pi * np.fft.fftfreq(n, self.dx)
        KX, KY = np.meshgrid(kx, kx, indexing='xy')
        self.kz = np.sqrt((mc.K_C ** 2 - KX ** 2 - KY ** 2).astype(complex))
        self.cent = (np.arange(mc.N_X) - (mc.N_X - 1) / 2) * self.d
        cx, cy = np.meshgrid(self.cent, self.cent, indexing='xy')
        self.ex, self.ey = cx.ravel(), cy.ravel()             # R-RIS element centres

    def _cov1d(self, u, a):
        h = self.dx / 2
        return np.clip(np.minimum(u + h, a / 2) - np.maximum(u - h, -a / 2), 0, self.dx) / self.dx

    def source(self, f, dxo=0.0, dyo=0.0, gamma_deg=0.0):
        """T-RIS aperture field: patches of side f*d, area-weighted rasterisation,
        after lateral offset (dxo, dyo) and in-plane rotation gamma of the layer."""
        a = f * self.d
        g = np.deg2rad(gamma_deg)
        Xs, Ys = self.X - dxo, self.Y - dyo
        Xr = np.cos(g) * Xs + np.sin(g) * Ys
        Yr = -np.sin(g) * Xs + np.cos(g) * Ys
        ix = np.clip(np.round(Xr / self.d + (mc.N_X - 1) / 2), 0, mc.N_X - 1)
        iy = np.clip(np.round(Yr / self.d + (mc.N_X - 1) / 2), 0, mc.N_X - 1)
        ux = Xr - (ix - (mc.N_X - 1) / 2) * self.d
        uy = Yr - (iy - (mc.N_X - 1) / 2) * self.d
        return (self._cov1d(ux, a) * self._cov1d(uy, a)).astype(complex)

    def propagate(self, U0, z):
        return np.fft.ifft2(np.fft.fft2(U0) * np.exp(1j * self.kz * z))

    def receive(self, U, f):
        """Exact pixel-overlap average of the field over each R-RIS patch
        (receive aperture = same patch; patch edges lie on pixel boundaries)."""
        key = round(f, 6)
        if key not in self._rx_idx:
            a = f * self.d
            idx = []
            for cx, cy in zip(self.ex, self.ey):
                cols = np.where(np.abs(self.x - cx) < a / 2)[0]
                rows = np.where(np.abs(self.x - cy) < a / 2)[0]
                idx.append((rows[:, None] * self.n + cols[None, :]).ravel())
            self._rx_idx[key] = np.array(idx)
        return U.ravel()[self._rx_idx[key]].mean(axis=1)

    def central(self, k=4):
        """Indices of the central k x k elements (free of edge diffraction)."""
        lo, hi = (mc.N_X - k) // 2, (mc.N_X + k) // 2
        ii = np.arange(mc.N_ELEM).reshape(mc.N_X, mc.N_X)[lo:hi, lo:hi].ravel()
        return ii

    def beam_weights(self, theta=TH1):
        x = self.ex
        return np.exp(1j * mc.K_C * x * np.sin(TH1)) * np.exp(-1j * mc.K_C * x * np.sin(theta))

    def gain(self, c, theta=TH1):
        return np.abs(np.sum(c * self.beam_weights(theta))) ** 2 / self.ex.size ** 2

    def peak_deg(self, c, span=3.0):
        th = np.deg2rad(np.linspace(mc.THETA1_DEG - span, mc.THETA1_DEG + span, 1201))
        x = self.ex
        af = np.abs((c[None, :] * np.exp(1j * mc.K_C * x * np.sin(TH1))[None, :]
                     * np.exp(-1j * mc.K_C * x[None, :] * np.sin(th)[:, None])).sum(1))
        return float(np.degrees(th[np.argmax(af)]))


def interp_x(xq, xp, yp):
    xq = np.asarray(xq, dtype=float)
    return np.interp(xq, xp, yp, left=np.nan, right=np.nan)


# =============================================================================
# Figures and table
# =============================================================================




def table_tolerance(out, T):
    rows = "\n".join(f"{r['what']} & {r['t05']} & {r['t10']} & {r['paper']} & {r['note']} \\\\" for r in T)
    tex = r"""\begin{table}[!t]
\centering
\caption{Hardware and Assembly Tolerance Budget ($d_g=2$\,mm, patch fill $f=FPRIM$)}
\label{tab:tolerance}
\footnotesize
\setlength{\tabcolsep}{1.3pt}
\begin{tabular}{@{}lcccl@{}}
\hline\hline
Impairment & 0.5\,dB & 1\,dB & Submitted & Note \\
\hline
""" + rows + r"""
\hline\hline
\multicolumn{5}{@{}p{0.97\columnwidth}@{}}{\scriptsize Tolerances give the impairment that costs 0.5 or 1\,dB of array gain (phase noise, assembly) or of coherent-integration gain (linewidth). Assembly values are maximum element displacements from the angular-spectrum model; ``Submitted'' is the on-axis exponential model \eqref{eq:C0}. A 1\,dB loss raises the position RMSE by 12\,\%, the estimator remaining efficient.}
\end{tabular}
\end{table}
"""
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'TH_tolerance_budget.tex').write_text(tex, encoding='utf-8')


def tol_at(disp, loss, level):
    disp, loss = np.asarray(disp), np.asarray(loss)
    idx = np.where(loss >= level)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(disp[0])
    return float(np.interp(level, [loss[i - 1], loss[i]], [disp[i - 1], disp[i]]))


# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description='phase noise and inter-layer misalignment')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'exp04_hardware_coupling'))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--trials', type=int, default=None)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--beta', type=float, default=mc.BETA_DEFAULT_DEG)
    ap.add_argument('--snr1', type=float, default=None)
    ap.add_argument('--fill', type=float, default=0.5, help='primary patch fill factor (8/16 = 0.5)')
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
    nu = mc.tone_offsets()
    log = []

    def say(s=''):
        print(s, flush=True)
        log.append(s)

    say('=' * 74)
    say(' PHASE 4 -- Phase noise (R3.1) and inter-layer misalignment (R3.3)')
    say('=' * 74)
    say(f' SNR1 = {mc.SNR1_DB:.1f} dB   trials = {ntr}   beta = {args.beta:.0f} deg')

    # ---------------- A1 static phase noise -----------------------------------------------
    say('\n[A1] Static element phase noise on top of 3-bit quantisation')
    sigmas = [0, 5, 10, 15, 20, 30, 40] if args.quick else [0, 5, 10, 13, 15, 20, 25, 30, 35, 40]
    q3_equiv = np.degrees(2 * np.pi / 2 ** mc.PHASE_BITS / np.sqrt(12))
    q3_loss_theory = -20 * np.log10(np.sinc(1 / 2 ** mc.PHASE_BITS))
    q3_loss_af = -20 * np.log10(np.abs(af_cut(np.array([TH1])))[0] / np.abs(af_cut(np.array([TH1]), bits=None))[0])
    say(f'    3-bit quantisation: loss {q3_loss_af:.3f} dB (theory sinc^2: {q3_loss_theory:.3f} dB), '
        f'uniform-error equivalent sigma = {q3_equiv:.1f} deg')
    nreal = 100 if args.quick else 300
    A1 = []
    ratio0 = None
    for s_ in sigmas:
        st = static_pattern_stats(s_, nreal, p2.child_rng(50, s_))

        def gains(rng, Bn, s_=s_):
            out_ = np.empty((Bn, nu.size), complex)
            for b in range(Bn):
                out_[b] = mc.rris_tone_gains(mc.FC + nu, TH1, TH1, bits=mc.PHASE_BITS,
                                             sigma_phase_rad=np.deg2rad(s_), rng=rng)
            return out_ / np.abs(mc.rris_tone_gains(np.array([mc.FC]), TH1, TH1, bits=None))[0]
        eL, ev, gm = sensing_mc(ntr, args.batch, p_rx, gains_fn=gains, seed=(41,))
        gdb = 10 * np.log10(np.mean(gm))
        snr_eff = mc.SNR1_DB + gdb
        cL = float(mc.crb_path(snr_eff, nu=nu))
        cth = float(mc.crb_angle_ula(mc.snr_snapshot_lin(snr_eff)))
        ea = p2.angle_mc(p2.child_rng(51, 0), ntr, snr_eff)
        r = dict(sigma_deg=s_, gain_loss_db=st['gain_loss_db'], gain_loss_p95_db=st['gain_loss_p95_db'],
                 sll_db=st['sll_db'], pointing_rms_deg=st['pointing_rms_deg'], mc_gain_db=float(gdb),
                 theory_loss_db=float(q3_loss_theory - 10 * np.log10(np.exp(-np.deg2rad(s_) ** 2)
                                                                     + (1 - np.exp(-np.deg2rad(s_) ** 2)) / mc.N_ELEM)),
                 path_rmse=float(np.sqrt(np.mean(eL ** 2))), path_crb=cL,
                 pos_rmse=pos_rmse(eL, ea['ml'], p_rx), pos_crb=float(mc.crb_position(cL, cth, mc.P_TARGET, p_rx)),
                 vel_rmse=float(np.sqrt(np.mean(ev ** 2))), q3_equiv_deg=float(q3_equiv))
        r['ratio'] = r['path_rmse'] / r['path_crb']
        ratio0 = ratio0 or r['ratio']
        A1.append(r)
        say(f"    sigma {s_:4.0f} deg  gain loss {r['gain_loss_db']:5.2f} dB (theory {r['theory_loss_db']:5.2f}, p95 {r['gain_loss_p95_db']:5.2f})"
            f"  SLL {r['sll_db']:6.1f} dB  point rms {r['pointing_rms_deg']:.3f} deg | path {r['path_rmse']*100:4.2f}/{cL*100:4.2f} cm"
            f"  pos {r['pos_rmse']*100:4.2f}/{r['pos_crb']*100:4.2f} cm")

    # ---------------- A2 switching jitter ---------------------------------------------------
    say('\n[A2] PIN switching jitter (independent per element and symbol)')
    fmax = nu.max()
    A2 = []
    for sj in (1e-12, 10e-12, 30e-12, 100e-12, 300e-12):
        loss = -10 * np.log10(np.exp(-(2 * np.pi * fmax * sj) ** 2))
        A2.append(dict(sigma_t_ps=sj * 1e12, loss_db_fmax=float(loss)))
        say(f'    sigma_t = {sj*1e12:5.0f} ps -> loss at {fmax/1e6:.0f} MHz: {loss:.4f} dB')
    sj05 = np.sqrt(np.log(10 ** 0.05)) / (2 * np.pi * fmax)
    sj10 = np.sqrt(np.log(10 ** 0.1)) / (2 * np.pi * fmax)
    say(f'    tolerance: {sj05*1e12:.0f} ps (0.5 dB), {sj10*1e12:.0f} ps (1 dB)')

    # ---------------- A3 common-mode oscillator phase noise --------------------------------
    say('\n[A3] Common-mode oscillator phase noise (Wiener, linewidth dnu) over the CPI')
    dnus = [0, 1, 10, 100] if args.quick else [0, 0.3, 1, 3, 10, 30, 100]
    A3 = []
    for dn in dnus:
        eL, ev, _ = sensing_mc(ntr, args.batch, p_rx, dnu=dn, seed=(43,))
        eta = float(wiener_loss(dn))
        phs = wiener_phase(p2.child_rng(44, 0), 200 if args.quick else 1000, dn)
        eta_mc = float(np.mean(np.abs(np.mean(np.exp(1j * phs.reshape(phs.shape[0], -1)), axis=1)) ** 2))
        snr_eff = mc.SNR1_DB + 10 * np.log10(eta)
        ea = p2.angle_mc(p2.child_rng(45, 0), ntr, snr_eff)
        r = dict(dnu=dn, eta=eta, eta_db=float(-10 * np.log10(eta)), eta_mc_db=float(-10 * np.log10(eta_mc)),
                 path_rmse=float(np.sqrt(np.mean(eL ** 2))), vel_rmse=float(np.sqrt(np.mean(ev ** 2))),
                 pos_rmse=pos_rmse(eL, ea['ml'], p_rx))
        A3.append(r)
        say(f"    dnu {dn:6.1f} Hz  loss {r['eta_db']:5.2f} dB (MC {r['eta_mc_db']:5.2f})  path {r['path_rmse']*100:6.2f} cm"
            f"  vel {r['vel_rmse']*100:6.2f} cm/s  pos {r['pos_rmse']*100:6.2f} cm")
    xs = np.logspace(-3, 3, 4000)
    lw05 = float(xs[np.argmax(-10 * np.log10(wiener_loss(xs)) >= 0.5)])
    lw10 = float(xs[np.argmax(-10 * np.log10(wiener_loss(xs)) >= 1.0)])
    say(f'    tolerance: dnu = {lw05:.1f} Hz (0.5 dB), {lw10:.1f} Hz (1 dB) -> Tx and Rx must share a frequency reference')

    # ---------------- B inter-layer coupling -----------------------------------------------
    say('\n[B] Inter-layer coupling: angular-spectrum (Rayleigh-Sommerfeld) model')
    M_ = ASModel(n=384 if args.quick else 512)
    fills = [8 / 16, 10 / 16, 14 / 16]           # even pixel widths -> exact rasterisation
    dg = 2e-3
    zg = np.concatenate([[0.01e-3], np.linspace(0.25e-3, 6e-3, 24 if args.quick else 47)])
    gap_curves, ref_c, nom = {}, {}, {}
    for f in fills:
        U0 = M_.source(f)
        c0 = M_.receive(U0, f)
        ref_c[f] = c0
        g0 = M_.gain(c0)
        e = []
        for z in zg:
            c = M_.receive(M_.propagate(U0, z), f) * np.exp(-1j * mc.K_C * z)
            e.append(10 * np.log10(M_.gain(c) / g0))
        gap_curves[f'{f:.4f}'] = e
        cn = M_.receive(M_.propagate(U0, dg), f) * np.exp(-1j * mc.K_C * dg)
        nom[f] = (cn, M_.gain(cn))
        say(f'    f = {f:.3f}: coupling at 0.01 mm {e[0]:+.2f} dB, at 2 mm {10*np.log10(nom[f][1]/g0):+.2f} dB, '
            f'at 6 mm {e[-1]:+.2f} dB (f^4 limit {40*np.log10(f):+.2f} dB)')
    paper = list(20 * np.log10(np.exp(-mc.K_C * zg)))
    say(f'    submitted Eq.(11) at 2 mm: {20*np.log10(np.exp(-mc.K_C*dg)):+.2f} dB; slope 5.10 dB/mm without saturation')
    fp = args.fill
    e_p = np.array(gap_curves[f'{fp:.4f}'])
    slope2 = float(-np.gradient(e_p, zg * 1e3)[np.argmin(np.abs(zg - dg))])
    say(f'    local gap slope at 2 mm (f = {fp}): {slope2:.2f} dB/mm')

    def mis_sweeps(f):
        cn0, g_nom = nom[f]
        res = {}
        # axial gap deviation (two-sided, worst side)
        dd = np.linspace(0, 1.5e-3, 13)
        lg = []
        for de in dd:
            l1 = []
            for sgn in (-1, 1):
                z = dg + sgn * de
                c = M_.receive(M_.propagate(M_.source(f), z), f) * np.exp(-1j * mc.K_C * z)
                l1.append(10 * np.log10(g_nom / M_.gain(c)))
            lg.append(max(l1))
        res['gap'] = dict(disp_mm=list(dd * 1e3), loss_db=lg, point_deg=[0.0] * dd.size)
        # lateral x and diagonal
        for key, dirv in (('lat_x', (1, 0)), ('lat_diag', (1 / np.sqrt(2), 1 / np.sqrt(2)))):
            ds = np.linspace(0, M_.d / 2 * np.sqrt(2) if key == 'lat_diag' else M_.d / 2, 9)
            ls, pts = [], []
            for s_ in ds:
                c = M_.receive(M_.propagate(M_.source(f, s_ * dirv[0], s_ * dirv[1]), dg), f) * np.exp(-1j * mc.K_C * dg)
                ls.append(10 * np.log10(g_nom / M_.gain(c)))
                pts.append(M_.peak_deg(c) - M_.peak_deg(cn0))
            res[key] = dict(disp_mm=list(ds * 1e3), loss_db=ls, point_deg=pts)
        # rotation
        rho = np.hypot(M_.cent.max(), M_.cent.max())
        gs = np.linspace(0, 3.0, 13)
        ls, pts = [], []
        for g_ in gs:
            c = M_.receive(M_.propagate(M_.source(f, gamma_deg=g_), dg), f) * np.exp(-1j * mc.K_C * dg)
            ls.append(10 * np.log10(g_nom / M_.gain(c)))
            pts.append(M_.peak_deg(c) - M_.peak_deg(cn0))
        res['rot'] = dict(disp_mm=list(rho * np.deg2rad(gs) * 1e3), loss_db=ls, point_deg=pts, angle_deg=list(gs))
        # tilt about y: local gap g_n = dg + x_n tan(alpha)
        half = M_.cent.max()
        amax = np.degrees(np.arctan((dg - 0.1e-3) / (half + f * M_.d / 2)))
        al = np.linspace(0, min(2.5, amax), 11)
        zs = np.linspace(max(0.05e-3, dg - half * np.tan(np.deg2rad(al.max())) - 0.05e-3),
                         dg + half * np.tan(np.deg2rad(al.max())) + 0.05e-3, 31)
        U0 = M_.source(f)
        Cz = np.array([M_.receive(M_.propagate(U0, z), f) * np.exp(-1j * mc.K_C * z) for z in zs])
        ls, pts = [], []
        for a_ in al:
            gn = dg + M_.ex * np.tan(np.deg2rad(a_))
            c = np.array([np.interp(gn[i], zs, Cz[:, i].real) + 1j * np.interp(gn[i], zs, Cz[:, i].imag)
                          for i in range(M_.ex.size)])
            ls.append(10 * np.log10(g_nom / M_.gain(c)))
            pts.append(M_.peak_deg(c) - M_.peak_deg(cn0))
        res['tilt'] = dict(disp_mm=list(half * np.tan(np.deg2rad(al)) * 1e3), loss_db=ls, point_deg=pts,
                           angle_deg=list(al), contact_deg=float(amax))
        # submitted model, lateral (co-located pair dominates)
        ds = np.linspace(0, M_.d / 2, 9)
        res['lat_paper'] = dict(disp_mm=list(ds * 1e3),
                                loss_db=list(20 * np.log10(np.exp(-mc.K_C * dg) / np.exp(-mc.K_C * np.hypot(dg, ds)))))
        return res

    mis = {}
    for f in (fp, fills[-1]):
        mis[f'{f:.4f}'] = mis_sweeps(f)
    MP, MF = mis[f'{fp:.4f}'], mis[f'{fills[-1]:.4f}']
    for key in ('gap', 'lat_x', 'lat_diag', 'rot', 'tilt'):
        r = MP[key]
        say(f"    f={fp}: {key:8s} max disp {r['disp_mm'][-1]:.2f} mm -> loss {r['loss_db'][-1]:+.2f} dB,"
            f" max pointing {np.max(np.abs(r['point_deg'])):.3f} deg")
    # physics checks on the central 4x4 block (free of edge diffraction)
    ci = M_.central(4)
    sat_err = []
    for f in fills:
        c6 = M_.receive(M_.propagate(M_.source(f), 6e-3), f) * np.exp(-1j * mc.K_C * 6e-3)
        c00 = M_.receive(M_.source(f), f)
        sat_err.append(abs(20 * np.log10(abs(c6[ci].mean()) / abs(c00[ci].mean())) - 40 * np.log10(f)))
        say(f'    central block, f = {f:.3f}: coupling at 6 mm {20*np.log10(abs(c6[ci].mean())/abs(c00[ci].mean())):+.2f} dB'
            f' (f^4 limit {40*np.log10(f):+.2f} dB)')
    cP = M_.receive(M_.propagate(M_.source(fp, M_.d, 0), dg), fp) * np.exp(-1j * mc.K_C * dg)
    per_err = abs(20 * np.log10(abs(cP[ci].mean()) / abs(nom[fp][0][ci].mean())))
    edge_dB = 10 * np.log10(nom[fp][1] / M_.gain(cP))
    say(f'    full-period shift: central block {per_err:.3f} dB; whole array {edge_dB:+.2f} dB (one column uncovered)')

    # ---------------- tolerance table -----------------------------------------------------
    a03 = [r for r in A3 if abs(r['dnu'] - 0.3) < 1e-9] or [A3[1]]
    vel03 = a03[0]['vel_rmse'] / A3[0]['vel_rmse']

    def fmt_t(v, unit, dec=2, rng=None):
        if v is not None:
            return f'{v:.{dec}f}\\,{unit}'
        return f'$>${rng:.1f}\\,{unit}' if rng is not None else '$>$range'
    sig = [r['sigma_deg'] for r in A1]
    los = [r['gain_loss_db'] - A1[0]['gain_loss_db'] for r in A1]
    T = [
        dict(what='Phase noise $\\sigma_\\phi$', t05=fmt_t(tol_at(sig, los, 0.5), '$^\\circ$', 0),
             t10=fmt_t(tol_at(sig, los, 1.0), '$^\\circ$', 0), paper='--', note='beyond 3-bit'),
        dict(what='Switching jitter $\\sigma_t$', t05=f'{sj05*1e12:.0f}\\,ps', t10=f'{sj10*1e12:.0f}\\,ps',
             paper='--', note='at 300\\,MHz'),
        dict(what='Osc. linewidth $\\Delta\\nu$', t05=f'{lw05:.1f}\\,Hz', t10=f'{lw10:.1f}\\,Hz',
             paper='--', note=f"$v$: $\\times${vel03:.1f} at 0.3\\,Hz"),
    ]
    for key, lab in (('gap', 'Axial gap $|\\Delta d_g|$'), ('lat_x', 'Lateral, $x$'),
                     ('lat_diag', 'Lateral, diagonal'), ('rot', 'Rotation'), ('tilt', 'Tilt')):
        r = MP[key]
        t05, t10 = tol_at(r['disp_mm'], r['loss_db'], 0.5), tol_at(r['disp_mm'], r['loss_db'], 1.0)
        if key == 'gap':
            pap = '0.10\\,/\\,0.20\\,mm'
        elif key == 'lat_x':
            rp = MP['lat_paper']
            pap = f"{tol_at(rp['disp_mm'], rp['loss_db'], 0.5):.2f}\\,/\\,{tol_at(rp['disp_mm'], rp['loss_db'], 1.0):.2f}\\,mm"
        else:
            pap = '--'
        note = ''
        if key == 'rot':
            note = f"{r['angle_deg'][-1]:.0f}$^\\circ$ = {r['disp_mm'][-1]:.1f}\\,mm"
            if t10 is None:
                t10 = None
                T_rot_max = r['disp_mm'][-1]
        if key == 'tilt':
            note = f"gain; $\\le${np.max(np.abs(r['point_deg'])):.2f}$^\\circ$ pointing"
        if key == 'tilt':
            T.append(dict(what=lab, t05='none', t10='none', paper=pap, note=note))
        else:
            T.append(dict(what=lab, t05=fmt_t(t05, 'mm'), t10=fmt_t(t10, 'mm', rng=r['disp_mm'][-1]), paper=pap, note=note))
    table_tolerance(tdir, T)
    tex = (tdir / 'TH_tolerance_budget.tex').read_text(encoding='utf-8').replace('FPRIM', f'{fp:.2f}')
    (tdir / 'TH_tolerance_budget.tex').write_text(tex, encoding='utf-8')

    B = dict(gap_mm=list(zg * 1e3), gap_curves=gap_curves, gap_paper_db=paper, f_primary=fp, sat_err_db=sat_err,
             mis=MP, mis_f875=MF, slope_2mm_db_per_mm=slope2,
             nominal_coupling_db={f'{f:.4f}': float(10 * np.log10(nom[f][1] / M_.gain(ref_c[f]))) for f in fills})
    p1 = fig_phase_noise(fdir, A1, A3)
    p2_ = fig_coupling(fdir, B)
    say(f'\n[C] Outputs\n    {p1}\n    {p2_}\n    {tdir / "TH_tolerance_budget.tex"}')

    # ---------------- gate ------------------------------------------------------------------
    say('\n[checks] Validation')
    within30 = [r for r in A1 if r['sigma_deg'] <= 30]
    eff = [r['ratio'] / ratio0 for r in within30]
    pos20 = [r['pos_rmse'] for r in A1 if r['sigma_deg'] <= 20]
    eta_err = max(abs(r['eta_db'] - r['eta_mc_db']) for r in A3)
    fl = {f: gap_curves[f'{f:.4f}'] for f in fills}
    checks = [
        ('3-bit loss matches sinc^2 theory (< 0.02 dB)', abs(q3_loss_af - q3_loss_theory) < 0.02),
        ('static phase-noise gain loss matches theory (< 0.1 dB, sigma <= 30 deg)',
         max(abs(r['gain_loss_db'] - r['theory_loss_db']) for r in within30) < 0.1),
        ('estimator stays efficient under phase noise (RMSE/CRB within 10% of sigma=0)',
         max(abs(e - 1) for e in eff) < 0.10),
        ('position RMSE rises < 10% for sigma_phi <= 20 deg', max(pos20) / pos20[0] < 1.10),
        ('Wiener integration loss: closed form = Monte Carlo (< 0.2 dB)', eta_err < 0.2),
        ('AS model: continuous to unit coupling as gap -> 0 (< 0.3 dB at 0.01 mm)', all(abs(fl[f][0]) < 0.3 for f in fills)),
        ('AS model: central block saturates at fundamental-mode limit f^4 (< 0.15 dB at 6 mm)',
         max(sat_err) < 0.15),
        ('AS model (f=0.5) within 1 dB of submitted C0 at 2 mm',
         abs(B['nominal_coupling_db'][f'{0.5:.4f}'] - 20 * np.log10(np.exp(-mc.K_C * dg))) < 1.0),
        ('lateral response periodic in the cell (central block, full-period shift < 0.1 dB)', per_err < 0.1),
    ]
    for n_, ok in checks:
        say(f"    [{'PASS' if ok else 'FAIL'}] {n_}")
    say(f'    time {time.time()-t0:.0f} s')

    res = dict(snr1_db=mc.SNR1_DB, beta_deg=args.beta, n_trials=ntr, q3_loss_db=float(q3_loss_af),
               q3_equiv_deg=float(q3_equiv), A1=A1, A2=A2, A3=A3,
               jitter_tol_ps=[float(sj05 * 1e12), float(sj10 * 1e12)], linewidth_tol_hz=[lw05, lw10],
               coupling=B, tolerance_rows=T, gate4={n_: bool(o) for n_, o in checks})
    (ddir / 'exp04_results.json').write_text(json.dumps(res, indent=2, default=float), encoding='utf-8')
    (ddir / 'exp04_summary.txt').write_text('\n'.join(log), encoding='utf-8')
    return 0 if all(o for _, o in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
