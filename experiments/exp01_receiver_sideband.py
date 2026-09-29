"""
exp01_receiver_sideband.py -- Complex-baseband receiver and sideband separation
===============================================================================
Per-symbol spectrum seen by the I/Q receiver, image-rejection ratio versus
quadrature imbalance (analytic and time-domain), modulator-bandwidth and
tone-plan trade-off, and the double-sideband CRB option.

Paper figures produced: Fig. 3 (F3rev_baseband_spectrum), Fig. S1 (FC_sideband_separation)
Outputs: results/exp01_receiver_sideband/ (data, tables, extra figures), results/figdata/ (figure data),
         paper/figures/ (paper figures)
Run    : python experiments/exp01_receiver_sideband.py [--quick]
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
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging

from mfsk import common as mc
from figures.fig03_baseband_spectrum import fig_baseband_spectrum
from figures.figS01_sideband_separation import fig_sideband_separation

logging.getLogger('fontTools').setLevel(logging.ERROR)


# -----------------------------------------------------------------------------
# I/Q receiver model
# -----------------------------------------------------------------------------
def iq_imbalance_coeffs(gain_db: float, phase_deg: float):
    """Standard receive I/Q imbalance model  y = mu*x + nu*conj(x).
    Gain ratio g and phase error phi applied to the Q branch."""
    g = 10.0 ** (gain_db / 20.0)
    phi = np.deg2rad(phase_deg)
    mu = 0.5 * (1.0 + g * np.exp(-1j * phi))
    nu = 0.5 * (1.0 - g * np.exp(1j * phi))
    return mu, nu


def irr_analytic_db(gain_db, phase_deg):
    g = 10.0 ** (np.asarray(gain_db) / 20.0)
    phi = np.deg2rad(np.asarray(phase_deg))
    num = 1.0 + 2.0 * g * np.cos(phi) + g ** 2
    den = 1.0 - 2.0 * g * np.cos(phi) + g ** 2
    return 10.0 * np.log10(num / np.maximum(den, 1e-30))


def simulate_symbol_correlator(nu_k: float, gain_db: float, phase_deg: float,
                               a_sb: float, a_car: float, dc_offset: complex,
                               fs: float = 4.0e9, tsym: float = mc.TSYM,
                               rng=None):
    """Time-domain simulation of one M-FSK symbol at the Rx (complex baseband,
    LO at f_c).  Returns the correlator outputs on +nu_k (desired bin) for the
    full signal and for each interferer alone, so leakage is measured
    directly rather than assumed."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = int(round(fs * tsym))
    t = np.arange(n) / fs
    ph = rng.uniform(0, 2 * np.pi, 3)
    x_up = a_sb * np.exp(1j * (2 * np.pi * nu_k * t + ph[0]))     # desired  +nu_k
    x_lo = a_sb * np.exp(1j * (-2 * np.pi * nu_k * t + ph[1]))    # image    -nu_k
    x_ca = a_car * np.exp(1j * ph[2]) * np.ones(n)                # carrier  DC
    x_dc = dc_offset * np.ones(n)                                 # Rx DC offset
    mu, nuc = iq_imbalance_coeffs(gain_db, phase_deg)
    ref = np.exp(-1j * 2 * np.pi * nu_k * t) / n

    def corr(x):
        y = mu * x + nuc * np.conj(x)
        return np.sum(y * ref)

    return {
        'desired': corr(x_up), 'image': corr(x_lo),
        'carrier': corr(x_ca), 'dc': corr(x_dc),
        'total': corr(x_up + x_lo + x_ca + x_dc),
    }


def iq_ghost_range_doppler(gain_db, phase_deg, nu, beta_deg=mc.BETA_DEFAULT_DEG):
    """Noise-free CPI cube after per-symbol correlation at +nu_k, including the
    I/Q-imbalance leakage of the co-steered image.  The image returns from the
    same target, so its leakage has the SAME delay slope across tones but the
    CONJUGATE slow-time Doppler: the ghost appears at (same path, -Doppler).
    Returns ghost level (dB re. target), range-peak shift, Doppler bins."""
    p_rx = mc.rx_position(beta_deg)
    L = mc.bistatic_path(mc.P_TARGET, p_rx)
    Ldot = mc.bistatic_path_rate(mc.P_TARGET, p_rx)
    mu, nuc = iq_imbalance_coeffs(gain_db, phase_deg)
    tp = (np.arange(mc.NSW) - (mc.NSW - 1) / 2) * mc.TSW
    k = np.arange(1, nu.size + 1)
    t = tp[None, :] + k[:, None] * mc.TSYM
    tau = (L + Ldot * t) / mc.C_LIGHT                      # path delay over the CPI
    up = np.exp(-1j * 2 * np.pi * (mc.FC + nu[:, None]) * tau)
    lo = np.exp(-1j * 2 * np.pi * (mc.FC - nu[:, None]) * tau)
    # Processing is linear, so the desired echo and the leakage are processed
    # separately and their peaks compared.  (In the summed cube the target's
    # own rectangular-window Doppler sidelobes overlap the ghost cell.)
    nr, nd = 8 * nu.size, 16 * mc.NSW

    def rd(z):
        return np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifft(z, n=nr, axis=0), n=nd, axis=1), axes=1)) ** 2

    Pt, Pg = rd(mu * up), rd(nuc * np.conj(lo))
    irt, idt = np.unravel_index(np.argmax(Pt), Pt.shape)
    irg, idg = np.unravel_index(np.argmax(Pg), Pg.shape)
    fd_axis = np.fft.fftshift(np.fft.fftfreq(nd, mc.TSW))
    ghost_db = 10 * np.log10(Pg.max() / Pt.max())
    ir, ir0, idd, i_mirror = irg, irt, idt, idg
    return float(ghost_db), int(ir - ir0), float(fd_axis[idd]), float(fd_axis[i_mirror])


# -----------------------------------------------------------------------------
# Tone-plan trade-off
# -----------------------------------------------------------------------------
def modulator_response(f, tau_sw):
    """First-order T-RIS modulator transfer (bias network + diode charge
    dynamics lumped into an effective time constant tau_sw)."""
    return 1.0 / (1.0 + 1j * 2 * np.pi * np.asarray(f) * tau_sw)


def tau_required(f_max, loss_db=1.0):
    return np.sqrt(10 ** (loss_db / 10.0) - 1.0) / (2 * np.pi * f_max)


def squint_deg(nu, theta_deg=mc.THETA1_DEG):
    """Magnitude of the exact phase-only squint (see mfsk_common.squint_exact_deg)."""
    return np.abs(mc.squint_exact_deg(nu, theta_deg))


def hpbw_deg(theta_deg=mc.THETA1_DEG, n=mc.N_X):
    return np.degrees(0.886 * mc.LAMBDA / (n * mc.D_ELEM) / np.cos(np.deg2rad(theta_deg)))


def dsb_crbs(snr1_db, nu):
    """Path CRB for SSB, DSB with a calibrated inter-sideband phase (one common
    nuisance phase over +/-nu), and DSB with independent sideband phases."""
    ssb = mc.crb_path(snr1_db, nu=nu)
    both = np.concatenate([-nu[::-1], nu])
    dsb_coh = mc.crb_path(snr1_db, nu=both)
    # independent phases: FIM is the sum of two independent SSB FIMs
    fim_one = (mc.C_LIGHT / ssb) ** 2
    dsb_ind = mc.C_LIGHT / np.sqrt(2.0 * fim_one)
    return float(ssb), float(dsb_coh), float(dsb_ind)


def dsb_mc_check(snr1_db, nu, n_trials, rng, calibrated=True):
    """Monte Carlo of the ML path estimator on the CPI-integrated tone phasors
    z_k = sqrt(SNR_CPI) e^{j phi} e^{-j 2 pi nu_k L/c} + w  (sufficient stat.)."""
    s = np.sqrt(mc.snr_cpi_lin(snr1_db))
    P = mc.path_ambiguity()
    L_true = 6.0 + rng.uniform(-0.5, 0.5, n_trials)
    both = np.concatenate([-nu[::-1], nu])
    M2 = both.size
    ph_u = rng.uniform(0, 2 * np.pi, n_trials)
    ph_l = ph_u if calibrated else rng.uniform(0, 2 * np.pi, n_trials)
    phase0 = np.where(both[None, :] > 0, ph_u[:, None], ph_l[:, None])
    z = (s * np.exp(1j * phase0) * np.exp(-1j * 2 * np.pi * both[None, :] * L_true[:, None] / mc.C_LIGHT)
         + (rng.standard_normal((n_trials, M2)) + 1j * rng.standard_normal((n_trials, M2))) / np.sqrt(2))
    grid = np.linspace(0, P, 24001)[:-1]
    E = np.exp(1j * 2 * np.pi * both[:, None] * grid[None, :] / mc.C_LIGHT)
    upper = both > 0
    if calibrated:
        spec = np.abs(z @ E) ** 2
    else:
        spec = np.abs(z[:, upper] @ E[upper]) ** 2 + np.abs(z[:, ~upper] @ E[~upper]) ** 2
    L_hat = grid[np.argmax(spec, axis=1)]
    err = (L_hat - L_true + P / 2) % P - P / 2
    return float(np.sqrt(np.mean(err ** 2)))


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------




def fig_dsb_profile(out_fig, nu):
    """Noise-free range profiles: SSB (submitted), DSB with independent sideband
    phases (incoherent sum, +3 dB) and DSB with calibrated phase (half the
    main lobe).  Gate-discussion figure for option D3."""
    P = mc.path_ambiguity()
    grid = np.linspace(0, P, 6000)
    L0 = 3.5
    both = np.concatenate([-nu[::-1], nu])

    def prof(f):
        z = np.exp(-1j * 2 * np.pi * f * L0 / mc.C_LIGHT)
        return np.abs(np.exp(1j * 2 * np.pi * np.outer(grid, f) / mc.C_LIGHT) @ z) ** 2

    ssb = prof(nu)
    ind = prof(nu) + prof(-nu)
    coh = prof(both)
    fig, ax = plt.subplots(figsize=(mc.IEEE_COL_W, 1.9))
    for y, c, ls, lab in [(ssb, mc.COLORS['coarse'], '-', 'SSB (submitted)'),
                          (ind, mc.COLORS['refined'], '--', 'DSB, independent phases'),
                          (coh, mc.COLORS['green'], '-', 'DSB, calibrated phase')]:
        ax.plot(grid, 10 * np.log10(y / y.max() + 1e-12), color=c, ls=ls, lw=1.0, label=lab)
    ax.axvline(L0, color='0.4', lw=0.5, ls=':')
    ax.set_xlim(L0 - 3, L0 + 3)
    ax.set_ylim(-40, 3)
    ax.set_xlabel('Bistatic path (m)')
    ax.set_ylabel('Range profile (dB)')
    ax.grid(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=5.6,
              handlelength=1.4, columnspacing=0.7, frameon=False)
    return mc.save_fig(fig, out_fig, 'FC_opt_dsb_profile')


# -----------------------------------------------------------------------------
# Table
# -----------------------------------------------------------------------------
def table_tone_options(out_tab, options):
    rows = []
    for o in options:
        rows.append(
            f"{o['f_off_mhz']:d} & {o['f_min_mhz']:d}--{o['f_max_mhz']:d} & "
            f"{o['guard_mhz']:d} ({o['guard_pct']:.2f}\\,\\%) & "
            f"{o['tau_req_ns']:.2f} & {o['squint_deg']:.2f} ({o['squint_pct_hpbw']:.1f}\\,\\%) & "
            f"{o['res_m']:.2f} & {o['crb_path_cm']:.2f} \\\\")
    tex = r"""\begin{table}[!t]
\centering
\caption{Tone-plan options: RF guard band versus required T-RIS modulation bandwidth}
\label{tab:tone_plan}
\footnotesize
\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{@{}ccccccc@{}}
\hline\hline
$f_{\mathrm{off}}$ & Tones & Image guard & $\tau_{\mathrm{sw}}^{\max}$ & Max squint & $\Delta R_{\mathrm{path}}$ & $\sqrt{\mathrm{CRB}_L}$ \\
(MHz) & (MHz) & (MHz) & (ns) & (deg, \% HPBW) & (m) & (cm) \\
\hline
""" + "\n".join(rows) + r"""
\hline\hline
\multicolumn{7}{@{}p{0.95\columnwidth}@{}}{\scriptsize $\tau_{\mathrm{sw}}^{\max}$: largest effective modulator time constant keeping the sideband loss at $f_{\max}$ within 1\,dB. Squint at $\theta_1=60^\circ$. Resolution and CRB are independent of $f_{\mathrm{off}}$; operating point $\mathrm{SNR}_1=OPSNR$\,dB.}
\end{tabular}
\end{table}
"""
    out = Path(out_tab)
    out.mkdir(parents=True, exist_ok=True)
    tex = tex.replace('OPSNR', f'{mc.SNR1_DB:.1f}')
    (out / 'TC_tone_plan_options.tex').write_text(tex, encoding='utf-8')


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', default=str(ROOT / 'results' / 'exp01_receiver_sideband'))
    ap.add_argument('--quick', action='store_true', help='fewer MC trials')
    ap.add_argument('--f-off-options', type=float, nargs='+', default=[0.0, 75.0, 150.0],
                    help='candidate tone offsets in MHz')
    ap.add_argument('--f-off-selected', type=float, default=0.0,
                    help='tone offset (MHz) used for the revised Fig. 3')
    ap.add_argument('--typ-gain-db', type=float, default=0.2)
    ap.add_argument('--typ-phase-deg', type=float, default=1.0)
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
    rng = np.random.default_rng(mc.RANDOM_SEED)
    log = []

    def say(s=''):
        print(s)
        log.append(s)

    say('=' * 74)
    say(' Receiver architecture and tone plan')
    say('=' * 74)
    mc.check_against_legacy_config(legacy_dir=mc.find_legacy_dir(__file__))

    # ---- 1. requirement on image rejection -----------------------------------
    # The I/Q mirror of the image lands on the desired tone with the conjugate
    # range phase -> after the range IFFT it forms a ghost at the mirror path.
    # It must stay below the noise floor of the M-tone coherent profile.
    irr_req = mc.SNR_CPI_DB + mc.db10(mc.M_TONES)
    say(f'\n[1] Image-rejection requirement: Doppler-mirror ghost below the noise of')
    say(f'    one range-Doppler cell (the leakage shares the target delay, not its Doppler)')
    say(f'    SNR1 = {mc.SNR1_DB:.1f} dB: SNR_CPI + 10log10(M) = {mc.SNR_CPI_DB:.1f} + {mc.db10(mc.M_TONES):.1f} = {irr_req:.1f} dB')

    # ---- 2. time-domain correlator validation --------------------------------
    say('\n[2] Time-domain symbol-correlator validation (LO at f_c, complex baseband)')
    a_sb, a_car = abs(mc.t_sb(mc.FC)) / 2.0, abs(mc.t_mean(mc.FC))
    nu = mc.tone_offsets(0.0)
    cyc = mc.DELTA_F * mc.TSYM
    say(f'    DELTA_F * T_sym = {cyc:.3f} cycles -> tones/image/carrier orthogonal: '
        f'{"YES" if abs(cyc - round(cyc)) < 1e-9 else "NO"}')
    grid_pts = [(g, p) for g in (0.0, 0.1, 0.2, 0.5, 1.0) for p in (0.0, 0.5, 1.0, 2.0, 5.0)]
    if args.quick:
        grid_pts = grid_pts[::3]
    sim_pts, max_dev = [], 0.0
    for g, p in grid_pts:
        r = simulate_symbol_correlator(nu[0], g, p, a_sb, a_car, dc_offset=0.05 + 0.03j, rng=rng)
        sir = mc.db20(r['desired'] / max(abs(r['image']), 1e-300)) if abs(r['image']) > 0 else np.inf
        car = mc.db20(r['carrier'] / r['desired'])
        dcl = mc.db20(r['dc'] / r['desired'])
        ana = irr_analytic_db(g, p)
        # perfect balance (g = p = 0) gives infinite rejection: both values sit
        # at the float64 floor (~300 dB) and are excluded from the comparison
        if np.isfinite(sir) and np.isfinite(ana) and ana < 150.0:
            max_dev = max(max_dev, abs(sir - ana))
        sim_pts.append({'gain_db': g, 'phase_deg': p, 'sir_sim_db': float(sir),
                        'irr_analytic_db': float(ana), 'carrier_leak_db': float(car),
                        'dc_leak_db': float(dcl)})
    typ_irr = float(irr_analytic_db(args.typ_gain_db, args.typ_phase_deg))
    r_typ = simulate_symbol_correlator(nu[0], args.typ_gain_db, args.typ_phase_deg,
                                       a_sb, a_car, dc_offset=0.05 + 0.03j, rng=rng)
    car_leak = float(mc.db20(r_typ['carrier'] / r_typ['desired']))
    say(f'    simulated SIR vs analytic IRR: max deviation {max_dev:.4f} dB '
        f'({len(sim_pts)} points; ideal-balance point excluded, both at float floor)')
    say(f'    carrier + direct-path leakage into the desired bin: {car_leak:.0f} dB (numerical floor)')
    say(f'    IRR at typical {args.typ_gain_db} dB / {args.typ_phase_deg} deg imbalance: {typ_irr:.1f} dB '
        f'-> margin {typ_irr - irr_req:+.1f} dB over requirement')
    ghost_db, rshift, fd_t, fd_g = iq_ghost_range_doppler(args.typ_gain_db, args.typ_phase_deg, nu)
    say(f'    range-Doppler check: target at f_D = {fd_t:+.1f} Hz, ghost at {fd_g:+.1f} Hz,')
    say(f'      ghost level {ghost_db:.1f} dB re. target (analytic -IRR = {-typ_irr:.1f} dB),'
        f' range-peak shift {rshift} bins')
    worst_ok = [p for p in sim_pts if p['irr_analytic_db'] >= irr_req]
    say(f'    grid points meeting requirement: {len(worst_ok)}/{len(sim_pts)}')

    # ---- 3. tone-plan options --------------------------------------------------
    say('\n[3] Tone-plan options')
    say('    f_off  tones(MHz)   guard(MHz, %)     tau_req(ns)  squint(deg,%HPBW)  dR(m)  CRB_L(cm)')
    hp = hpbw_deg()
    options = []
    for fo in args.f_off_options:
        nu_o = mc.tone_offsets(fo * 1e6)
        fmax, fmin = nu_o.max(), nu_o.min()
        o = dict(f_off_mhz=int(round(fo)), f_min_mhz=int(round(fmin / 1e6)), f_max_mhz=int(round(fmax / 1e6)),
                 guard_mhz=int(round(2 * fmin / 1e6)), guard_pct=float(2 * fmin / mc.FC * 100),
                 f_max_hz=float(fmax), tau_req_ns=float(tau_required(fmax) * 1e9),
                 squint_deg=float(squint_deg(fmax)), squint_pct_hpbw=float(squint_deg(fmax) / hp * 100),
                 res_m=float(mc.path_resolution()), crb_path_cm=float(mc.crb_path(mc.SNR1_DB, nu=nu_o) * 100),
                 tsb_db_fmax=float(mc.db20(mc.t_sb(mc.FC + fmax))))
        options.append(o)
        say(f"    {o['f_off_mhz']:4d}   {o['f_min_mhz']:3d}-{o['f_max_mhz']:3d}    "
            f"{o['guard_mhz']:4d} ({o['guard_pct']:.2f}%)      {o['tau_req_ns']:.3f}        "
            f"{o['squint_deg']:.2f} ({o['squint_pct_hpbw']:.1f}%)        {o['res_m']:.2f}   {o['crb_path_cm']:.3f}")
    tsb_spread = max(o['tsb_db_fmax'] for o in options) - min(o['tsb_db_fmax'] for o in options)

    # ---- 4. DSB option ----------------------------------------------------------
    say('\n[4] Option: coherent use of the co-steered image (double-sideband)')
    ssb, dsb_c, dsb_i = dsb_crbs(mc.SNR1_DB, nu)
    ntr = 400 if args.quick else 3000
    rm_c = dsb_mc_check(mc.SNR1_DB, nu, ntr, rng, calibrated=True)
    rm_i = dsb_mc_check(mc.SNR1_DB, nu, ntr, rng, calibrated=False)
    say(f'    path CRB  SSB (submitted)            : {ssb * 100:.3f} cm')
    say(f'    path CRB  DSB, independent phases    : {dsb_i * 100:.3f} cm  (x{ssb / dsb_i:.2f}, +3 dB, no calibration)')
    say(f'    path CRB  DSB, calibrated phase      : {dsb_c * 100:.3f} cm  (x{ssb / dsb_c:.2f})')
    say(f'    ML Monte Carlo ({ntr} trials): DSB-indep {rm_i * 100:.3f} cm, DSB-calibrated {rm_c * 100:.3f} cm')
    say(f'    DSB total-path resolution: {mc.C_LIGHT / (2 * nu.max()):.2f} m (vs {mc.path_resolution():.2f} m SSB)')

    # ---- 5. Figures and table -----------------------------------------------------
    say('\n[5] Writing figures and tables')
    p1 = fig_baseband_spectrum(fdir, args.f_off_selected * 1e6)
    p2 = fig_sideband_separation(fdir, sim_pts, irr_req, (args.typ_gain_db, args.typ_phase_deg), options)
    p3 = fig_dsb_profile(fdir, nu)
    table_tone_options(tdir, options)
    for p in (p1, p2, p3):
        say(f'    {p}')
    say(f"    {tdir / 'TC_tone_plan_options.tex'}")

    # ---- 6. Gate 1 --------------------------------------------------------------
    say('\n[checks] Validation')
    res_same = len({round(o['res_m'], 6) for o in options}) == 1
    crb_same = (max(o['crb_path_cm'] for o in options) - min(o['crb_path_cm'] for o in options)) < 1e-9
    squint_ok = all(o['squint_deg'] < hp / 2 for o in options)
    irr_ok = typ_irr >= irr_req
    val_ok = max_dev < 0.05
    checks = [('path resolution identical across tone plans', res_same),
              ('path CRB identical across tone plans', crb_same),
              (f'T_sb(f) flat across plans (spread {tsb_spread:.3f} dB < 0.1 dB)', tsb_spread < 0.1),
              ('worst-tone squint < HPBW/2 for every plan', squint_ok),
              ('typical-imbalance IRR meets requirement', irr_ok),
              ('time-domain SIR matches analytic IRR (< 0.05 dB)', val_ok),
              ('I/Q leakage: ghost only at mirror Doppler (-IRR), no range shift',
               abs(ghost_db + typ_irr) < 0.5 and rshift == 0)]
    for name, ok in checks:
        say(f'    [{"PASS" if ok else "FAIL"}] {name}')
    say('\n    Decision for the gate (see PHASE1_2_GATE_REPORT.md):')
    say('    D1  tone plan: keep f_off = 0 unless the T-RIS element can meet '
        f"tau_sw <= {options[-1]['tau_req_ns']:.2f} ns (f_off = {options[-1]['f_off_mhz']} MHz)")
    say(f'    time {time.time() - t0:.1f} s')

    results = {
        'snr1_db': float(mc.SNR1_DB), 'snr_cpi_db': float(mc.SNR_CPI_DB),
        'irr_requirement_db': float(irr_req), 'typical_imbalance': [args.typ_gain_db, args.typ_phase_deg],
        'typical_irr_db': typ_irr, 'irr_margin_db': typ_irr - irr_req,
        'carrier_leak_db': car_leak, 'correlator_validation_max_dev_db': max_dev,
        'iq_ghost_db': ghost_db, 'iq_ghost_range_shift_bins': rshift,
        'iq_ghost_fd_target_hz': fd_t, 'iq_ghost_fd_mirror_hz': fd_g,
        'sim_points': sim_pts, 'tone_options': options, 'tsb_spread_db': tsb_spread,
        'hpbw_deg': hp,
        'dsb': {'crb_ssb_cm': ssb * 100, 'crb_dsb_indep_cm': dsb_i * 100, 'crb_dsb_cal_cm': dsb_c * 100,
                'mc_dsb_indep_cm': rm_i * 100, 'mc_dsb_cal_cm': rm_c * 100,
                'res_dsb_m': mc.C_LIGHT / (2 * nu.max())},
        'gate1': {n: bool(o) for n, o in checks},
    }
    (ddir / 'exp01_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (ddir / 'exp01_summary.txt').write_text('\n'.join(log), encoding='utf-8')
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
