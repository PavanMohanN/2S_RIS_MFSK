"""
estimation_crb.py — Stage 4: Statistical Estimation — CRB + Monte Carlo
========================================================================
Derives the Cramér-Rao Bound (CRB) for range, angle, and velocity estimation
and validates via Monte Carlo simulation for the bistatic M-FSK RIS system.

Paper equations (Section V):
    Fisher information matrix:
        [I(θ)]_kl = (2/σ²) Re{ (∂s/∂θ_k)^H (∂s/∂θ_l) }    [Eq. 22]

    Signal model per symbol:
        s[k,n] = A·exp(-j2π·k·Δf·τ)·exp(+j2π·fd·n·Tsw)      [Eq. 16]

    CRB for range R:
        CRB_R = c² / (8π² · SNR_eff · (1/M) Σ_k (k·Δf)²)    [Eq. 24]

    CRB for velocity v_r:
        CRB_v = λ² / (8π² · SNR_eff · (1/N) Σ_n (n·Tsw)²)   [Eq. 26]

    Effective SNR after CPI:
        SNR_eff = SNR_single · M · NSw                          [Eq. 23]

    FFT velocity floor (fundamental limit):
        σ_FFT = λ / (2·N_FFT·Tsw·√(SNR_eff))                  [Eq. 27]

Output figures
--------------
F12 — RMSE vs SNR: range (left), position (right) + CRB    [double-col]
F13 — RMSE vs SNR: velocity + CRB + FFT estimation floor   [single-col]

Output tables
-------------
T7  — Estimation performance summary at operating SNR

Output data
-----------
data/estimation_crb_data.npy

Usage
-----
    python estimation_crb.py
    from estimation_crb import run, compute_crb, monte_carlo_estimation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# IEEE TVT publication style (consistent with Phases 1–3)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':          'serif',
    'font.serif':           ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':            cfg.FIG_FONT,
    'axes.labelsize':       cfg.FIG_LABEL,
    'axes.titlesize':       cfg.FIG_TITLE,
    'legend.fontsize':      cfg.FIG_LEGEND,
    'xtick.labelsize':      cfg.FIG_FONT,
    'ytick.labelsize':      cfg.FIG_FONT,
    'xtick.direction':      'out',
    'ytick.direction':      'out',
    'xtick.major.width':    0.8,
    'ytick.major.width':    0.8,
    'axes.linewidth':       0.8,
    'lines.linewidth':      cfg.FIG_LW,
    'lines.markersize':     cfg.FIG_MS,
    'grid.linewidth':       0.5,
    'grid.alpha':           0.35,
    'grid.linestyle':       '--',
    'grid.color':           '#888888',
    'figure.dpi':           cfg.FIG_DPI,
    'savefig.dpi':          cfg.FIG_DPI,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.03,
    'mathtext.fontset':     'cm',
})

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.TABLE_DIR,  exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Cramér-Rao Bounds
# ═══════════════════════════════════════════════════════════════════════════════

def compute_crb(snr_single_db_array, n_sweeps=None):
    """
    Compute the Cramér–Rao Bounds (CRBs) for range, angle, and velocity
    as a function of single-symbol SNR.

    The signal model:
        s[k, n] = A · exp(-j2π k Δf τ) · exp(+j2π fd n Tsw)

    where τ = R/c  and  fd = 2v_r/λ  (quasi-monostatic approximation).

    FIM derivation [Eq. 22–26]:
    ─────────────────────────────────────────────────────────────────────────
    Parameter   Partial derivative    FIM element
    ─────────────────────────────────────────────────────────────────────────
    τ (range)   ∂s/∂τ  = -j2π k Δf · s        I_ττ = SNR_eff · (2π)² Σ_k(kΔf)²/N_pix
    fd (vel)    ∂s/∂fd = +j2π n Tsw · s        I_ff = SNR_eff · (2π)² Σ_n(nTsw)²/N_pix
    ─────────────────────────────────────────────────────────────────────────

    The CRB is the inverse of the relevant diagonal FIM element:
        CRB_τ = 1 / I_ττ,    CRB_R = c²/4 · CRB_τ
        CRB_fd = 1 / I_ff,   CRB_v = λ²/4 · CRB_fd

    Parameters
    ----------
    snr_single_db_array : array_like  Single-symbol SNR values [dB]
    n_sweeps            : int         Number of sweeps per CPI (default cfg.NSw)

    Returns
    -------
    crb : dict with arrays (len = len(snr_single_db)):
        'snr_db'       single-symbol SNR sweep
        'snr_eff_db'   effective (post-coherent-combining) SNR
        'crb_r_m'      range CRB [m]  (= √(CRB_R^2) = RMSE lower bound)
        'crb_v_ms'     velocity CRB [m/s]
        'crb_pos_m'    position CRB [m]  (combines range + angle CRB)
        'fft_floor_ms' FFT velocity estimation floor [m/s]
        'snr_op_idx'   index of operating-point SNR
    """
    if n_sweeps is None:
        n_sweeps = cfg.NSw

    snr_db  = np.asarray(snr_single_db_array, dtype=float)
    snr_lin = 10.0 ** (snr_db / 10.0)                  # single-symbol SNR (linear)

    M      = cfg.M_TONES
    N      = n_sweeps
    Df     = cfg.DELTA_F
    Tsw    = cfg.TSW
    c      = cfg.C_LIGHT
    lam    = cfg.WAVELENGTH

    # ── Frequency moments (range FIM) ─────────────────────────────────────────
    k_vec  = np.arange(1, M + 1, dtype=float)            # tones 1…M
    mu_f   = np.mean(k_vec * Df)                          # E[kΔf]
    var_f  = np.mean((k_vec * Df) ** 2) - mu_f ** 2      # Var[kΔf]

    # ── Time moments (velocity FIM) ───────────────────────────────────────────
    n_vec  = np.arange(N, dtype=float)                    # sweeps 0…N-1
    mu_t   = np.mean(n_vec * Tsw)                         # E[nTsw]
    var_t  = np.mean((n_vec * Tsw) ** 2) - mu_t ** 2     # Var[nTsw]

    # ── Effective SNR after coherent combination of M×N samples ───────────────
    snr_eff = snr_lin * M * N                             # total SNR (linear)
    snr_eff_db = 10.0 * np.log10(snr_eff)

    # ── CRB for range R (via delay τ) [Eq. 24] ────────────────────────────────
    # I_ττ = SNR_eff · (2π)² · var_f
    # CRB_τ = 1 / I_ττ
    # CRB_R = (c/2)² · CRB_τ  (since τ = 2R/c for monostatic; R/c for bistatic one-way)
    I_tau  = snr_eff * (2.0 * np.pi) ** 2 * var_f        # [Hz^-2]^-1 equivalent
    crb_tau = 1.0 / I_tau                                 # [s²]
    crb_r   = (c / 2.0) * np.sqrt(crb_tau)               # [m]  (one-side path)

    # ── CRB for radial velocity v_r (via Doppler fd) [Eq. 26] ─────────────────
    # I_ff = SNR_eff · (2π)² · var_t
    # CRB_fd = 1 / I_ff
    # CRB_v = (λ/2)² · CRB_fd
    I_fd    = snr_eff * (2.0 * np.pi) ** 2 * var_t       # [s^2]^-1 equivalent
    crb_fd  = 1.0 / I_fd                                  # [Hz²]^-1
    crb_v   = (lam / 2.0) * np.sqrt(crb_fd)              # [m/s]

    # ── CRB for 2-D position (range × bearing) ────────────────────────────────
    # Angle CRB from UPA array: CRB_θ ≈ 1/(SNR_eff · (2π/λ)² · var_d)
    # where var_d = variance of element positions × d = λ/2
    N_ula  = cfg.N_ELEM_X                                 # 16 elements in ULA
    d_elem = cfg.D_ELEM
    pos_n  = np.arange(N_ula, dtype=float) * d_elem
    mu_d   = np.mean(pos_n)
    var_d  = np.mean(pos_n ** 2) - mu_d ** 2             # [m²]

    I_theta = snr_eff * (2.0 * np.pi / lam) ** 2 * var_d
    crb_theta = 1.0 / I_theta                             # [rad²]
    crb_theta_deg = np.sqrt(crb_theta) * (180.0 / np.pi)

    # 2-D position CRB: δpos² = δR² + (R·δθ)²
    crb_pos = np.sqrt(crb_r ** 2 +
                      (cfg.R_RIS_T * np.sqrt(crb_theta)) ** 2)

    # ── FFT velocity floor (RMSE from coarse FFT resolution) [Eq. 27] ─────────
    # σ_FFT ≈ Δv / √(12 · SNR_eff / M)  (triangular interpolation around peak)
    dv_fft = lam / (2.0 * N * Tsw)                       # FFT velocity resolution
    fft_floor = dv_fft / np.sqrt(12.0 * snr_eff / M)
    fft_floor = np.maximum(fft_floor, dv_fft * 0.05)     # floor at 5% of bin width

    # ── Operating point index ─────────────────────────────────────────────────
    op_idx = int(np.argmin(np.abs(snr_db - cfg.SNR_SINGLE_DB)))

    return {
        'snr_db':       snr_db,
        'snr_eff_db':   snr_eff_db,
        'crb_r_m':      crb_r,
        'crb_v_ms':     crb_v,
        'crb_pos_m':    crb_pos,
        'crb_theta_deg':crb_theta_deg,
        'fft_floor_ms': fft_floor,
        'var_f_Hz2':    var_f,
        'var_t_s2':     var_t,
        'snr_op_idx':   op_idx,
        'dv_fft_ms':    dv_fft * np.ones_like(snr_db),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Monte Carlo estimation
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo_estimation(snr_single_db_array, n_trials=None, n_sweeps=None,
                            seed=None):
    """
    Monte Carlo RMSE for range and velocity estimation over the SNR sweep.

    Estimation method:
      • Range : arg max |IFFT_k{y_avg[k]}|  on zero-padded spectrum
                after averaging over N sweeps to boost SNR.
      • Velocity: arg max |FFT_n{phase_diff[n]}|  from peak-phase tracking.
      • Position: propagated from range + angle estimates.

    Parameters
    ----------
    snr_single_db_array : array_like  SNR sweep [dB]
    n_trials            : int         MC trials per SNR point (default cfg.MC_TRIALS)
    n_sweeps            : int         Sweeps per CPI (default cfg.NSw // 16 for speed)
    seed                : int         RNG seed

    Returns
    -------
    mc : dict with arrays (len = len(snr_single_db_array))
        'rmse_r_m'      range RMSE [m]
        'rmse_v_ms'     velocity RMSE [m/s]
        'rmse_pos_m'    2-D position RMSE [m]
        'bias_r_m'      range bias [m]
        'bias_v_ms'     velocity bias [m/s]
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED if seed is None else seed)
    if n_trials  is None: n_trials  = cfg.MC_TRIALS
    if n_sweeps  is None: n_sweeps  = max(256, cfg.NSw // 4)  # 256: dv = 1.2 m/s/bin

    snr_db   = np.asarray(snr_single_db_array, dtype=float)
    n_snr    = len(snr_db)
    M        = cfg.M_TONES
    NFFT_r   = 8 * M                    # range zero-padding factor
    NFFT_v   = n_sweeps                 # Doppler FFT size

    # True parameters
    tau_true = cfg.R_RIS_T / cfg.C_LIGHT    # s
    fd_true  = (2.0 * cfg.V_TARGET / cfg.WAVELENGTH
                * np.cos(cfg.THETA1_RAD - cfg.THETA_T_RAD))  # Hz

    k_vec   = np.arange(1, M + 1, dtype=float)
    n_vec   = np.arange(n_sweeps, dtype=float)

    # Precompute clean signal
    s_range  = np.exp(-1j * 2.0 * np.pi * k_vec * cfg.DELTA_F * tau_true)   # [M]
    s_doppler = np.exp(1j * 2.0 * np.pi * fd_true * n_vec * cfg.TSW)         # [N]
    # Full signal matrix [M × N]
    S_clean  = np.sqrt(1.0) * s_range[:, None] * s_doppler[None, :]

    # DR axes  (consistent with compute_range_doppler: fftshift applied to D, not v_axis)
    dr     = cfg.C_LIGHT / (NFFT_r * cfg.DELTA_F)
    r_axis = np.arange(NFFT_r) * dr
    dv     = cfg.WAVELENGTH / (2.0 * NFFT_v * cfg.TSW)
    # v_axis must be monotonically centred so peak_v indexes into fftshifted D correctly
    v_axis = (np.arange(NFFT_v) - NFFT_v // 2) * dv   # [-N/2·dv … (N/2-1)·dv]

    rmse_r    = np.zeros(n_snr)
    rmse_v    = np.zeros(n_snr)
    rmse_pos  = np.zeros(n_snr)
    bias_r    = np.zeros(n_snr)
    bias_v    = np.zeros(n_snr)

    for i_snr, snr_db_val in enumerate(snr_db):
        snr_lin = 10.0 ** (snr_db_val / 10.0)

        est_r  = np.zeros(n_trials)
        est_v  = np.zeros(n_trials)

        for trial in range(n_trials):
            # Noisy beat matrix
            noise = ((rng.standard_normal((M, n_sweeps)) +
                      1j * rng.standard_normal((M, n_sweeps)))
                     / np.sqrt(2.0))
            Y = np.sqrt(snr_lin) * S_clean + noise

            # ── Range estimation: IFFT peak ───────────────────────────────────
            # Power-average over sweeps first, then find peak
            rp_all  = np.fft.ifft(Y, n=NFFT_r, axis=0)           # [NFFT_r × N]
            rp_avg_pwr = np.mean(np.abs(rp_all) ** 2, axis=1)    # power average
            peak_r  = int(np.argmax(rp_avg_pwr))
            est_r[trial] = r_axis[peak_r]

            # ── Velocity estimation: FFT of phase-coherent sweep axis ──────────
            # Use the strongest range bin for Doppler extraction
            tgt_bin   = peak_r
            phase_seq = rp_all[tgt_bin, :]                        # [N] complex
            # FFT for Doppler
            D = np.fft.fftshift(np.fft.fft(phase_seq, n=NFFT_v))
            peak_v  = int(np.argmax(np.abs(D)))
            est_v[trial] = v_axis[peak_v]

        # ── RMSE and bias ──────────────────────────────────────────────────────
        r_errors      = est_r - cfg.R_RIS_T
        v_errors      = est_v - fd_true * cfg.WAVELENGTH / 2.0
        rmse_r[i_snr] = np.sqrt(np.mean(r_errors ** 2))
        rmse_v[i_snr] = np.sqrt(np.mean(v_errors ** 2))
        bias_r[i_snr] = np.mean(r_errors)
        bias_v[i_snr] = np.mean(v_errors)

        # 2-D position RMSE (propagated from range + angle variance)
        # Angle is estimated from R-RIS array DOA; use CRB-based σ_θ here
        # for the position circle projection:
        sigma_theta_rad = 1.0 / np.sqrt(
            snr_lin * cfg.M_TONES * n_sweeps
            * (2.0 * np.pi / cfg.WAVELENGTH) ** 2
            * np.var(np.arange(cfg.N_ELEM_X) * cfg.D_ELEM))
        rmse_pos[i_snr] = np.sqrt(
            rmse_r[i_snr] ** 2 +
            (cfg.R_RIS_T * sigma_theta_rad) ** 2)

    return {
        'snr_db':    snr_db,
        'rmse_r_m':  rmse_r,
        'rmse_v_ms': rmse_v,
        'rmse_pos_m':rmse_pos,
        'bias_r_m':  bias_r,
        'bias_v_ms': bias_v,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Print estimation summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_estimation_summary(crb, mc):
    sep = '─' * 64
    op  = crb['snr_op_idx']
    print(f'\n{sep}')
    print('  Estimation Performance at Operating Point '
          f'(SNR₁ = {cfg.SNR_SINGLE_DB:.1f} dB)')
    print(sep)
    print(f'  Effective post-CPI SNR        = {crb["snr_eff_db"][op]:.1f} dB')
    print(f'  CRB range                     = {crb["crb_r_m"][op]*100:.2f} cm')
    print(f'  CRB velocity                  = {crb["crb_v_ms"][op]*100:.2f} cm/s')
    print(f'  CRB 2-D position              = {crb["crb_pos_m"][op]*100:.2f} cm')
    op_mc = int(np.argmin(np.abs(mc['snr_db'] - cfg.SNR_SINGLE_DB)))
    print(f'  MC RMSE range                 = {mc["rmse_r_m"][op_mc]*100:.2f} cm')
    print(f'  MC RMSE velocity              = {mc["rmse_v_ms"][op_mc]*100:.2f} cm/s')
    print(f'  MC RMSE position              = {mc["rmse_pos_m"][op_mc]*100:.2f} cm')
    print(f'  RMSE_R / CRB_R (efficiency)   = '
          f'{mc["rmse_r_m"][op_mc] / crb["crb_r_m"][op]:.3f}')
    print(f'  FFT velocity floor            = {crb["fft_floor_ms"][op]*100:.2f} cm/s')
    print(f'  FFT bin resolution Δv         = {crb["dv_fft_ms"][op]*100:.2f} cm/s')
    print(sep + '\n')


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Figure helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, name):
    base = os.path.join(cfg.OUTPUT_DIR, name)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=cfg.FIG_DPI)
    print(f'  Saved: {base}.pdf  |  {base}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Figure F12 — RMSE & CRB for range and position vs SNR
# ═══════════════════════════════════════════════════════════════════════════════

def plot_F12(crb, mc, save=True):
    """
    F12 — Two-panel double-column figure:
      (a) Range RMSE + CRB vs single-symbol SNR
      (b) 2-D position RMSE + position CRB vs SNR
    Vertical line at operating SNR_SINGLE = -14.5 dB.
    Double-column IEEE TVT (7.16 × 2.8 in).
    """
    fig = plt.figure(figsize=(cfg.FIG_WIDTH_2COL, 2.8),
                     constrained_layout=True)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
    snr = crb['snr_db']

    for i_panel, (rmse_key, crb_key, title, ylabel) in enumerate([
        ('rmse_r_m',   'crb_r_m',   '(a) Range estimation',   'RMSE (m)'),
        ('rmse_pos_m', 'crb_pos_m', '(b) 2-D position',       'RMSE (m)'),
    ]):
        ax = fig.add_subplot(gs[i_panel])

        # CRB
        ax.semilogy(snr, crb[crb_key], color=cfg.COLORS['crb'], lw=2,
                    ls='--', label='CRB', zorder=4)

        # MC RMSE
        ax.semilogy(mc['snr_db'], mc[rmse_key], color=cfg.COLORS['on'], lw=2,
                    ls='-', marker='o', ms=3.5, markevery=3,
                    label='MC RMSE', zorder=5)

        # Range resolution floor
        ax.axhline(crb['dv_fft_ms'][0] if i_panel == 0 else
                   crb['crb_pos_m'][-1] * 0.01,
                   color='none')   # invisible spacer for auto-scale

        if i_panel == 0:
            ax.axhline(cfg.C_LIGHT / (2.0 * cfg.M_TONES * cfg.DELTA_F),
                       color=cfg.COLORS['gray'], lw=2, ls=':',
                       label=r'$\Delta R$ (range res.)')

        # Operating-point vertical line
        ax.axvline(cfg.SNR_SINGLE_DB, color=cfg.COLORS['carrier'],
                   lw=2, ls=':', alpha=0.8, label=f'Operating SNR')
        ax.text(cfg.SNR_SINGLE_DB + 0.4, 0.04,
                'SNR$_1$', fontsize=10, color=cfg.COLORS['carrier'],
                va='bottom', transform=ax.get_xaxis_transform())

        ax.set_xlabel('Single-symbol SNR (dB)',fontsize = 10)
        ax.set_ylabel(ylabel,fontsize = 10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlim([snr.min(), snr.max()])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.grid(True, which='major')
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.3)
        ax.legend(loc='upper right', fontsize=9, handlelength=1.8)
        #ax.set_title(title, fontsize=cfg.FIG_TITLE,
         #            fontweight='normal', pad=3)

    if save:
        _save_fig(fig, 'F12_rmse_range_position')
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Figure F13 — Velocity RMSE + CRB + FFT floor vs SNR
# ═══════════════════════════════════════════════════════════════════════════════

def plot_F13(crb, mc, save=True):
    """
    F13 — Single-panel figure:
      Velocity RMSE + CRB + FFT floor vs single-symbol SNR.
    Shows three regimes: noise-dominated, CRB-tracking, FFT-floor-limited.
    Single-column IEEE TVT (3.5 × 2.6 in).
    """
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH_1COL, cfg.FIG_HEIGHT_STD))
    snr = crb['snr_db']

    # CRB
    ax.semilogy(snr, crb['crb_v_ms'], color=cfg.COLORS['crb'], lw=2,
                ls='--', label='CRB velocity', zorder=4)

    # MC RMSE
    ax.semilogy(mc['snr_db'], mc['rmse_v_ms'], color=cfg.COLORS['on'], lw=2,
                ls='-', marker='s', ms=3.5, markevery=3,
                label='MC RMSE', zorder=5)

    # FFT velocity floor
    ax.semilogy(snr, crb['fft_floor_ms'], color=cfg.COLORS['sideband'], lw=2,
                ls='-.', label='FFT floor', zorder=3)

    # Velocity resolution (bin width)
    ax.axhline(crb['dv_fft_ms'][0], color=cfg.COLORS['gray'], lw=2,
               ls=':', label=r'$\Delta v$ (bin width)')

    # Regime annotations
    snr_mid = (snr.min() + snr.max()) / 2.0
    ax.text(snr.min() + 2, crb['crb_v_ms'][0] * 0.5,
            'Noise\ndominated', fontsize=8, color='red',
            ha='left', va='top', style='italic')
    ax.text(snr.max() - 2, crb['dv_fft_ms'][0] * 1.4 + 0.3,
            'FFT\nlimited', fontsize=8, color='red',
            ha='right', va='bottom', style='italic')

    # Operating point
    ax.axvline(cfg.SNR_SINGLE_DB, color=cfg.COLORS['carrier'],
               lw=2, ls=':', alpha=0.8)
    ax.text(cfg.SNR_SINGLE_DB + 0.4, 0.04,
            'SNR$_1$', fontsize=8, color=cfg.COLORS['carrier'],
            va='bottom', transform=ax.get_xaxis_transform())

    ax.set_xlabel('Single-symbol SNR (dB)',fontsize =10)
    ax.set_ylabel('Velocity RMSE / bound (m/s)',fontsize =10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlim([snr.min(), snr.max()])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(True, which='major')
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.3)
    ax.legend(loc='upper right', fontsize=6.5, handlelength=1.8, ncol=2)

    fig.tight_layout(pad=0.4)
    if save:
        _save_fig(fig, 'F13_rmse_velocity')
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Table T7 — Estimation performance summary
# ═══════════════════════════════════════════════════════════════════════════════

def export_table_T7(crb, mc):
    """T7 — Estimation performance at operating point."""
    op  = crb['snr_op_idx']
    op_mc = int(np.argmin(np.abs(mc['snr_db'] - cfg.SNR_SINGLE_DB)))

    header = ['Metric', 'Value', 'Units']
    rows = [
        ['Operating single-symbol SNR',
         f'{cfg.SNR_SINGLE_DB:.1f}', 'dB'],
        ['Post-CPI SNR ($N_{{sw}}$={:d})'.format(cfg.NSw),
         f'{cfg.SNR_CPI_DB:.1f}', 'dB'],
        [r'CRB range $\sqrt{{\rm CRB}_R}$',
         f'{crb["crb_r_m"][op]*100:.2f}', 'cm'],
        [r'CRB velocity $\sqrt{{\rm CRB}_v}$',
         f'{crb["crb_v_ms"][op]*100:.2f}', 'cm/s'],
        [r'CRB position $\sqrt{{\rm CRB}_{\rm pos}}$',
         f'{crb["crb_pos_m"][op]*100:.2f}', 'cm'],
        [r'MC RMSE range',
         f'{mc["rmse_r_m"][op_mc]*100:.2f}', 'cm'],
        [r'MC RMSE velocity',
         f'{mc["rmse_v_ms"][op_mc]*100:.2f}', 'cm/s'],
        [r'MC RMSE position',
         f'{mc["rmse_pos_m"][op_mc]*100:.2f}', 'cm'],
        [r'Estimation efficiency $\eta_R$ = RMSE/CRB',
         f'{mc["rmse_r_m"][op_mc]/crb["crb_r_m"][op]:.3f}', ''],
        [r'FFT velocity floor',
         f'{crb["fft_floor_ms"][op]*100:.2f}', 'cm/s'],
        [r'Range resolution $\Delta R = c/(M\Delta f)$',
         f'{cfg.C_LIGHT/(2*cfg.M_TONES*cfg.DELTA_F):.2f}', 'm'],
        [r'Velocity resolution $\Delta v = \lambda/(2T_{\rm CPI})$',
         f'{crb["dv_fft_ms"][op]*100:.2f}', 'cm/s'],
        [r'MC trials per SNR point',
         f'{cfg.MC_TRIALS}', ''],
    ]

    tex = _tex_table_t7(header, rows)
    csv = _csv_table(header, rows)

    for ext, content in [('.tex', tex), ('.csv', csv)]:
        path = os.path.join(cfg.TABLE_DIR, 'T7_estimation_performance' + ext)
        with open(path, 'w') as fh:
            fh.write(content)
    print(f'  Exported: {os.path.join(cfg.TABLE_DIR, "T7_estimation_performance")}'
          f'.tex + .csv')


def _tex_table_t7(header, rows):
    lines = [
        r'\begin{table}[!t]',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{Estimation Performance Summary at Operating Point}',
        r'\label{tab:estimation_perf}',
        r'\centering',
        r'\begin{tabular}{l r c}',
        r'\hline\hline',
        ' & '.join(header) + r' \\',
        r'\hline',
    ]
    for row in rows:
        lines.append(' & '.join(str(c) for c in row) + r' \\')
    lines += [
        r'\hline\hline',
        r'\multicolumn{3}{l}{\footnotesize '
        r'Operating point: SNR$_1=' + f'{cfg.SNR_SINGLE_DB:.1f}' +
        r'$ dB, $N_{sw}=' + f'{cfg.NSw}' +
        r'$, $M=' + f'{cfg.M_TONES}' + r'$ tones.}' + ' \\\\',
        r'\hline',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def _csv_table(header, rows):
    lines = [','.join(str(h) for h in header)]
    for row in rows:
        lines.append(','.join(str(c) for c in row))
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Data export
# ═══════════════════════════════════════════════════════════════════════════════

def export_data(crb, mc):
    path = os.path.join(cfg.DATA_DIR, 'estimation_crb_data.npy')
    np.save(path, {'crb': crb, 'mc': mc}, allow_pickle=True)
    print(f'  Exported: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(show=False):
    """
    Run Stage 4 — CRB derivation + Monte Carlo RMSE estimation.

    Generates F12 (range/position RMSE + CRB), F13 (velocity), Table T7.

    Parameters
    ----------
    show : bool  If True, call plt.show() after saving.

    Returns
    -------
    Tuple (crb, mc)
    """
    print('\n' + '═' * 60)
    print('  Stage 4 — Statistical Estimation: CRB + Monte Carlo')
    print('  Generating Figures F12, F13 | Table T7')
    print('═' * 60)

    snr_sweep = cfg.SNR_SWEEP_DB     # defined in config: arange(-30, 21, 2)

    # ── CRB ───────────────────────────────────────────────────────────────────
    print(f'\n  Computing CRBs over SNR ∈ [{snr_sweep[0]:.0f}, '
          f'{snr_sweep[-1]:.0f}] dB …')
    crb = compute_crb(snr_sweep, n_sweeps=cfg.NSw)

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    print(f'  Running MC simulation '
          f'({cfg.MC_TRIALS} trials × {len(snr_sweep)} SNR points) …')
    n_sweeps_mc = max(256, cfg.NSw // 4)
    print(f'  [using {n_sweeps_mc} sweeps per CPI, Δv = '
          f'{cfg.WAVELENGTH/(2*n_sweeps_mc*cfg.TSW):.2f} m/s per bin]')
    mc  = monte_carlo_estimation(snr_sweep,
                                 n_trials=cfg.MC_TRIALS,
                                 n_sweeps=n_sweeps_mc)

    print_estimation_summary(crb, mc)

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_F12(crb, mc, save=True)
    plot_F13(crb, mc, save=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    export_table_T7(crb, mc)

    # ── Data ──────────────────────────────────────────────────────────────────
    export_data(crb, mc)

    if show:
        plt.show()
    plt.close('all')

    print('\n  Stage 4 complete — 2 figures + 1 table + data saved.')
    return crb, mc


if __name__ == '__main__':
    run()
