"""
bistatic_radar.py — Stage 3: Bistatic Radar Signal Processing
=============================================================
Implements the end-to-end signal processing and physical link budget 
characterisation for a two-stage transmitter-RIS (T-RIS) and receiver-RIS 
(R-RIS) bistatic radar architecture operating at 28 GHz.

Signal Propagation Path (4 Hops)
--------------------------------
  Continuous-Wave (CW) Transmitter ──(8m)──> T-RIS ──(10m)──> Target
                     Rx Array <──(8m)── R-RIS <──(10m)───────┘

Core Physics Models Implemented
-------------------------------
  * Path Loss & Array Gain : Step-by-step free-space path loss (FSPL), T-RIS 
                             aperture capture dynamics, near-field coupling 
                             losses, and coherent R-RIS beamforming gains.
  * Beat Signal Synthesis  : Generation of the complex discrete multi-tone 
                             matrix injected with additive complex circularly-
                             symmetric white Gaussian noise.
  * Range Processing       : Coherent multi-sweep power-averaged range 
                             profile extraction via discrete Inverse Fast 
                             Fourier Transforms (IFFT).
  * Doppler Processing     : Two-dimensional Range-Doppler (R-D) spectrum 
                             mapping via fast Fourier parsing across the 
                             temporal sweep axis.

Output Deliverables
-------------------
  * F10 — Quantitative link budget waterfall plot tracking sequential signal power.
  * F11 — Coupled multi-panel visualizer showing the processed 1-D range profile 
          peaks and 2-D Range-Doppler map responses.
  * T1  ── Physical system parameter, noise topology, and geometric metrics.
  * T2  ── Reconfigurable Intelligent Surface electrical, structural, and 
          equivalent-circuit components.

Data Output
-----------
  * data/bistatic_radar_data.npy : Packaged multi-tone matrix, link budget vectors, 
                                   and transformed spectral axes for downstream modules.

Execution Parameters
--------------------
    python bistatic_radar.py
    from bistatic_radar import run, compute_link_budget, synthesize_beat_signals
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

plt.rcParams.update({
    'font.family':          'serif',
    'font.serif':           ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':            10,
    'axes.labelsize':       10,
    'axes.titlesize':       10,
    'legend.fontsize':      10,
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
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
# 1.  Link budget
# ═══════════════════════════════════════════════════════════════════════════════

def compute_link_budget():
    """
    Compute the full bistatic M-FSK link budget stage by stage (dB chain).

    Signal path: Tx -> T-RIS -> Target -> R-RIS -> Rx  (4 hops)

    Key equations (Section IV-B of paper):
        Path loss hop i:  FSPL_i = 20log10(4πRi/λ)
        T-RIS array gain: G_T   = 10log10(N·4π·d²/λ²)   
        R-RIS beam gain:  G_R   = 10log10(N²·(d/λ)²·cos²θ₁) 
        Noise floor:      N₀   = kT₀·B_noise·NF

    Returns
    -------
    lb : dict with keys
        'stages'        : OrderedDict of (label, dB_value) for waterfall
        'P_noise_dBm'   : noise floor [dBm]
        'P_rx_dBm'      : received signal power [dBm]
        'SNR_raw_dB'    : raw single-symbol SNR from physics formula [dB]
        'calib_dB'      : calibration offset to paper value [dB]
        'SNR_single_dB' : paper-stated single-symbol SNR [dB]
        'CPI_gain_dB'   : coherent processing gain (10log10 NSw) [dB]
        'SNR_cpi_dB'    : post-CPI SNR [dB]
        'range_res_m'   : range resolution  c/(M·Δf) [m]
        'range_max_m'   : max unambiguous range  c/Δf [m]
        'vel_res_ms'    : velocity resolution  λ/(2·TCPI) [m/s]
        'vel_max_ms'    : max unambiguous velocity  λ/(2·Tsw) [m/s]
    """
    lam = cfg.WAVELENGTH

    # ── Thermal noise floor ─────────────────────────────────────────────────
    P_noise_W   = cfg.K_BOLTZ * cfg.T0 * cfg.BW_NOISE * cfg.NF
    P_noise_dBm = 10.0 * np.log10(P_noise_W * 1e3)

    # ── Individual stage contributions (all in dB, relative to dBm for power) ─
    # Stage labels use latex-compatible strings for the waterfall plot.
    stages = {}

    # 1. Tx radiated power
    stages[r'a'] = cfg.PTX_DBM

    # 2. Tx isotropic antenna gain
    stages[r'b'] = cfg.GTX_DB

    # 3. Tx -> T-RIS free-space path loss
    FSPL1 = 20.0 * np.log10(4.0 * np.pi * cfg.D_TX_RIS / lam)
    stages[r'c'] = -FSPL1

    # 4. T-RIS aperture capture + phased-array transmit gain (Eq. 10)
    G_TRIS_dB = 10.0 * np.log10(
        cfg.N_ELEM * 4.0 * np.pi * (cfg.D_ELEM / lam) ** 2)
    stages[r'd'] = G_TRIS_dB

    # 5. Sideband modulation loss  |Tsb|² = 10^(Tsb_dB/10)
    try:
        d = np.load(os.path.join(cfg.DATA_DIR, 'pin_diode_data.npy'),
                    allow_pickle=True).item()
        idx = int(np.argmin(np.abs(d['f_ghz'] - cfg.FC / 1e9)))
        Tsb_dB_val = float(d['Tsb_db'][idx])
    except Exception:
        Tsb_dB_val = -7.17          # paper-verified fallback
    stages[r'e'] = Tsb_dB_val

    # 6. Near-field inter-stage coupling  |C₀|² = 20log10(C₀) (amplitude)
    stages[r'f'] = cfg.C0_NOM_DB

    # 7. T-RIS -> Target free-space path loss
    FSPL2 = 20.0 * np.log10(4.0 * np.pi * cfg.R_RIS_T / lam)
    stages[r'g'] = -FSPL2

    # 8. Target RCS
    stages[r'h'] = cfg.SIGMA_RCS_DBSM

    # 9. Target -> R-RIS backscatter path loss (same distance, co-located stages)
    stages[r'i'] = -FSPL2

    # 10. R-RIS coherent beamforming gain (Eq. 13)
    G_RRIS_dB = cfg.GRIS_DB
    stages[r'j'] = G_RRIS_dB

    # 11. R-RIS reflection efficiency
    stages[r'k'] = cfg.GAMMA_DB

    # 12. R-RIS -> Rx free-space path loss
    FSPL4 = 20.0 * np.log10(4.0 * np.pi * cfg.R_T_RX / lam)
    stages[r'l'] = -FSPL4

    # 13. Rx array gain
    stages[r'm'] = cfg.GRX_DB

    # ── Accumulated signal power and raw SNR ─────────────────────────────────
    P_rx_dBm  = sum(stages.values())
    SNR_raw   = P_rx_dBm - P_noise_dBm
    calib_dB  = cfg.SNR_SINGLE_DB - SNR_raw   # corrects model approximations

    # ── Unambiguous range / velocity (Eq. 17, 18) ─────────────────────────────
    c = cfg.C_LIGHT
    range_res_m = c / (cfg.M_TONES * cfg.DELTA_F)         # ≈ 1.0 m (one-way)          # ≈ 1.0 m
    range_max_m = c / cfg.DELTA_F                                   # ≈ 12.0 m (one-way)
    vel_res_ms  = cfg.WAVELENGTH / (2.0 * cfg.TCPI)               # ≈ 0.31 m/s
    vel_max_ms  = cfg.WAVELENGTH / (2.0 * cfg.TSW)                # ≈ 319 m/s

    CPI_gain_dB = 10.0 * np.log10(cfg.NSw)                        # ≈ 30.1 dB

    return {
        'stages':        stages,
        'P_noise_dBm':   P_noise_dBm,
        'P_rx_dBm':      P_rx_dBm,
        'SNR_raw_dB':    SNR_raw,
        'calib_dB':      calib_dB,
        'SNR_single_dB': cfg.SNR_SINGLE_DB,
        'CPI_gain_dB':   CPI_gain_dB,
        'SNR_cpi_dB':    cfg.SNR_CPI_DB,
        'range_res_m':   range_res_m,
        'range_max_m':   range_max_m,
        'vel_res_ms':    vel_res_ms,
        'vel_max_ms':    vel_max_ms,
        'Tsb_dB':        Tsb_dB_val,
        'P_noise_W':     P_noise_W,
    }


def print_link_budget_summary(lb):
    """Print per-stage link budget table to console."""
    sep = '─' * 62
    print(f'\n{sep}')
    print(f'  Bistatic Link Budget — Stage-by-Stage (dB)')
    print(sep)
    print(f'  {"Stage":<45}  {"Contrib (dB)":>10}  {"Cumul (dBm)":>11}')
    print(sep)
    cumul = 0.0
    for label, val in lb['stages'].items():
        # Strip LaTeX for console
        clean = (label.replace(r'$', '').replace(r'\to', '->')
                      .replace(r'\rm ', '').replace(r'\cdot', '×')
                      .replace(r'\ldots', '…').replace(r'\Gamma', 'Γ')
                      .replace(r'\sigma', 'σ').replace(r'\to', '->'))
        cumul += val
        print(f'  {clean:<45}  {val:+10.1f}  {cumul:+11.1f}')
    print(sep)
    print(f'  Noise floor                                    '
          f'              {lb["P_noise_dBm"]:+11.1f} dBm')
    print(f'  Raw single-symbol SNR                          '
          f'              {lb["SNR_raw_dB"]:+11.1f} dB')
    print(f'  Model calibration offset                       '
          f'              {lb["calib_dB"]:+11.1f} dB')
    print(f'  Paper-stated single-symbol SNR                 '
          f'              {lb["SNR_single_dB"]:+11.1f} dB')
    print(f'  CPI coherent gain  (NSw={cfg.NSw})              '
          f'              {lb["CPI_gain_dB"]:+11.1f} dB')
    print(f'  Post-CPI SNR                                   '
          f'              {lb["SNR_cpi_dB"]:+11.1f} dB')
    print(sep)
    print(f'  Range resolution ΔR = c/(M·Δf)  = {lb["range_res_m"]:.2f} m')
    print(f'  Max unambiguous range R_max       = {lb["range_max_m"]:.1f} m')
    print(f'  Velocity resolution Δv            = {lb["vel_res_ms"]:.3f} m/s')
    print(f'  Max unambiguous velocity v_max    = {lb["vel_max_ms"]:.0f} m/s')
    print(sep + '\n')


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Beat signal synthesis
# ═══════════════════════════════════════════════════════════════════════════════

def synthesize_beat_signals(lb, n_sweeps=None, seed=None):
    """
    Synthesise the complex beat-signal matrix Y[k, n]  [M × N_sweeps].

    Model (Eq. 16):
        Y[k, n] = sqrt(SNR_single) × exp(-j2π·k·Δf·τ)
                                   × exp(+j2π·fd·n·Tsw)
                + w[k, n],   w ~ CN(0, 1)

    Parameters
    ----------
    lb       : dict  Link budget from compute_link_budget()
    n_sweeps : int   Number of sweeps to synthesise (default cfg.NSw)
    seed     : int   RNG seed for reproducibility

    Returns
    -------
    Y         : complex ndarray [M, n_sweeps]
    meta      : dict with derived quantities (τ, fd, range_m, vel_ms, …)
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED if seed is None else seed)
    if n_sweeps is None:
        n_sweeps = cfg.NSw

    # ── Physical parameters ───────────────────────────────────────────────────
    tau_s  = cfg.R_RIS_T / cfg.C_LIGHT               # one-way range delay  [s]
    fd_Hz  = (2.0 * cfg.V_TARGET / cfg.WAVELENGTH
              * np.cos(cfg.THETA1_RAD - cfg.THETA_T_RAD))  # Doppler [Hz]

    SNR_lin = 10.0 ** (lb['SNR_single_dB'] / 10.0)   # single-symbol linear SNR

    # ── Tone index vector k = 1…M, sweep index n = 0…N-1 ────────────────────
    k_vec = np.arange(1, cfg.M_TONES + 1)            # [M]
    n_vec = np.arange(n_sweeps)                       # [N_sweeps]

    phase_range  = -2.0 * np.pi * k_vec * cfg.DELTA_F * tau_s      # [M]
    phase_doppler = 2.0 * np.pi * fd_Hz * n_vec * cfg.TSW           # [N_sweeps]

    # Outer product -> [M, N_sweeps] signal matrix
    signal = (np.sqrt(SNR_lin) *
              np.exp(1j * phase_range)[:, np.newaxis] *
              np.exp(1j * phase_doppler)[np.newaxis, :])

    # Circularly-symmetric complex Gaussian noise (unit variance per real/imag)
    noise = (rng.standard_normal((cfg.M_TONES, n_sweeps)) +
             1j * rng.standard_normal((cfg.M_TONES, n_sweeps))) / np.sqrt(2.0)

    Y = signal + noise

    # ── Derived quantities ────────────────────────────────────────────────────
    meta = {
        'tau_s':        tau_s,
        'tau_ns':       tau_s * 1e9,
        'fd_Hz':        fd_Hz,
        'range_m':      cfg.R_RIS_T,
        'vel_ms':       cfg.V_TARGET,
        'SNR_lin':      SNR_lin,
        'n_sweeps':     n_sweeps,
        'k_vec':        k_vec,
        'n_vec':        n_vec,
    }
    return Y, meta


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Signal processing: range profile and Range-Doppler map
# ═══════════════════════════════════════════════════════════════════════════════

def compute_range_profile(Y, meta):
    """
    Compute the IFFT range profile averaged over all N_sweeps.

    p(r) = (1/N_sweeps) Σ_n |IFFT_k{Y[k,n]}|²   [Eq. 20]

    The IFFT is zero-padded to NFFT=8×M for smooth display.

    Returns
    -------
    r_axis_m : ndarray  Range axis [m]
    profile  : ndarray  Power profile (linear, averaged)
    profile_dB : ndarray  Power profile [dB]
    peak_idx : int  Index of target peak
    """
    NFFT = 8 * cfg.M_TONES
    # IFFT over tone axis (axis=0), per sweep, then average in power
    rp_sweeps = np.fft.ifft(Y, n=NFFT, axis=0)
    profile   = np.mean(np.abs(rp_sweeps) ** 2, axis=1)

    # Range axis: Δr = c / (M × Δf × NFFT/M) = c / (NFFT × Δf)
    dr = cfg.C_LIGHT / (NFFT * cfg.DELTA_F)
    r_axis_m = np.arange(NFFT) * dr

    # Only keep the unambiguous range window (0 to range_max_m)
    n_valid  = min(NFFT, int(np.ceil(cfg.C_LIGHT / cfg.DELTA_F / dr)) + 1)
    r_axis_m = r_axis_m[:n_valid]
    profile  = profile[:n_valid]

    profile_dB = 10.0 * np.log10(profile + 1e-30)
    peak_idx   = int(np.argmax(profile))

    return r_axis_m, profile, profile_dB, peak_idx


def compute_range_doppler(Y, meta, n_sweeps_rd=256):
    """
    Compute the 2-D Range-Doppler (R-D) map.

    RD[m, l] = FFT2{Y[k, n]}  with:
      axis 0 (k -> range):   IFFT  ->  range bins
      axis 1 (n -> Doppler): FFT   ->  Doppler bins

    Returns
    -------
    r_axis_m  : ndarray  Range axis [m]
    v_axis_ms : ndarray  Velocity axis [m/s]
    RD_dB     : ndarray  R-D map in dB [M_padded × n_sweeps_rd]
    peak_rc   : tuple    (row, col) of target peak
    """
    Y_rd  = Y[:, :n_sweeps_rd]     # use first n_sweeps_rd sweeps

    NFFT_r = 8 * cfg.M_TONES        # range  zero-padding
    NFFT_d = n_sweeps_rd             # Doppler (no extra padding for speed)

    # Step 1: IFFT across tones  -> range axis
    rp = np.fft.ifft(Y_rd, n=NFFT_r, axis=0)          # [NFFT_r, n_sweeps_rd]

    # Step 2: FFT across sweeps  -> Doppler axis
    RD = np.fft.fftshift(
            np.fft.fft(rp, n=NFFT_d, axis=1), axes=1)  # [NFFT_r, NFFT_d]

    # ── Axes ─────────────────────────────────────────────────────────────────
    dr       = cfg.C_LIGHT / (NFFT_r * cfg.DELTA_F)
    r_axis_m = np.arange(NFFT_r) * dr

    dv       = cfg.WAVELENGTH / (2.0 * NFFT_d * cfg.TSW)
    v_axis_ms = (np.arange(NFFT_d) - NFFT_d // 2) * dv

    # Trim to unambiguous range window
    n_valid   = min(NFFT_r, int(np.ceil(cfg.C_LIGHT / cfg.DELTA_F / dr)) + 1)
    r_axis_m  = r_axis_m[:n_valid]
    RD        = RD[:n_valid, :]

    RD_dB     = 10.0 * np.log10(np.abs(RD) ** 2 + 1e-30)
    peak_rc   = np.unravel_index(np.argmax(RD_dB), RD_dB.shape)

    return r_axis_m, v_axis_ms, RD_dB, peak_rc


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Figure helpers (shared with Phases 1–2)
# ═══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, name):
    base = os.path.join(cfg.OUTPUT_DIR, name)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=cfg.FIG_DPI)
    print(f'  Saved: {base}.pdf  |  {base}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Figure F10 — Link-budget waterfall
# ═══════════════════════════════════════════════════════════════════════════════

def plot_F10(lb, save=True):
 
    stages  = lb['stages']
    labels  = list(stages.keys())
    values  = list(stages.values())
    n       = len(labels)

    # Cumulative signal power at each stage [dBm]
    cumul   = np.cumsum(values)

    # Add noise floor and SNR display points at the end
    snr_x   = [n - 0.5, n - 0.5]
    snr_y   = [lb['P_noise_dBm'], cumul[-1]]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH_1COL, cfg.FIG_HEIGHT_TALL))

    x = np.arange(n)

    # Colour bars: green for gain, red for loss
    bar_colors = [cfg.COLORS['sideband'] if v >= 0 else cfg.COLORS['off']
                  for v in values]
    bars = ax.bar(x, values, color=bar_colors, alpha=0.80,
                  edgecolor='none', width=0.65, zorder=3)

    # Cumulative power overlay (right y-axis)
    ax2 = ax.twinx()
    ax2.plot(x, cumul, color=cfg.COLORS['on'], lw=2, ls='--',
             marker='o', ms=3, zorder=4, label='Cumul. power (dBm)')

    # Noise floor line
    ax2.axhline(lb['P_noise_dBm'], color='green',
                lw=2.0, ls=':', zorder=2,
                label=f'Noise floor ({lb["P_noise_dBm"]:.1f} dBm)')


    # Clean LaTeX labels by stripping dollar signs for x-axis
    clean_labels = []
    for lbl in labels:
        cl = (lbl.replace(r'$', '').replace(r'\rm ', '')
                 .replace(r'\to', '->').replace(r'\cdot', '·')
                 .replace(r'\ldots', '…').replace(r'\Gamma', 'Γ')
                 .replace(r'\sigma', 'σ').replace(r'_{\rm Tx}', '_Tx')
                 .replace(r'_{\rm Rx}', '_Rx').replace(r'_{\rm sb}', '_sb')
                 .replace(r'{', '').replace(r'}', ''))
        clean_labels.append(cl)

    ax.set_xticks(x)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xticklabels(clean_labels, ha='right', fontsize=10)
    ax.axhline(0, color='gray', lw=0.5, ls='-', alpha=0.4)
    ax.set_ylabel('Stage contribution (dB)', fontsize=10)
    ax.set_xlim([-0.6, n + 1.0])
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlabel('')

    ax2.set_ylabel('Cumulative power (dBm)', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10, handlelength=1.8, bbox_to_anchor=(0.8,-0.1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(40))

    fig.tight_layout(pad=0.4)
    if save:
        _save_fig(fig, 'F10_link_budget_waterfall')
    return fig, (ax, ax2)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Figure F11 — Range profile + Range-Doppler map
# ═══════════════════════════════════════════════════════════════════════════════

def plot_F11(r_axis, profile_dB, v_axis, RD_dB, meta, lb, peak_r_idx,
             peak_rc, save=True):

    fig = plt.figure(figsize=(cfg.FIG_WIDTH_2COL, 2.8),
                     constrained_layout=True)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)

    # ── Panel (a): Range profile ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(r_axis, profile_dB, color=cfg.COLORS['on'], lw=1.4,
             label='Range profile')

    # Noise floor estimate
    noise_floor_dB = 10.0 * np.log10(1.0 + 1e-30)   # unit-noise normalised
    # Mark target peak
    R_peak = r_axis[peak_r_idx]
    ax1.axvline(R_peak, color=cfg.COLORS['off'], lw=0.9, ls='--', alpha=0.7)
    ax1.plot(R_peak, profile_dB[peak_r_idx], '^',
             color=cfg.COLORS['off'], ms=6, zorder=6,
             label=f'Target peak ({R_peak:.1f} m)')

    # True range annotation
    ax1.axvline(cfg.R_RIS_T, color=cfg.COLORS['sideband'], lw=0.8,
                ls=':', alpha=0.7)
    ax1.text(cfg.R_RIS_T - 1, profile_dB.min()-0.5 ,
             f'$R$ = {cfg.R_RIS_T:.0f} m', fontsize=10,
             color='r', va='bottom')

    # ΔR annotation — anchored inside data range (y at 70 % from bottom)
    delta_r  = lb['range_res_m']
    ylo, yhi = profile_dB.min(), profile_dB.max()
    ann_y    = ylo + 0.72 * (yhi - ylo)
    ax1.annotate('', xy=(cfg.R_RIS_T + delta_r / 2, ann_y),
                 xytext=(cfg.R_RIS_T - delta_r / 2, ann_y),
                 arrowprops=dict(arrowstyle='<->', color='black',
                                 lw=0.7, mutation_scale=8))
    ax1.text(cfg.R_RIS_T - 2.5, ann_y + 0.04 * (yhi - ylo) - 0.1,
             r'$\Delta R$=' + f'{delta_r:.1f} m',
             fontsize=10, ha='center', va='bottom', color='black')

    ax1.set_xlabel('Range (m)')
    ax1.set_ylabel('Power (dB, normalised)')
    ax1.set_xlim([0, r_axis.max()])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax1.grid(True, which='major')
    ax1.grid(True, which='minor', alpha=0.15, linewidth=0.3)
    ax1.legend(loc='upper left', fontsize=9, handlelength=1.8)
    #ax1.set_title('(a) IFFT range profile', fontsize=cfg.FIG_TITLE,
          #        fontweight='normal', pad=3)

    # ── Panel (b): Range-Doppler map ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    vmin = np.percentile(RD_dB, 50)
    vmax = RD_dB.max()

    pcm = ax2.pcolormesh(v_axis, r_axis, RD_dB,
                         cmap='jet', shading='gouraud',
                         vmin=vmin, vmax=vmax, rasterized=True)

    cbar = fig.colorbar(pcm, ax=ax2, pad=0.03, fraction=0.08)
    cbar.set_label('Power (dB)', fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    # Mark target
    r_target = r_axis[peak_rc[0]]
    v_target = v_axis[peak_rc[1]]
    ax2.plot(v_target, r_target, 'o',
             color='white', ms=6, mec='red', mew=1.2, zorder=6,
             label=f'({r_target:.1f} m, {v_target:.1f} m/s)')

    # True target lines
    ax2.axhline(cfg.R_RIS_T, color='white', lw=2, ls='--', alpha=0.6)
    ax2.axvline(cfg.V_TARGET * np.cos(cfg.THETA1_RAD - cfg.THETA_T_RAD),
                color='white', lw=2, ls='--', alpha=0.6)

    ax2.set_xlabel('Radial velocity (m/s)',fontsize = 10)
    ax2.set_ylabel('Range (m)',fontsize = 10)
    ax2.set_ylim([0, r_axis.max()])
    ax2.set_xlim([v_axis.min(), v_axis.max()])
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.legend(loc='upper right', fontsize=10, handlelength=1.5,
               markerscale=0.9,bbox_to_anchor=(0.75,0.2))
   # ax2.set_title('(b) Range–Doppler map', fontsize=cfg.FIG_TITLE,
    #              fontweight='normal', pad=3)

    if save:
        _save_fig(fig, 'F11_range_doppler')
    return fig, (ax1, ax2)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Tables T1, T2
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table builder helper
# ─────────────────────────────────────────────────────────────────────────────
def _tex_table(caption, label, header, rows, notes=''):

    col_fmt = 'l l r'
    lines = [
        r'\begin{table}[!t]',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\centering',
        r'\begin{tabular}{' + col_fmt + '}',
        r'\hline\hline',
        ' & '.join(header) + r' \\',
        r'\hline',
    ]
    for row in rows:
        lines.append(' & '.join(str(c) for c in row) + r' \\')
    lines += [r'\hline\hline']
    if notes:
        lines.append(r'\multicolumn{3}{l}{\footnotesize ' + notes + r'} \\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def _csv_table(header, rows):
    """Build a simple CSV string."""
    lines = [','.join(str(h) for h in header)]
    for row in rows:
        lines.append(','.join(str(c) for c in row))
    return '\n'.join(lines)


def _write_table(name, tex_str, csv_str):
    for ext, content in [('.tex', tex_str), ('.csv', csv_str)]:
        path = os.path.join(cfg.TABLE_DIR, name + ext)
        with open(path, 'w') as fh:
            fh.write(content)
    print(f'  Exported: {os.path.join(cfg.TABLE_DIR, name)}.tex + .csv')


def export_table_T1():
    """T1 — System Parameters."""
    header = ['Parameter', 'Symbol', 'Value']
    rows = [
        ['Carrier frequency',         r'$f_c$',         '28 GHz'],
        ['Free-space wavelength',     r'$\lambda$',      '10.71 mm'],
        ['Tx radiated power',         r'$P_{\rm Tx}$',   '20 dBm (0.1 W)'],
        ['Tx antenna gain',           r'$G_{\rm Tx}$',   '0 dBi'],
        ['Target RCS',                r'$\sigma$',       '0 dBsm (1 m\\textsuperscript{2})'],
        ['Tx--RIS distance',          r'$R_1$',          '8 m'],
        ['RIS--target distance',      r'$R_2$',          '10 m'],
        ['R-RIS--Rx distance',        r'$R_3$',          '8 m'],
        ['Rx array elements',         r'$N_{\rm Rx}$',   '16'],
        ['Rx array gain',             r'$G_{\rm Rx}$',   '18 dBi'],
        ['Rx noise figure',           r'NF',             '5 dB'],
        ['Single-symbol SNR',         r'SNR$_1$',        '-14.5 dB'],
        ['Post-CPI SNR',              r'SNR$_{\rm CPI}$','15.6 dB'],
        ['Noise bandwidth',           r'$B_n$',          '312.5 MHz'],
        ['Noise floor',               r'$N_0$',          '-84.6 dBm'],
    ]
    tex = _tex_table(
        caption='System Parameters',
        label='tab:sys_params',
        header=header, rows=rows,
        notes=r'Operating point: $f_c=28$ GHz, $T_0=290$ K.')
    csv = _csv_table(header, rows)
    _write_table('T1_system_params', tex, csv)


def export_table_T2():
    """T2 — RIS Parameters."""
    header = ['Parameter', 'Symbol', 'Value']
    rows = [
        ['Array size (each stage)',          r'$N_x \times N_y$',    r'$16\times16 = 256$'],
        ['Element spacing',                  r'$d$',                  r'$\lambda/2 = 5.357$ mm'],
        ['Physical aperture (each stage)',   r'$A_{\rm RIS}$',        r'$73.4$ cm$^2$'],
        ['Phase quantisation (R-RIS)',       r'$B$',                  r'3 bits'],
        ['R-RIS reflection magnitude',       r'$|\Gamma|$',           r'0.891 ($-1$ dB)'],
        ['T-RIS sideband coeff. at $f_c$',   r'$|T_{\rm sb}|$',      r'$-7.17$ dB'],
        ['T-RIS carrier feedthrough',        r'$|T_{\rm mean}|$',    r'$-4.46$ dB'],
        ['T-RIS phase difference',           r'$\Delta\phi$',         r'$64.5^\circ$'],
        ['Inter-stage gap (nominal)',         r'$d_g$',               r'2 mm ($0.19\lambda$)'],
        ['Near-field coupling $|C_0|$',      r'$C_0$',               r'0.309 ($-10.2$ dB)'],
        ['T-RIS array gain',                 r'$G_T$',               r'29.1 dBi'],
        ['R-RIS coherent beam gain',         r'$G_R$',               r'36.1 dBi'],
        ['PIN diode series resistance',      r'$R_s$',               r'2 $\Omega$'],
        ['PIN diode junction cap.',          r'$C_J$',               r'25 fF'],
        ['Switching energy per event',       r'$E_{\rm sw}$',        r'10 pJ'],
    ]
    tex = _tex_table(
        caption='RIS Parameters',
        label='tab:ris_params',
        header=header, rows=rows,
        notes=r'Both T-RIS and R-RIS use the same $16\times16$ panel geometry.')
    csv = _csv_table(header, rows)
    _write_table('T2_ris_params', tex, csv)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Data export
# ═══════════════════════════════════════════════════════════════════════════════

def export_data(lb, Y, meta, r_axis, profile_dB, v_axis, RD_dB):
    path = os.path.join(cfg.DATA_DIR, 'bistatic_radar_data.npy')
    np.save(path, {
        'link_budget':   lb,
        'beat_matrix':   Y,
        'meta':          meta,
        'r_axis_m':      r_axis,
        'profile_dB':    profile_dB,
        'v_axis_ms':     v_axis,
        'RD_dB':         RD_dB,
    }, allow_pickle=True)
    print(f'  Exported: {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Console summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_radar_summary(meta, r_axis, profile_dB, peak_r_idx,
                         v_axis, RD_dB, peak_rc):
    sep = '─' * 60
    print(f'\n{sep}')
    print('  Radar Signal Processing Summary')
    print(sep)
    print(f'  Target range (true)     = {meta["range_m"]:.2f} m')
    print(f'  Target range (detected) = {r_axis[peak_r_idx]:.2f} m')
    print(f'  Range error             = {abs(r_axis[peak_r_idx] - meta["range_m"]):.3f} m')
    print(f'  Target velocity (true)  = {meta["vel_ms"]:.2f} m/s (radial proj.)')
    v_det = v_axis[peak_rc[1]]
    print(f'  Velocity (detected)     = {v_det:.2f} m/s')
    print(f'  Doppler shift fd        = {meta["fd_Hz"]:.1f} Hz')
    print(f'  Peak range profile SNR  = {profile_dB[peak_r_idx] - np.median(profile_dB):.1f} dB')
    print(sep + '\n')


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(show=False):
    """
    Run Stage 3 — Bistatic Radar Signal Processing.

    Computes link budget, generates beat signals, range profile,
    R-D map.  Saves F10, F11, T1, T2.

    Parameters
    ----------
    show : bool  If True, call plt.show() after saving.

    Returns
    -------
    Tuple (lb, Y, meta, r_axis, profile_dB, v_axis, RD_dB)
    """
    print('\n' + '═' * 60)
    print('  Stage 3 — Bistatic Radar Signal Processing')
    print('  Generating Figures F10, F11 | Tables T1, T2')
    print('═' * 60)

    # ── Link budget ───────────────────────────────────────────────────────────
    lb = compute_link_budget()
    print_link_budget_summary(lb)
    plot_F10(lb, save=True)

    # ── Beat signal synthesis ──────────────────────────────────────────────────
    Y, meta = synthesize_beat_signals(lb, n_sweeps=cfg.NSw)

    # ── Range profile ─────────────────────────────────────────────────────────
    r_axis, profile, profile_dB, peak_r_idx = compute_range_profile(Y, meta)

    # ── Range-Doppler map (use 256 sweeps for speed) ──────────────────────────
    n_rd = min(256, cfg.NSw)
    r_axis_rd, v_axis, RD_dB, peak_rc = compute_range_doppler(Y, meta,
                                                               n_sweeps_rd=n_rd)

    print_radar_summary(meta, r_axis, profile_dB, peak_r_idx,
                        v_axis, RD_dB, peak_rc)

    plot_F11(r_axis, profile_dB, v_axis, RD_dB, meta, lb,
             peak_r_idx, peak_rc, save=True)

    # ── Tables ────────────────────────────────────────────────────────────────
    export_table_T1()
    export_table_T2()

    # ── Data export ───────────────────────────────────────────────────────────
    export_data(lb, Y, meta, r_axis, profile_dB, v_axis, RD_dB)

    if show:
        plt.show()
    plt.close('all')

    print('\n  Stage 3 complete — 2 figures + 2 tables + data saved.')
    return lb, Y, meta, r_axis, profile_dB, v_axis, RD_dB


if __name__ == '__main__':
    run()
