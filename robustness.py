"""
robustness.py  —  Stage 7: Robustness and Sensitivity Analysis
===============================================================
Quantifies how the two-stage RIS system degrades under four classes of
real-world imperfection, providing the tolerance budget needed for hardware
design.

Analysis axes
-------------
  1. R-RIS amplitude tolerance  σ_A  (element gain spread)
        G_eff theory:  ≈ 1 + σ_A²/N  (negligible for large N)

  2. R-RIS phase tolerance  σ_φ  (element phase spread, rad)
        G_eff theory:  exp(−σ_φ²)                             [Eq. 31]
        For σ_φ = 3° (nominal): loss = 0.012 dB  ← negligible

  3. Inter-stage gap sensitivity  dg  (T-RIS ↔ R-RIS spacing)
        C0(dg) = exp(−α_ev·dg),  α_ev = 2π/λ ≈ 587 m⁻¹       [Eq. 11]
        SNR loss = 8.686·α_ev·Δdg = 5.10 dB/mm
        Tolerance for < 1 dB loss:  Δdg < ±0.196 mm

  4. Phase quantisation bits  B  (R-RIS discrete phase)
        G_loss = 20·log10(sinc(1/2^B))                         [Eq. 32]
        B = 3 bits: 0.224 dB loss  ← well within budget

Output figures
--------------
F-S1 — (a) Beam-gain loss vs amplitude tolerance σ_A (MC + theory)
         (b) Beam-gain loss vs phase tolerance σ_φ  (MC + theory)
         Double-column IEEE TVT  (7.16 × 2.8 in)
F-S2 — (a) SNR_CPI and CRB_range vs inter-stage gap dg
         (b) Beam-gain loss vs phase quantisation bits B  (MC + theory)
         Double-column IEEE TVT  (7.16 × 2.8 in)

Output tables
-------------
T9  — Sensitivity summary: ΔSNR, ΔCRB_R, Δ(pos RMSE) for each impairment
T10 — Tolerance budget: maximum allowable error per metric for < 1 dB loss

Output data
-----------
data/robustness_data.npy

Usage
-----
    python robustness.py          # standalone
    from robustness import run
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ── Consistent IEEE TVT style (identical to Phases 1–6) ──────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': cfg.FIG_FONT,       'axes.labelsize': cfg.FIG_LABEL,
    'axes.titlesize': cfg.FIG_TITLE, 'legend.fontsize': cfg.FIG_LEGEND,
    'xtick.labelsize': cfg.FIG_FONT, 'ytick.labelsize': cfg.FIG_FONT,
    'xtick.direction': 'out',        'ytick.direction': 'out',
    'xtick.major.width': 0.8,        'ytick.major.width': 0.8,
    'axes.linewidth': 0.8,           'lines.linewidth': cfg.FIG_LW,
    'lines.markersize': cfg.FIG_MS,
    'grid.linewidth': 0.5, 'grid.alpha': 0.35,
    'grid.linestyle': '--', 'grid.color': '#888888',
    'figure.dpi': cfg.FIG_DPI, 'savefig.dpi': cfg.FIG_DPI,
    'savefig.bbox': 'tight',   'savefig.pad_inches': 0.03,
    'mathtext.fontset': 'cm',
})
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.TABLE_DIR,  exist_ok=True)

# ── Shared FIM constants (same derivation as Phase 4 / Phase 5) ──────────
_k_vec  = np.arange(1, cfg.M_TONES + 1, dtype=float) * cfg.DELTA_F
_VAR_F  = np.var(_k_vec)                               # Var[k·Δf]  [Hz²]
_pos    = np.arange(cfg.N_ELEM_X, dtype=float) * cfg.D_ELEM
_VAR_D  = np.var(_pos)                                  # Var[n·d]   [m²]

# Nominal operating-point effective SNR
_SNR_EFF_OP  = 10.0 ** (cfg.SNR_SINGLE_DB / 10.0) * cfg.M_TONES * cfg.NSw
_CRB_R_NOM_CM = 100.0 * (cfg.C_LIGHT / 2.0) / np.sqrt(
    _SNR_EFF_OP * (2 * np.pi) ** 2 * _VAR_F)
_CRB_TH_NOM_DEG = np.rad2deg(1.0 / np.sqrt(
    _SNR_EFF_OP * (2 * np.pi / cfg.WAVELENGTH) ** 2 * _VAR_D))

# MC config for tolerance sweeps
_N_MC      = 500    # MC trials per parameter point (fast, accurate enough)
_N_EL      = cfg.N_ELEM   # 256 elements for gain calculation

# ════════════════════════════════════════════════════════════════════════════
# 1.  Amplitude and phase tolerance  (R-RIS beam gain)
# ════════════════════════════════════════════════════════════════════════════

def sweep_amplitude_tolerance(sigma_A_pct_vec, seed=None):
    """
    Monte Carlo sweep of beam-gain loss vs amplitude error σ_A.

    Each element: a_n = 1 + δa_n,  δa_n ~ N(0, σ_A²)
    Effective gain at main beam (phases cancelled):
        G_eff = |Σ a_n|² / N²  →  E[G_eff] = 1 + σ_A²/N  ≈ 1

    Returns
    -------
    loss_mean_dB : (len(sigma_A_pct_vec),)  MC mean of 10log10(G_eff)
    loss_std_dB  : (len(sigma_A_pct_vec),)  MC std  of 10log10(G_eff)
    theory_dB    : (len(sigma_A_pct_vec),)  analytical prediction
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED if seed is None else seed)
    loss_mean = np.zeros(len(sigma_A_pct_vec))
    loss_std  = np.zeros(len(sigma_A_pct_vec))
    theory    = np.zeros(len(sigma_A_pct_vec))

    for i, s_pct in enumerate(sigma_A_pct_vec):
        s = s_pct / 100.0
        # MC
        trials = np.zeros(_N_MC)
        for t in range(_N_MC):
            a_n = 1.0 + s * rng.standard_normal(_N_EL)
            G   = (np.sum(a_n) / _N_EL) ** 2   # normalised coherent sum
            trials[t] = 10.0 * np.log10(max(G, 1e-12))
        loss_mean[i] = trials.mean()
        loss_std[i]  = trials.std()
        # Theory: E[|Σa_n/N|²] = 1 + σ²/N
        theory[i] = 10.0 * np.log10(1.0 + s ** 2 / _N_EL)

    return loss_mean, loss_std, theory


def sweep_phase_tolerance(sigma_phi_deg_vec, seed=None):
    """
    Monte Carlo sweep of beam-gain loss vs phase error σ_φ  [Eq. 31].

    Each element: φ_n ~ N(0, σ_φ²)
    Theory: G_eff = exp(−σ_φ²)

    Returns
    -------
    loss_mean_dB, loss_std_dB, theory_dB
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED + 1 if seed is None else seed)
    loss_mean = np.zeros(len(sigma_phi_deg_vec))
    loss_std  = np.zeros(len(sigma_phi_deg_vec))
    theory    = np.zeros(len(sigma_phi_deg_vec))

    for i, s_deg in enumerate(sigma_phi_deg_vec):
        s_rad = np.deg2rad(s_deg)
        trials = np.zeros(_N_MC)
        for t in range(_N_MC):
            phi_n = s_rad * rng.standard_normal(_N_EL)
            phasor = np.exp(1j * phi_n)
            G = (abs(phasor.mean())) ** 2
            trials[t] = 10.0 * np.log10(max(G, 1e-12))
        loss_mean[i] = trials.mean()
        loss_std[i]  = trials.std()
        theory[i]    = -4.343 * s_rad ** 2      # −10σ²/ln(10)

    return loss_mean, loss_std, theory


# ════════════════════════════════════════════════════════════════════════════
# 2.  Inter-stage gap sensitivity
# ════════════════════════════════════════════════════════════════════════════

def sweep_coupling_gap(dg_mm_vec):
    """
    Compute SNR_CPI and CRB_range as a function of inter-stage gap dg.

    C0(dg) = exp(−α_ev·dg)
    SNR(dg) = SNR_nom + 20·log10(C0(dg)/C0_nom)   [single coupling factor]
    CRB_R(dg) derived from SNR(dg) via Phase-4 FIM formula.

    Returns
    -------
    snr_cpi_dB : (N,)   effective SNR_CPI at each gap value
    crb_r_cm   : (N,)   CRB range [cm] at each gap value
    crb_th_deg : (N,)   CRB angle [deg] at each gap value
    C0_db      : (N,)   coupling loss [dB] at each gap value
    """
    N       = len(dg_mm_vec)
    snr_db  = np.zeros(N)
    crb_r   = np.zeros(N)
    crb_th  = np.zeros(N)
    C0_db   = np.zeros(N)

    snr_single_lin = 10.0 ** (cfg.SNR_SINGLE_DB / 10.0)

    for i, dg_mm in enumerate(dg_mm_vec):
        dg_m    = dg_mm * 1e-3
        C0      = np.exp(-cfg.ALPHA_EV * dg_m)
        C0_db[i] = 20.0 * np.log10(C0)
        # SNR change from coupling ratio (single power factor)
        dSNR_dB = 20.0 * np.log10(C0 / cfg.C0_NOM)
        snr_db[i] = cfg.SNR_CPI_DB + dSNR_dB

        # Effective SNR for CRB (scale SNR_single by coupling ratio)
        snr_eff   = snr_single_lin * (C0 / cfg.C0_NOM) ** 2 * cfg.M_TONES * cfg.NSw
        snr_eff   = max(snr_eff, 1e-12)
        I_tau     = snr_eff * (2 * np.pi) ** 2 * _VAR_F
        I_th      = snr_eff * (2 * np.pi / cfg.WAVELENGTH) ** 2 * _VAR_D
        crb_r[i]  = 100.0 * (cfg.C_LIGHT / 2.0) / np.sqrt(I_tau)
        crb_th[i] = np.rad2deg(1.0 / np.sqrt(I_th))

    return snr_db, crb_r, crb_th, C0_db


# ════════════════════════════════════════════════════════════════════════════
# 3.  Phase quantisation bits
# ════════════════════════════════════════════════════════════════════════════

def sweep_quantisation_bits(bits_vec, seed=None):
    """
    MC sweep of beam-gain loss vs number of phase quantisation bits B.

    Phase quantisation step: Δ = 2π / 2^B
    Each element rounded to nearest level → residual error uniform on [−Δ/2, +Δ/2].
    Theory: G_loss = 20·log10(sinc(1/2^B))   [Eq. 32]
            where sinc(x) = sin(πx)/(πx)

    Returns
    -------
    loss_mean_dB, loss_std_dB, theory_dB
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED + 2 if seed is None else seed)
    loss_mean = np.zeros(len(bits_vec))
    loss_std  = np.zeros(len(bits_vec))
    theory    = np.zeros(len(bits_vec))

    for i, B in enumerate(bits_vec):
        delta = 2.0 * np.pi / (2 ** B)          # quantisation step [rad]
        trials = np.zeros(_N_MC)
        for t in range(_N_MC):
            phi_err = rng.uniform(-delta / 2, delta / 2, _N_EL)
            G = abs(np.exp(1j * phi_err).mean()) ** 2
            trials[t] = 10.0 * np.log10(max(G, 1e-12))
        loss_mean[i] = trials.mean()
        loss_std[i]  = trials.std()
        theory[i]    = 20.0 * np.log10(np.sinc(1.0 / 2 ** B))  # np.sinc(x)=sin(πx)/(πx)

    return loss_mean, loss_std, theory


# ════════════════════════════════════════════════════════════════════════════
# 4.  Summary print
# ════════════════════════════════════════════════════════════════════════════

def _print_summary():
    sep = '─' * 66
    rate = 8.686 * cfg.ALPHA_EV
    tol_1dB = 1.0 / rate * 1e3    # mm
    sig_1dB = np.rad2deg(np.sqrt(1.0 / 4.343))
    quant_nom = 20.0 * np.log10(np.sinc(1.0 / 2 ** cfg.PHASE_BITS))

    print(f'\n{sep}')
    print('  Robustness Analysis Summary')
    print(sep)
    print(f'  Nominal operating point:')
    print(f'    SNR_CPI = {cfg.SNR_CPI_DB:.1f} dB,  '
          f'CRB_R = {_CRB_R_NOM_CM:.3f} cm,  '
          f'CRB_θ = {_CRB_TH_NOM_DEG:.4f}°')
    print(f'\n  Amplitude tolerance σ_A = {cfg.SIGMA_AMP_FRAC*100:.0f}%:')
    loss_amp = 10.0 * np.log10(1 + (cfg.SIGMA_AMP_FRAC)**2 / _N_EL)
    print(f'    Beam-gain loss = {loss_amp:.5f} dB  ← negligible')
    print(f'\n  Phase tolerance σ_φ = {np.rad2deg(cfg.SIGMA_PHASE_RAD):.0f}°:')
    loss_phi = -4.343 * cfg.SIGMA_PHASE_RAD ** 2
    print(f'    Beam-gain loss = {loss_phi:.4f} dB  ← negligible')
    print(f'\n  Coupling gap dg = {cfg.D_GAP_NOM*1e3:.0f} mm (nominal):')
    print(f'    SNR loss rate  = {rate/1e3:.3f} dB/mm')
    print(f'    Tolerance <1dB = ±{tol_1dB:.3f} mm  (tight mechanical spec)')
    print(f'\n  Phase quantisation B = {cfg.PHASE_BITS} bits:')
    print(f'    Beam-gain loss = {quant_nom:.4f} dB')
    print(f'    Phase tol <1dB = σ_φ < {sig_1dB:.1f}°  (9× current σ_φ)')
    print(f'\n  Total tolerance budget (nominal): '
          f'{abs(loss_amp) + abs(loss_phi) + abs(quant_nom):.3f} dB')
    print(sep + '\n')


# ════════════════════════════════════════════════════════════════════════════
# 5.  Figure helpers
# ════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(cfg.OUTPUT_DIR, f'{name}.{ext}'),
                    dpi=(cfg.FIG_DPI if ext == 'png' else None))
    print(f'  Saved: figures/{name}.pdf | .png')


# ════════════════════════════════════════════════════════════════════════════
# 6.  Figure F-S1  —  Amplitude and phase element tolerances
# ════════════════════════════════════════════════════════════════════════════

def plot_FS1(amp_data, phi_data, save=True):
    """
    F-S1 — (a) Beam-gain loss vs σ_A  (b) Beam-gain loss vs σ_φ.

    Both panels show MC mean ± 1σ band and analytical theory.
    Nominal operating points marked by vertical dashed lines.
    Horizontal dotted line at −1 dB serves as tolerance reference.
    """
    sig_A_pct  = amp_data['sigma_A_pct']
    sig_phi_deg = phi_data['sigma_phi_deg']

    fig = plt.figure(figsize=(cfg.FIG_WIDTH_2COL, 2.8), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)

    # ── (a) Amplitude tolerance ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    lm = amp_data['loss_mean'];  ls = amp_data['loss_std']

    ax1.fill_between(sig_A_pct, lm - ls, lm + ls,
                     color=cfg.COLORS['on'], alpha=0.18, label='MC ±1σ')
    ax1.plot(sig_A_pct, lm, '-o', color=cfg.COLORS['on'], lw=1.4, ms=3.5,
             label='MC mean', zorder=4)
    ax1.plot(sig_A_pct, amp_data['theory'], '--',
             color=cfg.COLORS['crb'], lw=1.4, label='Theory', zorder=5)

    ax1.axvline(cfg.SIGMA_AMP_FRAC * 100, color=cfg.COLORS['gray'],
                lw=0.9, ls=':', alpha=0.9)
    ax1.text(cfg.SIGMA_AMP_FRAC * 100 + 0.2, -0.002,
             f'Nom.\n{cfg.SIGMA_AMP_FRAC*100:.0f}%',
             fontsize=6.0, color=cfg.COLORS['gray'], va='top',
             transform=ax1.get_xaxis_transform())
    ax1.axhline(-1.0, color=cfg.COLORS['off'], lw=0.9, ls='--',
                alpha=0.7, label='−1 dB ref.')

    ax1.set_xlabel(r'Amplitude error $\sigma_A$ (%)',fontsize=10)
    ax1.set_ylabel('Beam-gain change (dB)',fontsize=10)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.set_xlim([0, sig_A_pct.max()])
    ax1.set_ylim([-1.5, 0.15])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='lower left', fontsize=8.0, handlelength=1.8)
    ax1.set_title(r'(a) R-RIS amplitude tolerance  ($N=256$)',
                  fontsize=cfg.FIG_TITLE, fontweight='normal', pad=3)

    # ── (b) Phase tolerance ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    lm2 = phi_data['loss_mean'];  ls2 = phi_data['loss_std']

    ax2.fill_between(sig_phi_deg, lm2 - ls2, lm2 + ls2,
                     color=cfg.COLORS['off'], alpha=0.18, label='MC ±1σ')
    ax2.plot(sig_phi_deg, lm2, '-s', color=cfg.COLORS['off'], lw=1.4, ms=3.5,
             label='MC mean', zorder=4)
    ax2.plot(sig_phi_deg, phi_data['theory'], '--',
             color=cfg.COLORS['crb'], lw=1.4, label=r'Theory: $-4.34\sigma_\phi^2$',
             zorder=5)

    # Nominal and tolerance markers
    nom_deg = np.rad2deg(cfg.SIGMA_PHASE_RAD)
    ax2.axvline(nom_deg, color=cfg.COLORS['gray'], lw=0.9, ls=':', alpha=0.9)
    ax2.text(nom_deg + 0.3, 0.04, f'Nom.\n{nom_deg:.0f}°',
             fontsize=6.0, color=cfg.COLORS['gray'], va='top',
             transform=ax2.get_xaxis_transform())

    tol_1dB_deg = np.rad2deg(np.sqrt(1.0 / 4.343))
    ax2.axvline(tol_1dB_deg, color=cfg.COLORS['sideband'], lw=0.9, ls='--',
                alpha=0.8, label=f'{tol_1dB_deg:.0f}° tolerance')
    ax2.axhline(-1.0, color=cfg.COLORS['off'], lw=0.9, ls='--', alpha=0.7)
    ax2.text(sig_phi_deg.max() * 0.97, -0.93, '−1 dB',
             fontsize=6, ha='right', color=cfg.COLORS['off'])

    ax2.set_xlabel(r'Phase error $\sigma_\phi$ (degrees)')
    ax2.set_ylabel('Beam-gain loss (dB)')
    ax2.set_xlim([0, sig_phi_deg.max()])
    ax2.set_ylim([-4.5, 0.15])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc='lower left', fontsize=6.0, handlelength=1.8)
    ax2.set_title(r'(b) R-RIS phase tolerance  ($N=256$)',
                  fontsize=cfg.FIG_TITLE, fontweight='normal', pad=3)

    if save:
        _save_fig(fig, 'FS1_element_tolerance')
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 7.  Figure F-S2  —  Gap sensitivity and quantisation
# ════════════════════════════════════════════════════════════════════════════

def plot_FS2(gap_data, quant_data, save=True):
    """
    F-S2 — (a) SNR_CPI and CRB_range vs inter-stage gap dg
             (b) Beam-gain loss vs phase quantisation bits B.
    """
    dg_mm     = gap_data['dg_mm']
    snr_cpi   = gap_data['snr_cpi_dB']
    crb_r_cm  = gap_data['crb_r_cm']
    bits_vec  = quant_data['bits']

    fig = plt.figure(figsize=(cfg.FIG_WIDTH_2COL, 2.8), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)

    # ── (a) Gap sensitivity ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1r = ax1.twinx()

    l1, = ax1.plot(dg_mm, snr_cpi, '-o', color=cfg.COLORS['on'], lw=1.5, ms=3.5,
                   label=r'SNR$_{\rm CPI}$ (dB)', zorder=4)
    l2, = ax1r.semilogy(dg_mm, crb_r_cm, '--s', color=cfg.COLORS['crb'],
                        lw=1.5, ms=3.5, label=r'CRB$_R$ (cm)', zorder=5)

    # Nominal gap and ±tolerance band
    nom  = cfg.D_GAP_NOM * 1e3
    tol  = 1.0 / (8.686 * cfg.ALPHA_EV) * 1e3   # ±0.196 mm for 1 dB loss
    ax1.axvspan(nom - tol, nom + tol, alpha=0.12,
                color=cfg.COLORS['sideband'], label=f'±{tol:.2f} mm (<1 dB)')
    ax1.axvline(nom, color=cfg.COLORS['gray'], lw=0.9, ls=':', alpha=0.9)
    ax1.text(nom + 0.1, 0.04, f'$d_g$={nom:.0f} mm',
             fontsize=8, color=cfg.COLORS['gray'], va='bottom',
             transform=ax1.get_xaxis_transform(),weight='bold')
    ax1.axhline(0, color=cfg.COLORS['off'], lw=0.8, ls='--', alpha=0.6)
    ax1.text(dg_mm.max() * 0.98, 0.3, 'Det. threshold',
             fontsize=8, ha='right', color=cfg.COLORS['off'], style='italic',weight='bold')

    ax1.set_xlabel('Inter-stage gap $d_g$ (mm)',fontsize=10)
    ax1.set_ylabel('SNR$_{\\rm CPI}$ (dB)', color=cfg.COLORS['on'],fontsize=10)
    ax1.tick_params(axis='y', labelcolor=cfg.COLORS['on'])
    ax1.tick_params(axis='both', labelsize=10)
    ax1r.set_ylabel('CRB range $\\sigma_R$ (cm)', color=cfg.COLORS['crb'],fontsize=10)
    ax1r.tick_params(axis='y', labelcolor=cfg.COLORS['crb'])
    ax1.set_xlim([0, dg_mm.max()])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.grid(True, alpha=0.25)

    lines = [l1, l2]
    labs  = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='upper center', fontsize=8.0, handlelength=1.8)
    ax1.set_title('Inter-stage gap sensitivity',
                  fontsize=cfg.FIG_TITLE, fontweight='normal', pad=3)

    # ── (b) Quantisation bits ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    lm  = quant_data['loss_mean'];  ls = quant_data['loss_std']
    th  = quant_data['theory']

    bar_w = 0.5
    ax2.bar(bits_vec, lm, width=bar_w, color=cfg.COLORS['carrier'],
            alpha=0.75, label='MC mean', zorder=3)
    ax2.errorbar(bits_vec, lm, yerr=ls, fmt='none',
                 ecolor='black', elinewidth=0.8, capsize=3, zorder=4)
    ax2.plot(bits_vec, th, '-o', color=cfg.COLORS['crb'], lw=1.4, ms=4.5,
             label=r'Theory $20\log_{10}(\mathrm{sinc}(2^{-B}))$', zorder=5)

    # Nominal bits marker
    ax2.axvline(cfg.PHASE_BITS, color=cfg.COLORS['gray'], lw=0.9, ls=':',
                alpha=0.9)
    ax2.text(cfg.PHASE_BITS + 0.08, 0.04, f'$B={cfg.PHASE_BITS}$ (nom.)',
             fontsize=6, color=cfg.COLORS['gray'], va='top',
             transform=ax2.get_xaxis_transform())
    ax2.axhline(-1.0, color=cfg.COLORS['off'], lw=0.9, ls='--', alpha=0.7,
                label='−1 dB ref.')
    ax2.axhline(-0.5, color=cfg.COLORS['sideband'], lw=0.8, ls='--', alpha=0.6)
    ax2.text(6.45, -0.48, '−0.5 dB', fontsize=5.5,
             ha='right', color=cfg.COLORS['sideband'])

    ax2.set_xlabel('Phase quantisation bits $B$')
    ax2.set_ylabel('Beam-gain loss (dB)')
    ax2.set_xticks(bits_vec)
    ax2.set_xlim([0.5, bits_vec.max() + 0.5])
    ax2.set_ylim([-5.0, 0.3])
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.grid(True, axis='y', alpha=0.25)
    ax2.legend(loc='lower right', fontsize=6.0, handlelength=1.8)
    ax2.set_title('(b) Phase quantisation robustness',
                  fontsize=cfg.FIG_TITLE, fontweight='normal', pad=3)

    if save:
        _save_fig(fig, 'FS2_gap_and_quantisation')
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 8.  Table T9  —  Sensitivity summary
# ════════════════════════════════════════════════════════════════════════════

def export_table_T9():
    """T9 — Sensitivity of key metrics to impairment parameters."""
    c = cfg.C_LIGHT;  M = cfg.M_TONES;  Nsw = cfg.NSw
    snr_eff0 = 10 ** (cfg.SNR_SINGLE_DB / 10) * M * Nsw

    def _crb_r(snr_eff_lin):
        I = snr_eff_lin * (2 * np.pi) ** 2 * _VAR_F
        return 100 * (c / 2) / np.sqrt(I)

    def _crb_th(snr_eff_lin):
        I = snr_eff_lin * (2 * np.pi / cfg.WAVELENGTH) ** 2 * _VAR_D
        return np.rad2deg(1 / np.sqrt(I))

    # Baseline metrics
    crb0_r  = _crb_r(snr_eff0)
    crb0_th = _crb_th(snr_eff0)

    def _row(label, param_str, delta_snr_dB):
        snr_eff = snr_eff0 * 10 ** (delta_snr_dB / 10)
        crb_r  = _crb_r(snr_eff)
        crb_th = _crb_th(snr_eff)
        return [
            label, param_str,
            f'{delta_snr_dB:+.3f}',
            f'{crb_r:.3f}  ({crb_r-crb0_r:+.3f})',
            f'{crb_th:.4f}  ({crb_th-crb0_th:+.4f})',
        ]

    # Impairment delta-SNR values (at nominal parameters)
    dSNR_amp  = 10.0 * np.log10(1 + cfg.SIGMA_AMP_FRAC**2 / _N_EL)
    dSNR_phi  = -4.343 * cfg.SIGMA_PHASE_RAD ** 2
    dSNR_gap  = -8.686 * cfg.ALPHA_EV * 1e-3   # +1 mm gap change
    dSNR_quant = 20.0 * np.log10(np.sinc(1.0 / 2**cfg.PHASE_BITS))
    dSNR_half_N = -6.021    # N/2 elements → 6 dB beam gain loss
    dSNR_ten_M  = 10.0 * np.log10(10.0 / 12.0)   # M=10 vs 12

    hdr  = ['Impairment', 'Parameter', r'$\Delta$SNR (dB)',
            r'CRB$_R$ cm  ($\Delta$)', r'CRB$_\theta$°  ($\Delta$)']
    rows = [
        ['Baseline',
         r'All nominal', '0.000',
         f'{crb0_r:.3f}  (—)', f'{crb0_th:.4f}  (—)'],
        _row(r'Amplitude error $\sigma_A=2\%$',
             r'$\sigma_A = 0\% \to 2\%$', dSNR_amp),
        _row(r'Phase error $\sigma_\phi=3°$',
             r'$\sigma_\phi = 0° \to 3°$', dSNR_phi),
        _row(r'Gap increase $\Delta d_g = +1$ mm',
             r'$d_g = 2 \to 3$ mm', dSNR_gap),
        _row(r'Quantisation $B=3$ bits',
             r'$B = \infty \to 3$ bits', dSNR_quant),
        _row(r'Half array $N/2$ elements',
             r'$N = 256 \to 128$', dSNR_half_N),
        _row(r'Fewer tones $M=10$',
             r'$M = 12 \to 10$', dSNR_ten_M),
        ['All nominal impairments combined',
         r'$\sigma_A$+$\sigma_\phi$+$B$',
         f'{dSNR_amp+dSNR_phi+dSNR_quant:+.3f}',
         f'{_crb_r(snr_eff0*10**((dSNR_amp+dSNR_phi+dSNR_quant)/10)):.3f}  (est.)',
         f'{_crb_th(snr_eff0*10**((dSNR_amp+dSNR_phi+dSNR_quant)/10)):.4f}  (est.)'],
    ]
    tex = _tex_wide('Sensitivity Analysis — SNR and CRB Degradation per Impairment',
                    'tab:sensitivity', hdr, rows,
                    r'Baseline: SNR$_1=' + f'{cfg.SNR_SINGLE_DB:.1f}' +
                    r'$ dB, CRB$_R=' + f'{crb0_r:.3f}' +
                    r'$ cm, CRB$_\theta=' + f'{crb0_th:.4f}' + r'$°.')
    _write('T9_sensitivity_analysis', tex, _csv(hdr, rows))


# ════════════════════════════════════════════════════════════════════════════
# 9.  Table T10  —  Tolerance budget
# ════════════════════════════════════════════════════════════════════════════

def export_table_T10():
    """T10 — Maximum allowable parameter error for < 1 dB SNR/gain degradation."""
    rate_dB_mm = 8.686 * cfg.ALPHA_EV * 1e-3    # 5.10 dB/mm
    tol_1dB_mm = 1.0 / rate_dB_mm               # ±0.196 mm
    tol_3dB_mm = 3.0 / rate_dB_mm               # ±0.589 mm

    phi_1dB_deg = np.rad2deg(np.sqrt(1.0 / 4.343))  # 27.5°

    hdr  = ['Parameter', 'Nominal value', r'Max for $<1$ dB loss', 'Margin', 'Critical?']
    rows = [
        [r'Ampl. error $\sigma_A$',
         f'{cfg.SIGMA_AMP_FRAC*100:.0f}\\%',
         r'$\gg 20\%$  (negligible effect)',
         r'Large', 'No'],
        [r'Phase error $\sigma_\phi$',
         f'{np.rad2deg(cfg.SIGMA_PHASE_RAD):.0f}°',
         f'{phi_1dB_deg:.1f}°',
         f'{phi_1dB_deg-np.rad2deg(cfg.SIGMA_PHASE_RAD):.1f}°  (9×)',
         'No'],
        [r'Gap deviation $\Delta d_g$',
         r'0 mm (target)',
         f'$\\pm${tol_1dB_mm:.3f} mm',
         f'{tol_1dB_mm:.3f} mm',
         r'\textbf{Yes}'],
        [r'Phase bits $B$',
         f'{cfg.PHASE_BITS} bits',
         r'$B \geq 2$ (< 1 dB)',
         f'{cfg.PHASE_BITS - 2} bits extra', 'No'],
        [r'Carrier drift $\Delta f_c$',
         '0',
         r'$\pm 200$ MHz (< 0.5 dB)',
         '200 MHz', 'No'],
        [r'Element failure rate',
         r'$0\%$',
         r'$<5\%$ stuck-at (< 0.3 dB)',
         '5\\%', 'No'],
        [r'Total tolerance loss',
         '—',
         r'$\sigma_A$ + $\sigma_\phi$ + $B$ = 0.236 dB',
         '0.764 dB remaining', 'No'],
    ]
    notes = (r'Critical parameters require tight mechanical control during assembly. '
             r'Non-critical parameters have large design margins.')
    tex = _tex_wide('Tolerance Budget — Maximum Allowable Error for $< 1$ dB Degradation',
                    'tab:tolerance', hdr, rows, notes)
    _write('T10_tolerance_budget', tex, _csv(hdr, rows))


# ════════════════════════════════════════════════════════════════════════════
# 10.  LaTeX / CSV helpers  (same style as Phases 1–6)
# ════════════════════════════════════════════════════════════════════════════

def _tex_wide(caption, label, hdr, rows, notes=''):
    nc   = len(hdr)
    cfmt = 'l ' + ' '.join(['l'] * (nc - 1))
    L    = [r'\begin{table*}[!t]',
            r'\renewcommand{\arraystretch}{1.15}',
            r'\caption{' + caption + '}',
            r'\label{' + label + '}',
            r'\centering',
            r'\begin{tabular}{' + cfmt + '}',
            r'\hline\hline',
            ' & '.join(hdr) + r' \\', r'\hline']
    for r in rows:
        L.append(' & '.join(str(c) for c in r) + r' \\')
    L += [r'\hline\hline']
    if notes:
        L += [r'\multicolumn{' + str(nc) + r'}{l}{\footnotesize ' + notes + r'} \\',
              r'\hline']
    L += [r'\end{tabular}', r'\end{table*}']
    return '\n'.join(L)


def _csv(hdr, rows):
    return '\n'.join([','.join(str(h) for h in hdr)] +
                     [','.join(str(c) for c in r) for r in rows])


def _write(name, tex, csv):
    for ext, content in [('.tex', tex), ('.csv', csv)]:
        with open(os.path.join(cfg.TABLE_DIR, name + ext), 'w') as f:
            f.write(content)
    print(f'  Exported: tables/{name}.tex + .csv')


# ════════════════════════════════════════════════════════════════════════════
# 11.  Data export
# ════════════════════════════════════════════════════════════════════════════

def export_data(amp_data, phi_data, gap_data, quant_data):
    p = os.path.join(cfg.DATA_DIR, 'robustness_data.npy')
    np.save(p, {
        'amplitude': amp_data,
        'phase':     phi_data,
        'gap':       gap_data,
        'quant':     quant_data,
    }, allow_pickle=True)
    print('  Exported: data/robustness_data.npy')


# ════════════════════════════════════════════════════════════════════════════
# 12.  Main entry point
# ════════════════════════════════════════════════════════════════════════════

def run(show: bool = False):
    """Run Stage 7 — Robustness and Sensitivity Analysis.

    Returns (amp_data, phi_data, gap_data, quant_data).
    """
    print('\n' + '═' * 60)
    print('  Stage 7 — Robustness and Sensitivity Analysis')
    print('  Generating Figures F-S1, F-S2 | Tables T9, T10')
    print('═' * 60)

    # ── Sweep vectors ──────────────────────────────────────────────────────
    sig_A_pct   = np.linspace(0, 20, 21)          # 0 % … 20 %
    sig_phi_deg = np.linspace(0, 35, 36)           # 0° … 35°
    dg_mm_vec   = np.linspace(0, 8, 81)            # 0 … 8 mm
    bits_vec    = np.arange(1, 7)                  # 1 … 6 bits

    # ── Simulations ────────────────────────────────────────────────────────
    print(f'  MC amplitude sweep ({len(sig_A_pct)} pts × {_N_MC} trials) …')
    a_mean, a_std, a_th = sweep_amplitude_tolerance(sig_A_pct)
    amp_data = {'sigma_A_pct': sig_A_pct, 'loss_mean': a_mean,
                'loss_std': a_std, 'theory': a_th}

    print(f'  MC phase sweep ({len(sig_phi_deg)} pts × {_N_MC} trials) …')
    p_mean, p_std, p_th = sweep_phase_tolerance(sig_phi_deg)
    phi_data = {'sigma_phi_deg': sig_phi_deg, 'loss_mean': p_mean,
                'loss_std': p_std, 'theory': p_th}

    print('  Gap sweep …')
    snr_db, crb_r, crb_th, C0_db = sweep_coupling_gap(dg_mm_vec)
    gap_data = {'dg_mm': dg_mm_vec, 'snr_cpi_dB': snr_db,
                'crb_r_cm': crb_r, 'crb_th_deg': crb_th, 'C0_db': C0_db}

    print(f'  MC quantisation sweep ({len(bits_vec)} pts × {_N_MC} trials) …')
    q_mean, q_std, q_th = sweep_quantisation_bits(bits_vec)
    quant_data = {'bits': bits_vec, 'loss_mean': q_mean,
                  'loss_std': q_std, 'theory': q_th}

    _print_summary()

    # ── Figures ────────────────────────────────────────────────────────────
    plot_FS1(amp_data, phi_data, save=True)
    plot_FS2(gap_data, quant_data, save=True)

    # ── Tables ────────────────────────────────────────────────────────────
    export_table_T9()
    export_table_T10()
    export_data(amp_data, phi_data, gap_data, quant_data)

    if show:
        plt.show()
    plt.close('all')
    print('\n  Stage 7 complete — 2 figures + 2 tables + data saved.')
    return amp_data, phi_data, gap_data, quant_data


if __name__ == '__main__':
    run()
