"""
multi_target.py  —  Stage 6a: Multi-Target Sensing + FMCW Ghost-Target Comparison
==================================================================================
Demonstrates the range–Doppler decoupling advantage of M-FSK over single-chirp
FMCW for multi-target bistatic sensing.

Key physics
-----------
In FMCW, the dechirped beat frequency of a moving target contains both a range
term and a Doppler term (range–Doppler coupling):
    f_beat = (2B·R)/(c·Tsw)  +  (2v·fc)/c                      [Eq. A1]
Without 2-D chirp processing, the apparent range is displaced by:
    ΔR_FMCW = v · fc · Tsw / B                                   [Eq. A2]

In M-FSK, tone k is CW for the symbol duration Tsym.  The tone-to-tone phase
encodes only the range delay τ = R/c; the Doppler shift appears orthogonally
in the sweep-to-sweep domain.  Range and Doppler are completely decoupled:
    s[k, n] = A · exp(−j2πkΔfτ) · exp(+j2πfd·n·Tsw)            [Eq. 16]

Three-target scenario
---------------------
    Target 1:  R =  4 m,  v =  0 m/s  (static clutter)
    Target 2:  R =  7 m,  v = +3 m/s  (vehicle, receding)
    Target 3:  R = 10 m,  v = −2 m/s  (vehicle, approaching)

FMCW comparison  (same carrier, same bandwidth = M·Δf, same CPI duration)
    fc    = 28 GHz,  B_FMCW = M·Δf = 300 MHz,  Tsw_FMCW = T_CPI = 17.20 ms
    ΔR/v  = fc·Tsw/B = 28e9 × 17.20e-3 / 300e6 = 1.606 m·s/m

    T2 apparent range:  7 + 3 × 1.606 = 11.82 m  (displaced toward boundary)
    T3 apparent range: 10 − 2 × 1.606 =  6.79 m  (displaced into T2's slot!)

Output figure
-------------
F16 — (a) M-FSK IFFT: clean peaks at exactly R = 4, 7, 10 m
       (b) FMCW FFT:  Doppler-displaced peaks at 4.0, 6.79, 11.82 m
       Double-column IEEE TVT  (7.16 × 2.8 in)

Output data
-----------
data/multi_target_data.npy

Usage
-----
    python multi_target.py
    from multi_target import run
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

# ── Consistent IEEE TVT style (identical to Phases 1–5) ──────────────────
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
    'grid.linewidth': 0.5,  'grid.alpha': 0.35,
    'grid.linestyle': '--', 'grid.color': '#888888',
    'figure.dpi': cfg.FIG_DPI, 'savefig.dpi': cfg.FIG_DPI,
    'savefig.bbox': 'tight',   'savefig.pad_inches': 0.03,
    'mathtext.fontset': 'cm',
})
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ── FMCW parameters: same carrier, same bandwidth, same CPI duration ─────
# Using equal-bandwidth (B = M·Δf = 300 MHz) and equal-CPI (Tsw = T_CPI)
# gives a fair Doppler-resolution comparison and maximises displacement.
_FC_FMCW  = cfg.FC                           # 28 GHz  (shared carrier)
_B_FMCW   = cfg.M_TONES * cfg.DELTA_F        # 300 MHz = M·Δf
_TSW_FMCW = cfg.TCPI                         # 17.20 ms = T_CPI
_DR_FMCW  = cfg.C_LIGHT / (2.0 * _B_FMCW)   # 0.500 m  range resolution
# Displacement coefficient  ΔR = v · fc · Tsw / B  [m per m/s]   [Eq. A2]
DISP_COEFF = _FC_FMCW * _TSW_FMCW / _B_FMCW  # ≈ 1.606 m/(m/s)

# Target colours: consistent across both panels
_TGT_COLORS = [cfg.COLORS['on'], cfg.COLORS['off'], cfg.COLORS['sideband']]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Scenario definition
# ═══════════════════════════════════════════════════════════════════════════

def define_targets() -> list:
    """
    Define the three-target scenario.

    Each target dict contains:
        'R'   true range [m],  'v'  radial velocity [m/s],
        'rcs' radar cross-section [m²],  'label'  plot label,
        'R_fmcw'  FMCW apparent range [m]  = R + v·DISP_COEFF.
    """
    raw = [
        {'R': 4.0,  'v':  0.0, 'rcs': 1.0, 'label': 'T1'},
        {'R': 7.0,  'v': +3.0, 'rcs': 0.8, 'label': 'T2'},
        {'R': 10.0, 'v': -2.0, 'rcs': 0.5, 'label': 'T3 '},
    ]
    for tgt in raw:
        tgt['R_fmcw'] = tgt['R'] + tgt['v'] * DISP_COEFF
    return raw


def _print_scenario(targets):
    sep = '─' * 72
    print(f'\n{sep}')
    print('  Multi-Target Scenario  (FMCW apparent ranges)')
    print(sep)
    print(f'  {"Target":<36}  {"R (m)":>6}  {"v (m/s)":>8}  '
          f'{"R_FMCW (m)":>11}  {"ΔR (m)":>7}')
    print('  ' + '-' * 68)
    for tgt in targets:
        dR = tgt['R_fmcw'] - tgt['R']
        print(f'  {tgt["label"]:<36}  {tgt["R"]:6.1f}  {tgt["v"]:+8.1f}  '
              f'{tgt["R_fmcw"]:+11.2f}  {dR:+7.2f}')
    print(f'\n  FMCW: B = {_B_FMCW/1e6:.0f} MHz,  Tsw = {_TSW_FMCW*1e3:.2f} ms,  '
          f'ΔR/v = {DISP_COEFF:.3f} m/(m/s)')
    print(f'  M-FSK: M = {cfg.M_TONES} tones,  Δf = {cfg.DELTA_F/1e6:.0f} MHz,  '
          f'ΔR = {cfg.C_LIGHT/(cfg.M_TONES*cfg.DELTA_F):.1f} m  (no Doppler coupling)\n')
    print(sep + '\n')


# ═══════════════════════════════════════════════════════════════════════════
# 2.  M-FSK multi-target beat signal and IFFT range profile
# ═══════════════════════════════════════════════════════════════════════════

def synthesize_mfsk(targets: list, n_sweeps: int = None,
                    seed: int = None) -> tuple:
    """
    Synthesise M-FSK beat matrix for K targets and compute IFFT range profile.

    Y[k, n] = Σ_q √(SNR_q) · exp(−j2π·k·Δf·τ_q) · exp(+j2π·fd_q·n·Tsw) + w

    Returns
    -------
    r_axis : ndarray  Range axis [m]  (unambiguous window)
    psd_dB : ndarray  Power profile [dB, normalised to noise median]
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED if seed is None else seed)
    if n_sweeps is None:
        n_sweeps = cfg.NSw

    M = cfg.M_TONES;  Df = cfg.DELTA_F;  Tsw = cfg.TSW
    k = np.arange(1, M + 1, dtype=float)
    n = np.arange(n_sweeps, dtype=float)

    noise = ((rng.standard_normal((M, n_sweeps)) +
              1j * rng.standard_normal((M, n_sweeps))) / np.sqrt(2.0))
    Y = noise.copy()

    for tgt in targets:
        tau = tgt['R'] / cfg.C_LIGHT
        fd  = 2.0 * tgt['v'] / cfg.WAVELENGTH
        amp = np.sqrt(10.0 ** (cfg.SNR_SINGLE_DB / 10.0) * tgt['rcs'])
        Y  += amp * (np.exp(-1j * 2.0 * np.pi * k * Df * tau)[:, None]
                     * np.exp(+1j * 2.0 * np.pi * fd * n * Tsw)[None, :])

    NFFT  = 8 * M
    rp    = np.fft.ifft(Y, n=NFFT, axis=0)
    power = np.mean(np.abs(rp) ** 2, axis=1)

    dr     = cfg.C_LIGHT / (NFFT * Df)
    r_axis = np.arange(NFFT) * dr
    n_v    = min(NFFT, int(np.ceil(cfg.C_LIGHT / Df / dr)) + 1)
    r_axis = r_axis[:n_v];  power = power[:n_v]

    med    = np.median(power)
    psd_dB = 10.0 * np.log10(power / (med + 1e-30) + 1e-30)
    return r_axis, psd_dB


# ═══════════════════════════════════════════════════════════════════════════
# 3.  FMCW beat signal and FFT range profile
# ═══════════════════════════════════════════════════════════════════════════

def synthesize_fmcw(targets: list, seed: int = None) -> tuple:
    """
    Synthesise a single FMCW chirp dechirped beat signal for K targets.

    y(t) = Σ_q A_q · exp(j·2π·f_beat_q·t)
    where  f_beat_q = (2B·R_q)/(c·Tsw) + (2v_q·fc)/c          [Eq. A1]

    Without 2-D processing, the FFT peak maps to apparent range:
        R_app = f_beat_measured · c · Tsw / (2B) = R + v·DISP_COEFF  [Eq. A2]

    Returns
    -------
    r_app  : ndarray  Apparent range axis [m]
    psd_dB : ndarray  FFT PSD [dB, normalised to noise median]
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED + 1 if seed is None else seed)

    N_samp = 8192
    dt     = _TSW_FMCW / N_samp
    t_vec  = np.arange(N_samp) * dt

    noise = ((rng.standard_normal(N_samp) +
              1j * rng.standard_normal(N_samp)) / np.sqrt(2.0))
    sig   = noise.copy()

    for tgt in targets:
        f_rng = 2.0 * _B_FMCW * tgt['R'] / (cfg.C_LIGHT * _TSW_FMCW)
        f_dop = 2.0 * tgt['v'] * _FC_FMCW / cfg.C_LIGHT
        amp   = np.sqrt(10.0 ** (cfg.SNR_SINGLE_DB / 10.0) * tgt['rcs'])
        sig  += amp * np.exp(1j * 2.0 * np.pi * (f_rng + f_dop) * t_vec)

    S     = np.fft.fft(sig, n=N_samp)
    freqs = np.fft.fftfreq(N_samp, d=dt)

    # Keep only positive-frequency half
    mask   = freqs >= 0
    freqs  = freqs[mask];   S = S[mask]
    psd    = np.abs(S) ** 2
    r_app  = freqs * cfg.C_LIGHT * _TSW_FMCW / (2.0 * _B_FMCW)

    # Trim to 15 m display window
    m15    = r_app <= 15.0
    r_app  = r_app[m15];   psd = psd[m15]

    med    = np.median(psd)
    psd_dB = 10.0 * np.log10(psd / (med + 1e-30) + 1e-30)
    return r_app, psd_dB


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Figure F16  —  M-FSK vs FMCW range profiles
# ═══════════════════════════════════════════════════════════════════════════

def plot_F16(targets, r_mfsk, psd_mfsk, r_fmcw, psd_fmcw, save=True):
    """
    F16 — Side-by-side comparison of M-FSK IFFT (a) and FMCW FFT (b).

    Each panel shows the same three targets; vertical dashed lines mark the
    TRUE ranges; coloured markers identify where each system detects the peaks.
    FMCW shows clearly displaced peaks (ghost targets) while M-FSK is clean.
    """
    # Normalise both profiles so noise floor = 0 dB
    psd_m = psd_mfsk - psd_mfsk.min()
    psd_f = psd_fmcw - psd_fmcw.min()
    y_max = max(psd_m.max(), psd_f.max()) * 1.08
    y_min = -3.0

    fig = plt.figure(figsize=(cfg.FIG_WIDTH_2COL, 2.8), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)

    panel_info = [
        (r_mfsk, psd_m,
         r'M-FSK IFFT  ($\Delta R = '
         + f'{cfg.C_LIGHT/(cfg.M_TONES*cfg.DELTA_F):.0f}' + r'$ m)',
         'M-FSK'),
        (r_fmcw, psd_f,
         r'FMCW FFT  ($B = '
         + f'{_B_FMCW/1e6:.0f}$ MHz, $T_{{\\rm sw}}='
         + f'{_TSW_FMCW*1e3:.0f}$ ms)',
         'FMCW'),
    ]
    counter = 0 # helper to plot
    for i_pan, (r_ax, psd, title, mode) in enumerate(panel_info):
        ax = fig.add_subplot(gs[i_pan])

        # Profile fill and line
        ax.fill_between(r_ax, y_min, psd,
                        color=cfg.COLORS['gray'], alpha=0.12, zorder=1)
        ax.plot(r_ax, psd, color=cfg.COLORS['gray'], lw=0.9, alpha=0.7, zorder=2)

        for tgt, col in zip(targets, _TGT_COLORS):
            R_true = tgt['R']
            # True-range vertical marker (both panels)
            ax.axvline(R_true, color=col, lw=3, ls='--', alpha=0.70, zorder=3)

            if mode == 'M-FSK':
                # Peak at true range
                idx = int(np.argmin(np.abs(r_ax - R_true)))
                ax.plot(r_ax[idx], psd[idx], '^', color=col, ms=7, zorder=6,
                        label=tgt['label'])
                # Range label above peak
                ylo, yhi = y_min, y_max
            
            else:
                # Peak at apparent (displaced) range
                R_app = tgt['R_fmcw']
                dR    = R_app - R_true
                ylo, yhi = y_min, y_max
    
                if 0 < R_app <= r_ax.max():
                    idx_app = int(np.argmin(np.abs(r_ax - R_app)))
                    ax.plot(r_ax[idx_app], psd[idx_app], 'v',
                            color=col, ms=7, zorder=6,
                            label=f'{tgt["label"]}')
                    # Displacement arrow (anchored inside data range)
                    ann_y = ylo + 0.55 * (yhi - ylo)
                    if counter == 1:
                        ann_y = ann_y-10
                    ax.annotate('', xy=(R_app, ann_y), xytext=(R_true, ann_y),
                                arrowprops=dict(arrowstyle='<->',
                                                color=col, lw=2,
                                                mutation_scale=8))
                    ax.text((R_true + R_app) / 2, ann_y + 0.03 * (yhi - ylo),
                            f'Δ = {dR:+.1f} m', fontsize=8,
                            ha='center', color=col, weight='bold')
                    counter += 1
                else:
                    ax.text(0.97, 0.55 - 0.12 * targets.index(tgt),
                            f'{tgt["label"]}\n→ {R_app:.1f} m (OOB)',
                            transform=ax.transAxes, fontsize=5.5,
                            ha='right', color=col, weight='bold')
                    ax.plot([], [], 'v', color=col, ms=7,
                            label=f'{tgt["label"]}  →  {R_app:.1f} m (OOB)')



        ax.set_xlabel('Range (m)',fontsize=10)
        ax.set_ylabel('Power (dB, normalised)',fontsize=10)
        ax.set_xlim([0.0, 12.5]);  ax.set_ylim([y_min, y_max])
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.grid(True, which='major')
        ax.grid(True, which='minor', alpha=0.15, lw=0.3)
        ax.legend(loc='upper left', fontsize=8, handlelength=1.5,
                  borderpad=0.3, labelspacing=0.2)
        ax.set_title(title, fontsize=cfg.FIG_TITLE, fontweight='normal', pad=3)

    if save:
        name = 'F16_multitarget_vs_fmcw'
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(cfg.OUTPUT_DIR, f'{name}.{ext}'),
                        dpi=(cfg.FIG_DPI if ext == 'png' else None))
        print(f'  Saved: figures/{name}.pdf | .png')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Data export  +  main entry point
# ═══════════════════════════════════════════════════════════════════════════

def export_data(targets, r_m, psd_m, r_f, psd_f):
    p = os.path.join(cfg.DATA_DIR, 'multi_target_data.npy')
    np.save(p, {'targets': targets, 'r_mfsk': r_m, 'psd_mfsk': psd_m,
                'r_fmcw': r_f, 'psd_fmcw': psd_f}, allow_pickle=True)
    print('  Exported: data/multi_target_data.npy')


def run(show: bool = False):
    """Run Stage 6a — Multi-Target + FMCW comparison.  Returns data tuple."""
    print('\n' + '═' * 60)
    print('  Stage 6a — Multi-Target Sensing + FMCW Comparison')
    print('  Generating Figure F16')
    print('═' * 60)

    targets = define_targets()
    _print_scenario(targets)

    print(f'  Synthesising M-FSK beat signals ({cfg.NSw} sweeps) …')
    r_mfsk, psd_mfsk = synthesize_mfsk(targets)

    print('  Synthesising FMCW beat signal …')
    r_fmcw, psd_fmcw = synthesize_fmcw(targets)

    plot_F16(targets, r_mfsk, psd_mfsk, r_fmcw, psd_fmcw, save=True)
    export_data(targets, r_mfsk, psd_mfsk, r_fmcw, psd_fmcw)

    if show:
        plt.show()
    plt.close('all')
    print('\n  Stage 6a complete — 1 figure + data saved.')
    return targets, r_mfsk, psd_mfsk, r_fmcw, psd_fmcw


if __name__ == '__main__':
    run()
