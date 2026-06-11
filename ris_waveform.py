"""
ris_waveform.py — Stage 2: RIS Waveform Layer — M-FSK Generation
=================================================================
Simulates the multi-frequency shift keying (M-FSK) radar waveform generated 
by a transmitter-RIS (T-RIS) via direct digital synthesis (DDS) phase-profile 
switching, and evaluates the accompanying beam-squint limits across the 
operational bandwidth.

Mathematical and Waveform Principles
------------------------------------
  * Multi-Tone Phase Modulation : Models the discrete non-linear reflection profile 
                                  where dynamic element state changes step through a 
                                  sequence of localized frequency sidebands.
  * Sideband Separation Logic   : Tracks the simultaneous generation of symmetrical 
                                  desired upper sidebands and mirror lower image 
                                  sidebands alongside residual carrier leakage feedthrough.
  * Frequency-Dependent Squint   : Evaluates spatial beam squint properties over the total 
                                  signal bandwidth to ensure angular deflection deviations 
                                  remain safely within the array's main half-power beamwidth.

Time-Frequency Synthesis Methods
--------------------------------
  * Complex Baseband Modeling   : Constructs continuous time-domain signal blocks stepping 
                                  sequentially through the tone allocations, utilizing windowing 
                                  interpolation functions for clean phase boundaries.
  * Short-Time Fourier Parsing  : Applies short-time Fourier transforms (STFT) to map the sequential 
                                  frequency steps over successive complete waveform sweep cycles.

Functional Deliverables
-----------------------
  * F7 — Discretized multi-tone power spectrum contrasting upper sidebands against image sidebands 
          and showing bandpass filter (BPF) attenuation regions.
  * F8 — Two-sweep baseband time-frequency spectrogram demonstrating temporal symbol stepping intervals.

Data Output
-----------
  * data/mfsk_waveform_data.npy : Structured array dictionary preserving spectral locations, absolute 
                                   amplitudes, and squint calculations for downstream link budget evaluation.

Execution Parameters
--------------------
    python ris_waveform.py
    from ris_waveform import run, compute_mfsk_spectrum, compute_beam_squint
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
from scipy import signal as sp_signal
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from pin_diode_model import compute_all as pin_compute, get_fc_values

# ─────────────────────────────────────────────────────────────────────────────
# plot properties
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


# ═══════════════════════════════════════════════════════════════════════════════
# M-FSK waveform characterisation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mfsk_spectrum(pin_data=None):
    """
    Compute the M-FSK spectral line positions and amplitudes.

    The T-RIS output for symbol k [Eq. 7]:
        s_n(t) = Tmean·δ(f − fc)
               + Tsb/2·[δ(f − (fc+fm,k)) + δ(f − (fc−fm,k))]

    Parameters
    ----------
    pin_data : dict or None
        Output of pin_diode_model.compute_all(); computed here if None.

    Returns
    -------
    spec : dict with keys:
        'fc_ghz'         : carrier frequency [GHz]
        'tones_upper_ghz': M upper-sideband centre frequencies [GHz]
        'tones_lower_ghz': M lower-sideband (image) frequencies [GHz]
        'Tsb_db'         : sideband amplitude at fc [dB]
        'Tmean_db'       : carrier feedthrough at fc [dB]
        'Tsb_lin'        : sideband amplitude (linear)
        'Tmean_lin'      : carrier feedthrough (linear)
        'bpf_lo_ghz'     : BPF lower edge [GHz]
        'bpf_hi_ghz'     : BPF upper edge [GHz]
    """
    if pin_data is None:
        pin_data = pin_compute(np.linspace(20e9, 35e9, 1501))

    fc_vals = get_fc_values(pin_data)
    Tsb_db   = fc_vals['Tsb_db']
    Tmean_db = fc_vals['Tmean_db']

    return {
        'fc_ghz':           cfg.FC / 1e9,
        'tones_upper_ghz':  (cfg.FC + cfg.FM_OFFSETS) / 1e9,
        'tones_lower_ghz':  (cfg.FC - cfg.FM_OFFSETS) / 1e9,
        'Tsb_db':           Tsb_db,
        'Tmean_db':         Tmean_db,
        'Tsb_lin':          10.0 ** (Tsb_db / 20.0),
        'Tmean_lin':        10.0 ** (Tmean_db / 20.0),
        'bpf_lo_ghz':       cfg.BPF_LO_HZ / 1e9,
        'bpf_hi_ghz':       cfg.BPF_HI_HZ / 1e9,
    }


def compute_beam_squint():
    """
    Compute beam squint Δθ_squint for each M-FSK tone  [Eq. 14].

    Δθ_squint ≈ arcsin(sin(θ1) · fm,k / fc)

    Returns
    -------
    sq : dict with keys:
        'fm_mhz'      : tone offsets [MHz]
        'squint_deg'  : squint angle per tone [degrees]
        'hpbw_deg'    : 3-dB HPBW of 16-element ULA at θ1 [degrees]
        'max_squint'  : worst-case squint (k = M) [degrees]
        'squint_hpbw_ratio' : max squint / HPBW [fraction]
    """
    sin_theta1 = np.sin(cfg.THETA1_RAD)
    squint_rad = np.arcsin(np.clip(sin_theta1 * cfg.FM_OFFSETS / cfg.FC, -1, 1))
    squint_deg = np.rad2deg(squint_rad)

    # HPBW for 16-element ULA: 0.886λ / (N·d·cosθ1)
    hpbw_rad = 0.886 * cfg.WAVELENGTH / (cfg.N_ELEM_X * cfg.D_ELEM * np.cos(cfg.THETA1_RAD))
    hpbw_deg = np.rad2deg(hpbw_rad)

    return {
        'fm_mhz':            cfg.FM_OFFSETS / 1e6,
        'squint_deg':        squint_deg,
        'hpbw_deg':          hpbw_deg,
        'max_squint':        squint_deg[-1],
        'squint_hpbw_ratio': squint_deg[-1] / hpbw_deg,
    }


def generate_baseband_signal(n_sweeps=2):
    """
    Generate a time-domain baseband M-FSK signal for spectrogram visualisation.

    Each symbol period: DDS steps to fm,k; T-RIS produces:
        upper sideband at +fm,k (desired)
        lower sideband at −fm,k (image)
    Modelled in complex baseband (fc removed).

    Parameters
    ----------
    n_sweeps : int   Number of complete M-tone sweeps to generate

    Returns
    -------
    t_us    : ndarray  Time vector [μs]
    sig     : ndarray  Complex baseband signal
    fs_bb   : float    Baseband sampling rate [Hz]
    sym_map : list     [(t_start_us, fm_mhz), …] one entry per symbol
    """
    T_segs = n_sweeps * cfg.M_TONES

    # Baseband BW: (M+2)·Δf with 4× oversampling for clean spectrogram
    fs_bb = 4.0 * (cfg.M_TONES + 2) * cfg.DELTA_F   # ≈ 1.4 GHz
    N_tot = int(np.ceil(T_segs * cfg.TSYM * fs_bb))
    t     = np.arange(N_tot) / fs_bb

    sig     = np.zeros(N_tot, dtype=complex)
    sym_map = []

    for seg in range(T_segs):
        k_idx   = seg % cfg.M_TONES
        fm_k    = cfg.FM_OFFSETS[k_idx]
        t_s     = seg * cfg.TSYM
        t_e     = (seg + 1) * cfg.TSYM
        mask    = (t >= t_s) & (t < t_e)
        n_seg   = int(np.sum(mask))
        if n_seg < 4:
            continue
        win = np.hanning(n_seg)
        # Both sidebands in complex baseband:
        sig[mask] = (np.exp(1j * 2.0 * np.pi * fm_k  * t[mask]) +
                     np.exp(-1j * 2.0 * np.pi * fm_k * t[mask])) * win
        sym_map.append((t_s * 1e6, fm_k / 1e6))   # (μs, MHz)

    return t * 1e6, sig, fs_bb, sym_map


# ═══════════════════════════════════════════════════════════════════════════════
# Figure helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, name):
    base = os.path.join(cfg.OUTPUT_DIR, name)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=cfg.FIG_DPI)
    print(f"  Saved: {base}.pdf  |  {base}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure F7 — M-FSK spectrum
# ═══════════════════════════════════════════════════════════════════════════════

def plot_F7(spec, save=True):
    """
    F7 — M-FSK spectrum showing all 12 upper/lower sidebands, carrier, BPF.

    Upper sideband tones: fc + k·Δf  (desired — inside BPF passband).
    Lower sideband tones: fc − k·Δf  (image  — rejected by BPF).
    Carrier feedthrough at fc.

    Demonstrates that a single BPF isolates all 12 desired tones
    from all 12 images simultaneously.

    """
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH_1COL, cfg.FIG_HEIGHT_STD))

    fc  = spec['fc_ghz']
    Tsb = spec['Tsb_db']
    Tmc = spec['Tmean_db']

    # ── BPF passband shading ──────────────────────────────────────────────────
    blo, bhi = spec['bpf_lo_ghz'], spec['bpf_hi_ghz']
    ax.axvspan(blo, bhi, alpha=0.09, color=cfg.COLORS['sideband'],
               label='BPF passband', zorder=0)
    for edge in [blo, bhi]:
        ax.axvline(edge, color=cfg.COLORS['sideband'], lw=0.9,
                   ls=':', alpha=0.6, zorder=1)

    # ── Carrier feedthrough (fc) ──────────────────────────────────────────────
    markerline, stemlines, _ = ax.stem(
        [fc], [Tmc], linefmt='-',
        markerfmt='D', basefmt='none', label=r'Carrier ($T_{\mathrm{mean}}$)')
    plt.setp(stemlines,   color=cfg.COLORS['carrier'], lw=1.5)
    plt.setp(markerline,  color=cfg.COLORS['carrier'], ms=5)

    # ── Upper sidebands (desired) ─────────────────────────────────────────────
    for i, fu in enumerate(spec['tones_upper_ghz']):
        ml, sl, _ = ax.stem(
            [fu], [Tsb], linefmt='-',
            markerfmt='^', basefmt='none',
            label='Upper sideband (desired)' if i == 0 else None)
        plt.setp(sl, color=cfg.COLORS['sideband'], lw=1.3)
        plt.setp(ml, color=cfg.COLORS['sideband'], ms=4)

    # ── Lower sidebands (image) ───────────────────────────────────────────────
    for i, fl in enumerate(spec['tones_lower_ghz']):
        ml, sl, _ = ax.stem(
            [fl], [Tsb], linefmt='--',
            markerfmt='v', basefmt='none',
            label='Lower sideband (image)' if i == 0 else None)
        plt.setp(sl, color=cfg.COLORS['off'], lw=1.2, linestyle='--')
        plt.setp(ml, color=cfg.COLORS['off'], ms=4)

    # ── fc reference ──────────────────────────────────────────────────────────
    ax.axvline(fc, color=cfg.COLORS['gray'], lw=0.7, ls=':', alpha=0.5, zorder=0)

    # ── Tone annotations — low-position labels to avoid legend clash ─────────
    fu0, fu_end = spec['tones_upper_ghz'][0],  spec['tones_upper_ghz'][-1]
    fl0         = spec['tones_lower_ghz'][0]
    lbl_y       = -32.5   # near the bottom, clear of stems

    ax.text(fu0 + 0.01, lbl_y+3, '+25\nMHz', ha='left', va='top',
            fontsize=10, color=cfg.COLORS['sideband'], style='italic')
    ax.text(fu_end, lbl_y+3, '+300\nMHz', ha='center', va='top',
            fontsize=10, color=cfg.COLORS['sideband'], style='italic')
    ax.text(fl0 - 0.01, lbl_y+3, '\u221225\nMHz', ha='right', va='top',
            fontsize=10, color=cfg.COLORS['off'], style='italic')

    ax.text(fc, -34.0, r'$f_c$', ha='center', fontsize=10)

    # ── Baselines ─────────────────────────────────────────────────────────────
    ax.axhline(-35, color='none')   # force y-axis range

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel('Frequency (GHz)',fontsize = 10)
    ax.set_ylabel('Normalised amplitude (dB)',fontsize = 10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlim([fc - 0.44, fc + 0.44])
    ax.set_ylim([-35, 5])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, which='major', alpha=0.3)
    ax.tick_params(axis='x', labelrotation=30)
    ax.legend(loc='upper right', fontsize=8, handlelength=1.5,
              ncol=1, borderpad=0.4, labelspacing=0.3, framealpha=0.9,bbox_to_anchor=(0.55,0.5))

    fig.tight_layout(pad=0.4)
    if save:
        _save_fig(fig, 'F7_mfsk_spectrum')
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Figure F8 — Time-frequency spectrogram
# ═══════════════════════════════════════════════════════════════════════════════

def plot_F8(spec, save=True):
    """
    F8 — Time-frequency spectrogram of the M-FSK signal over 2 sweeps.

    Uses STFT with a Hanning window whose length is 80 % of one symbol period.
    Both upper (desired, +fm) and lower (image, −fm) sidebands are visible.
    Tone steps are annotated in MHz.


    """
    t_us, sig, fs_bb, sym_map = generate_baseband_signal(n_sweeps=2)

    # ── Short-time Fourier transform ─────────────────────────────────────────
    nperseg  = max(64, int(fs_bb * cfg.TSYM * 0.80))
    noverlap = nperseg // 2

    f_spec, t_spec, Sxx = sp_signal.spectrogram(
        sig, fs=fs_bb,
        window=sp_signal.windows.hann(nperseg),
        nperseg=nperseg, noverlap=noverlap,
        return_onesided=False,
        scaling='density')

    # fftshift to centre at DC
    f_mhz = np.fft.fftshift(f_spec) / 1e6
    Sxx_s = np.fft.fftshift(Sxx, axes=0)
    t_us_spec = t_spec * 1e6

    Sxx_db = 10.0 * np.log10(np.abs(Sxx_s) + 1e-30)
    vmin   = np.percentile(Sxx_db, 68)
    vmax   = Sxx_db.max()

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH_1COL, cfg.FIG_HEIGHT_STD))

    pcm = ax.pcolormesh(t_us_spec, f_mhz, Sxx_db,
                        cmap='inferno', shading='gouraud',
                        vmin=vmin, vmax=vmax, rasterized=True)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.08)
    cbar.set_label('Power spectral density (dB)', fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    # ── Annotate tone frequencies for first sweep ─────────────────────────────
    for i, (t_s, fm_mhz) in enumerate(sym_map[:cfg.M_TONES]):
        t_mid = t_s + cfg.TSYM * 1e6 * 0.5
        if t_mid < t_us_spec.max():
            ax.text(t_mid, fm_mhz + 16, f'{fm_mhz:.0f}',
                    fontsize=5, color='white', ha='center', va='bottom',
                    fontweight='bold')

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axhline(0, color='white', lw=0.6, ls='--', alpha=0.45)

    # BPF boundary markers (baseband reference at ±12.5 and ±312.5 MHz)
    for bnd in [cfg.BPF_LO_HZ - cfg.FC, cfg.BPF_HI_HZ - cfg.FC]:
        ax.axhline(bnd / 1e6, color='cyan', lw=0.6, ls=':', alpha=0.55)

    # Tone boundary arrows (upper sideband desired band)
    ax.text(t_us_spec.max() * 0.97,  (cfg.BPF_LO_HZ - cfg.FC) / 1e6 + 5,
            'BPF', fontsize=5.5, color='cyan', ha='right', va='bottom')
    ax.text(t_us_spec.max() * 0.97,  (cfg.BPF_HI_HZ - cfg.FC) / 1e6 - 5,
            'BPF', fontsize=5.5, color='cyan', ha='right', va='top')

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel(r'Time ($\mu$s)')
    ax.set_ylabel('Baseband frequency (MHz)')
    ax.set_ylim([-340, 340])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.set_xlim([t_us_spec.min(), t_us_spec.max()])

    fig.tight_layout(pad=0.4)
    if save:
        _save_fig(fig, 'F8_mfsk_spectrogram')
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Data export
# ═══════════════════════════════════════════════════════════════════════════════

def export_data(spec, sq):
    """Save spectrum and squint data for downstream stages."""
    path = os.path.join(cfg.DATA_DIR, 'mfsk_waveform_data.npy')
    np.save(path, {'spectrum': spec, 'squint': sq}, allow_pickle=True)
    print(f"  Exported: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Console summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_waveform_summary(spec, sq):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  M-FSK Waveform Summary")
    print(sep)
    print(f"  Tones M              = {cfg.M_TONES}")
    print(f"  Tone spacing Δf      = {cfg.DELTA_F/1e6:.0f} MHz")
    print(f"  Tone range           = {cfg.FM_OFFSETS[0]/1e6:.0f}–{cfg.FM_OFFSETS[-1]/1e6:.0f} MHz above fc")
    print(f"  |Tsb| at fc          = {spec['Tsb_db']:.2f} dB")
    print(f"  |Tmean| at fc        = {spec['Tmean_db']:.2f} dB")
    print(f"  BPF passband         = fc+{(cfg.BPF_LO_HZ-cfg.FC)/1e6:.1f} to fc+{(cfg.BPF_HI_HZ-cfg.FC)/1e6:.1f} MHz  ({spec['bpf_lo_ghz']:.4f}–{spec['bpf_hi_ghz']:.4f} GHz)")
    print(sep)
    print(f"  Symbol rate (two-stage DDS)   = {cfg.F_SYMBOL/1e3:.1f} kHz")
    print(f"  Symbol rate (single-stage SPI)= {cfg.F_SYMBOL_SINGLE_STAGE/1e3:.1f} kHz")
    print(f"  Speed-up factor               = {cfg.F_SYMBOL/cfg.F_SYMBOL_SINGLE_STAGE:.0f}×")
    print(sep)
    print(f"  Beam squint (Eq. 14)")
    print(f"    θ1                 = {np.rad2deg(cfg.THETA1_RAD):.1f}°")
    print(f"    HPBW (N={cfg.N_ELEM_X})       = {sq['hpbw_deg']:.2f}°")
    print(f"    Max squint (k={cfg.M_TONES})  = {sq['max_squint']:.3f}°")
    print(f"    Max squint / HPBW  = {sq['squint_hpbw_ratio']*100:.1f}%  (< 10% target)")
    print(sep + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(show=False):
    """
    Run Stage 2 — RIS Waveform Layer.

    Computes M-FSK spectrum and beam squint, generates F7 and F8,
    exports data for the pipeline.

    Parameters
    ----------
    show : bool  If True, call plt.show() after saving.

    Returns
    -------
    spec : dict  Spectrum characterisation (from compute_mfsk_spectrum)
    sq   : dict  Beam squint analysis    (from compute_beam_squint)
    """
    print("\n" + "═" * 60)
    print("  Stage 2 — RIS Waveform Layer")
    print("  Generating Figures F7, F8")
    print("═" * 60)

    # Get PIN diode data for accurate Tsb / Tmean values at fc
    pin_data = pin_compute(np.linspace(20e9, 35e9, 1501))
    spec     = compute_mfsk_spectrum(pin_data)
    sq       = compute_beam_squint()

    print_waveform_summary(spec, sq)

    plot_F7(spec, save=True)
    plot_F8(spec, save=True)
    export_data(spec, sq)

    if show:
        plt.show()
    plt.close('all')

    print("  Stage 2 complete — 2 figures + data saved.")
    return spec, sq


if __name__ == '__main__':
    run()
