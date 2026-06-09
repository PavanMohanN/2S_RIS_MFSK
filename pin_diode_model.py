"""
pin_diode_model.py — Stage 1: Device Physics — PIN Diode Model
==============================================================
Implements the PIN diode equivalent-circuit model for 28 GHz T-RIS elements
and generates publication-ready Figures F4, F5, F6.

Paper equations:
    Z1(f, state) = Rs + RJ(state) / [1 + jωCJ·RJ(state)]    [Eq. 1]
    Tn(f, state) = 2Z0 / [2Z0 + Z1(f, state)]               [Eq. 2]
    Tmean(f)     = [Tn(f,on) + Tn(f,off)] / 2               [Eq. 5]
    Tsb(f)       = |Tn(f,on) − Tn(f,off)| / 2               [Eq. 8]

Output figures
--------------
F4 — |Tn(f)| amplitude: on-state, off-state, sideband coeff, carrier feedthrough
F5 — ∠Tn(f) phase response: on/off states, annotated phase difference at fc
F6 — |Z1(f)| impedance magnitude (log scale): on/off states, Z0 reference

Output data
-----------
data/pin_diode_data.npy — dict with all computed arrays (for pipeline use)

Usage
-----
    python pin_diode_model.py          # run standalone, save figures
    from pin_diode_model import run, compute_all, transmission_coeff
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# IEEE TVT publication style
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
    'text.usetex':          False,
    'mathtext.fontset':     'cm',
})


# ═══════════════════════════════════════════════════════════════════════════════
# Core physics model
# ═══════════════════════════════════════════════════════════════════════════════

def pin_impedance(f_hz, state='on'):
    """
    Compute PIN diode series impedance Z1(f, state)  [Ω].

    Z1 = Rs + RJ(state) / (1 + jω·CJ·RJ(state))      [Eq. 1]

    The impedance models a series RS in series with the parallel combination
    of the junction resistance RJ and junction capacitance CJ.

    Parameters
    ----------
    f_hz  : array_like  frequency [Hz]
    state : str         'on'  → forward-biased (low-impedance)
                        'off' → reverse-biased (high-impedance, capacitive)

    Returns
    -------
    Z1 : complex ndarray  [Ω]
    """
    f = np.asarray(f_hz, dtype=float)
    omega = 2.0 * np.pi * f
    RJ = cfg.RJ_ON if state == 'on' else cfg.RJ_OFF
    # Parallel junction combination: RJ ∥ (1/jωCJ)
    Z_junc = RJ / (1.0 + 1j * omega * cfg.CJ * RJ)
    return cfg.RS + Z_junc


def transmission_coeff(f_hz, state='on'):
    """
    Compute T-RIS element transmission coefficient Tn(f, state).

    Tn = 2Z0 / (2Z0 + Z1)    [Eq. 2]
    Models the element as a shunt impedance Z1 in a Z0 transmission line.

    Returns
    -------
    Tn : complex ndarray  (dimensionless)
    """
    Z1 = pin_impedance(f_hz, state)
    return (2.0 * cfg.Z0) / (2.0 * cfg.Z0 + Z1)


def sideband_coeff(f_hz):
    """
    Sideband conversion coefficient Tsb(f)  [Eq. 8].

    Tsb = |Tn(f,on) − Tn(f,off)| / 2

    Half the peak-to-peak modulation depth; sets the power in each sideband.
    """
    return np.abs(transmission_coeff(f_hz, 'on') -
                  transmission_coeff(f_hz, 'off')) / 2.0


def carrier_feedthrough(f_hz):
    """
    Carrier feedthrough amplitude |Tmean(f)|  [Eq. 5].

    Tmean = (Tn(f,on) + Tn(f,off)) / 2

    DC component present at output regardless of switching state.
    """
    return np.abs((transmission_coeff(f_hz, 'on') +
                   transmission_coeff(f_hz, 'off')) / 2.0)


def compute_all(f_hz):
    """
    Compute all PIN diode quantities over the frequency vector f_hz.

    Returns
    -------
    d : dict
        'f_ghz'        : frequency [GHz]
        'Z1_on'        : complex impedance, on-state  [Ω]
        'Z1_off'       : complex impedance, off-state [Ω]
        'Ton'          : complex transmission, on-state
        'Toff'         : complex transmission, off-state
        'Tsb'          : sideband coeff (real, ≥ 0)
        'Tmean'        : carrier feedthrough (real, ≥ 0)
        'Ton_db'       : |Ton| in dB
        'Toff_db'      : |Toff| in dB
        'Tsb_db'       : Tsb in dB
        'Tmean_db'     : Tmean in dB
        'phase_on_deg' : ∠Ton [degrees]
        'phase_off_deg': ∠Toff [degrees]
        'Z1_on_mag'    : |Z1_on| [Ω]
        'Z1_off_mag'   : |Z1_off| [Ω]
    """
    f  = np.asarray(f_hz, dtype=float)
    Ton  = transmission_coeff(f, 'on')
    Toff = transmission_coeff(f, 'off')
    Tsb  = np.abs(Ton - Toff) / 2.0
    Tmean = np.abs((Ton + Toff) / 2.0)
    Z1_on  = pin_impedance(f, 'on')
    Z1_off = pin_impedance(f, 'off')

    def to_db(x):
        return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))

    return {
        'f_ghz':         f / 1e9,
        'Z1_on':         Z1_on,
        'Z1_off':        Z1_off,
        'Ton':           Ton,
        'Toff':          Toff,
        'Tsb':           Tsb,
        'Tmean':         Tmean,
        'Ton_db':        to_db(Ton),
        'Toff_db':       to_db(Toff),
        'Tsb_db':        to_db(Tsb),
        'Tmean_db':      to_db(Tmean),
        'phase_on_deg':  np.angle(Ton,  deg=True),
        'phase_off_deg': np.angle(Toff, deg=True),
        'Z1_on_mag':     np.abs(Z1_on),
        'Z1_off_mag':    np.abs(Z1_off),
    }


def get_fc_values(d):
    """Extract key scalar values at fc from compute_all() output dict."""
    idx = np.argmin(np.abs(d['f_ghz'] - cfg.FC / 1e9))
    return {k: v[idx] for k, v in d.items() if isinstance(v, np.ndarray)}


def print_operating_point(d):
    """Print key scalar values at fc for paper verification."""
    v = get_fc_values(d)
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  PIN Diode — Operating point at fc = {cfg.FC/1e9:.1f} GHz")
    print(sep)
    print(f"  |Tn(on)|         = {v['Ton_db']:.2f} dB")
    print(f"  |Tn(off)|        = {v['Toff_db']:.2f} dB")
    print(f"  |Tsb|            = {v['Tsb_db']:.2f} dB   ← sideband amplitude")
    print(f"  |Tmean|          = {v['Tmean_db']:.2f} dB   ← carrier feedthrough")
    print(f"  ∠Tn(on)          = {v['phase_on_deg']:.1f}°")
    print(f"  ∠Tn(off)         = {v['phase_off_deg']:.1f}°")
    print(f"  ΔPhase (off−on)  = {v['phase_off_deg'] - v['phase_on_deg']:.1f}°")
    print(f"  |Z1(on)|         = {v['Z1_on_mag']:.2f} Ω   (≪ Z0 = {cfg.Z0:.0f} Ω)")
    print(f"  |Z1(off)|        = {v['Z1_off_mag']:.1f} Ω")
    print(f"  Tsb variation    = "
          f"{d['Tsb_db'].max() - d['Tsb_db'].min():.2f} dB over 20–35 GHz  (< 0.4 dB target)")
    print(sep + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, name):
    """Save figure as both PDF and PNG into cfg.OUTPUT_DIR."""
    base = os.path.join(cfg.OUTPUT_DIR, name)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=cfg.FIG_DPI)
    print(f"  Saved: {base}.pdf  |  {base}.png")


def _mark_fc(ax, d, key, color, marker='o'):
    """Mark operating point at fc on an Axes."""
    idx = np.argmin(np.abs(d['f_ghz'] - cfg.FC / 1e9))
    ax.plot(cfg.FC / 1e9, d[key][idx],
            marker, color=color, ms=cfg.FIG_MS, zorder=6, clip_on=False)
    return d[key][idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Figure F4 — Transmission amplitude vs frequency
# ═══════════════════════════════════════════════════════════════════════════════

def plot_F4(d, save=True):
    """
    F4 — |Tn(f)| amplitude [dB] vs frequency.

    Four curves:
      • On-state  |Tn(f, on)|             (near-flat ≈ −0.26 dB)
      • Off-state |Tn(f, off)|            (−8.02 dB at fc)
      • Sideband  |Tsb(f)|                (−7.17 dB at fc)
      • Carrier   |Tmean(f)|              (−4.46 dB at fc)

    Format: single-column IEEE TVT (3.5 × 2.6 in).
    """
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH_1COL, cfg.FIG_HEIGHT_STD))

    fc_ghz = cfg.FC / 1e9
    f = d['f_ghz']

    # ── Curves ──────────────────────────────────────────────────────────────
    curves = [
        ('Ton_db',   cfg.COLORS['on'],       '-',  r'$|T_n(f,\mathrm{on})|$'),
        ('Toff_db',  cfg.COLORS['off'],       '--', r'$|T_n(f,\mathrm{off})|$'),
        ('Tsb_db',   cfg.COLORS['sideband'],  '-.', r'$|T_{\mathrm{sb}}(f)|$'),
        ('Tmean_db', cfg.COLORS['carrier'],   ':',  r'$|T_{\mathrm{mean}}(f)|$'),
    ]
    for key, col, ls, lbl in curves:
        ax.plot(f, d[key], color=col, linestyle=ls, lw=cfg.FIG_LW, label=lbl)

    # ── fc reference line ────────────────────────────────────────────────────
    ax.axvline(fc_ghz, color=cfg.COLORS['gray'], lw=0.7, ls=':', alpha=0.7, zorder=1)

    # ── Mark & annotate operating point for each curve ───────────────────────
    markers = ['o', 's', '^', 'D']
    for (key, col, _, _), mkr in zip(curves, markers):
        _mark_fc(ax, d, key, col, marker=mkr)

    # Value annotations at fc (right side)
    idx = np.argmin(np.abs(f - fc_ghz))
    offsets = {'Ton_db': 0.6, 'Toff_db': -0.8, 'Tsb_db': 0.6, 'Tmean_db': -0.8}
    for key, col, _, _ in curves:
        val = d[key][idx]
        ax.annotate(f'{val:.1f}', xy=(fc_ghz, val),
                    xytext=(fc_ghz + 0.5, val + offsets.get(key, 0)),
                    fontsize=10, color=col, ha='left', va='center',
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.5))

    ax.text(fc_ghz - 0.2, -21.0, r'$f_c$=28 GHz',
            fontsize=10,  ha='center', va='bottom')

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xlabel('Frequency (GHz)',fontsize = 10)
    ax.set_ylabel('Transmission coefficient (dB)',fontsize = 10)
    ax.set_xlim([20, 35])
    ax.set_ylim([-22, 2])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, which='major')
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.3)
    ax.legend(loc='lower left',bbox_to_anchor=(0,0.16), ncol=2, fontsize=8,
              handlelength=2.2, columnspacing=0.8, borderpad=0.5,
              framealpha=0.9)

    fig.tight_layout(pad=0.4)
    if save:
        _save_fig(fig, 'F4_transmission_amplitude')
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Figure F5 — Phase response vs frequency
# ═══════════════════════════════════════════════════════════════════════════════

def plot_F5(d, save=True):
    """
    F5 — Phase ∠Tn(f) [degrees] vs frequency.

    Two curves: on-state (≈ 0° flat) and off-state (rises with frequency
    due to capacitive CJ).  Annotates the 64.5° phase difference at fc.

    Format: single-column IEEE TVT (3.5 × 2.6 in).
    """
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH_1COL, cfg.FIG_HEIGHT_STD))

    fc_ghz = cfg.FC / 1e9
    f = d['f_ghz']

    # ── Curves ──────────────────────────────────────────────────────────────
    ax.plot(f, d['phase_on_deg'],  color=cfg.COLORS['on'],  lw=cfg.FIG_LW,
            label=r'$\angle T_n(f,\mathrm{on})$')
    ax.plot(f, d['phase_off_deg'], color=cfg.COLORS['off'], lw=cfg.FIG_LW,
            ls='--', label=r'$\angle T_n(f,\mathrm{off})$')

    # ── fc reference ─────────────────────────────────────────────────────────
    ax.axvline(fc_ghz, color=cfg.COLORS['gray'], lw=0.7, ls=':', alpha=0.7)

    # ── Annotate phase difference at fc ──────────────────────────────────────
    idx = np.argmin(np.abs(f - fc_ghz))
    ph_on  = d['phase_on_deg'][idx]
    ph_off = d['phase_off_deg'][idx]
    delta  = ph_off - ph_on
    mid    = (ph_on + ph_off) / 2.0

    # Double-headed arrow
    ax.annotate('', xy=(fc_ghz, ph_off), xytext=(fc_ghz, ph_on),
                arrowprops=dict(arrowstyle='<->', color='black',
                                lw=0.9, mutation_scale=10))
    # Build label string carefully: mixing f-string value with LaTeX \circ
    delta_label = r'$\Delta\phi = ' + f'{delta:.1f}' + r'^{\circ}$'
    ax.text(fc_ghz + 0.35, mid, delta_label,
            fontsize=10, va='center', color='black')

    # Operating-point markers
    ax.plot(fc_ghz, ph_on,  'o', color=cfg.COLORS['on'],  ms=cfg.FIG_MS, zorder=6)
    ax.plot(fc_ghz, ph_off, 's', color=cfg.COLORS['off'], ms=cfg.FIG_MS, zorder=6)

    # fc label — fixed y position below both curves
    ax.text(fc_ghz - 0.25, -4.0,
            r'$f_c\!=\!28$ GHz', fontsize=10, 
            ha='center', va='top')

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xlabel('Frequency (GHz)',fontsize=10)
    ax.set_ylabel('Phase (degrees)',fontsize = 10)
    ax.set_xlim([20, 35])
    ax.set_ylim([-10, 80])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(True, which='major')
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.3)
    ax.legend(loc='upper right', fontsize=10, handlelength=2.0,bbox_to_anchor=(0.45,0.5))

    fig.tight_layout(pad=0.4)
    if save:
        _save_fig(fig, 'F5_phase_response')
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Figure F6 — Impedance magnitude vs frequency
# ═══════════════════════════════════════════════════════════════════════════════

def plot_F6(d, save=True):
    """
    F6 — |Z1(f)| impedance magnitude [Ω] vs frequency (log-y scale).

    On-state: near-resistive, flat (dominated by Rs = 2 Ω).
    Off-state: capacitive rolloff as Cj reactance decreases with frequency.
    Reference line at Z0 = 50 Ω.

    Format: single-column IEEE TVT (3.5 × 2.6 in).
    """
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH_1COL, cfg.FIG_HEIGHT_STD))

    fc_ghz = cfg.FC / 1e9
    f = d['f_ghz']

    # ── Curves ──────────────────────────────────────────────────────────────
    ax.semilogy(f, d['Z1_on_mag'],  color=cfg.COLORS['on'],  lw=cfg.FIG_LW,
                label=r'$|Z_1(f,\mathrm{on})|$')
    ax.semilogy(f, d['Z1_off_mag'], color=cfg.COLORS['off'], lw=cfg.FIG_LW,
                ls='--', label=r'$|Z_1(f,\mathrm{off})|$')

    # ── Z0 reference line ────────────────────────────────────────────────────
    ax.axhline(cfg.Z0, color=cfg.COLORS['gray'], lw=0.9, ls=':',
               label=f'$Z_0 = {cfg.Z0:.0f}\\,\\Omega$')

    # ── fc reference ─────────────────────────────────────────────────────────
    ax.axvline(fc_ghz, color=cfg.COLORS['gray'], lw=0.7, ls=':', alpha=0.7)

    # ── Annotate values at fc ─────────────────────────────────────────────────
    idx = np.argmin(np.abs(f - fc_ghz))
    z_on  = d['Z1_on_mag'][idx]
    z_off = d['Z1_off_mag'][idx]

    ax.plot(fc_ghz, z_on,  'o', color=cfg.COLORS['on'],  ms=cfg.FIG_MS, zorder=6)
    ax.plot(fc_ghz, z_off, 's', color=cfg.COLORS['off'], ms=cfg.FIG_MS, zorder=6)

    ax.annotate(f'{z_on:.1f}$\\,\\Omega$',
                xy=(fc_ghz, z_on),
                xytext=(fc_ghz + 0.5, z_on * 1.5),
                fontsize=10, color=cfg.COLORS['on'],
                arrowprops=dict(arrowstyle='-', color=cfg.COLORS['on'], lw=0.5))
    ax.annotate(f'{z_off:.0f}$\\,\\Omega$',
                xy=(fc_ghz, z_off),
                xytext=(fc_ghz + 0.5, z_off * 0.4),
                fontsize=10, color=cfg.COLORS['off'],
                arrowprops=dict(arrowstyle='-', color=cfg.COLORS['off'], lw=0.5))

    ax.text(fc_ghz - 0.3, ax.get_ylim()[0] * 2,
            r'$f_c$', fontsize=10,  ha='center')

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xlabel('Frequency (GHz)',fontsize = 10)
    ax.set_ylabel(r'Impedance magnitude ($\Omega$)',fontsize = 10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlim([20, 35])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, which='both')
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.3)
    ax.legend(loc='upper right', fontsize=10, handlelength=2.0,bbox_to_anchor=(0.45,0.5))

    fig.tight_layout(pad=0.4)
    if save:
        _save_fig(fig, 'F6_impedance_magnitude')
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Data export
# ═══════════════════════════════════════════════════════════════════════════════

def export_data(d):
    """Save computed arrays to data/ for downstream pipeline stages."""
    path = os.path.join(cfg.DATA_DIR, 'pin_diode_data.npy')
    np.save(path, d, allow_pickle=True)
    print(f"  Exported: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(f_start=20e9, f_stop=35e9, n_pts=1501, show=False):
    """
    Run Stage 1 — Device physics: compute PIN model, print summary, save figures.

    Parameters
    ----------
    f_start : float  Frequency sweep start [Hz]  (default 20 GHz)
    f_stop  : float  Frequency sweep stop  [Hz]  (default 35 GHz)
    n_pts   : int    Number of frequency points  (default 1501)
    show    : bool   Call plt.show() after saving (default False)

    Returns
    -------
    d : dict  All computed quantities (keys listed in compute_all docstring)
    """
    print("\n" + "═" * 60)
    print("  Stage 1 — Device Physics: PIN Diode Model")
    print("  Generating Figures F4, F5, F6")
    print("═" * 60)

    f_hz = np.linspace(f_start, f_stop, n_pts)
    d    = compute_all(f_hz)

    print_operating_point(d)

    plot_F4(d, save=True)
    plot_F5(d, save=True)
    plot_F6(d, save=True)
    export_data(d)

    if show:
        plt.show()
    plt.close('all')

    print(f"\n  Stage 1 complete — 3 figures + data saved.")
    return d


if __name__ == '__main__':
    run()
