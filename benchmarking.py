"""
benchmarking.py  —   System-Level Benchmarking
=========================================================
Performs a system-level quantitative performance and power consumption
comparison across four distinct sensing and radar architectures.

Architectures Evaluated
-----------------------
  1. Two-stage T-RIS/R-RIS (Proposed) — Internal DDS-driven clock boundary
  2. Single-stage STM-RIS (Baseline)  — Direct serialization SPI reprogramming boundary
  3. Conventional FMCW Radar          — Continuous chirp-based architecture
  4. AESA Phased-Array Radar          — Active electronically-scanned transmitter/receiver array

Core Metrics Compared
---------------------
  * Power Consumption (W) : Component-level breakdown of RF, bias, and processing overhead
  * Waveform Agility      : Reprogramming rate and clock-limited symbol velocity limits
  * Range Resolution      : Physical wave mechanics and multi-tone frequency bandwidth scaling
  * Beam Agility          : Phase-profile updating latency bounds
  * Hardware Cost         : Normalized qualitative scaling of structural and element complexity

Output Deliverables
-------------------
  * F17 — Stacked horizontal power breakdown bar chart for all architectures
  * F18 — Polar radar/spider visualization chart of normalized performance limits
  * T4  — Two-stage vs. single-stage structural and electrical engineering comparison
  * T5  — Comparative summary against reported values in recent sensing literature
  * T6  — Algorithmic and processing-step computational complexity scaling properties
  * T8  — Component-level breakdown and fractional analysis of the proposed two-stage system

Data Output
-----------
  * data/benchmarking_data.npy : Unified architecture and normalized performance metric arrays

Execution Parameters
--------------------
    python benchmarking.py
    from benchmarking import run
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ── defines plots and their properties ──────────────────
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
os.makedirs(cfg.TABLE_DIR,  exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Architecture definitions
# ═══════════════════════════════════════════════════════════════════════════

def define_architectures() -> list:
    """
    Build the four architecture records.

    Each record is a dict with:
        'name'   : multi-line display name
        'color'  : matplotlib colour
        'power'  : OrderedDict  component -> Watts
        'total'  : float  total power [W]  (computed here)
        'metrics': dict  metric_key -> raw value (normalised in plot_F18)


    """
    N  = cfg.N_ELEM
    c  = cfg.C_LIGHT
    M  = cfg.M_TONES
    Df = cfg.DELTA_F

    # ── Two-stage T-RIS / R-RIS (proposed) ───────────────────────────────
    P_PA   = cfg.PTX_W / 0.20              # 0.500 W  (PA input, η = 20 %)
    P_bT   = N * 2.0e-3                    # 0.512 W  T-RIS PIN bias
    P_bR   = N * 2.0e-3                    # 0.512 W  R-RIS PIN bias
    P_dds  = N * cfg.ESW * M * cfg.F_SYMBOL  # ≈0.022 W  DDS switching
    P_dsp  = 0.550                         # W  DSP + ADC
    P_lna  = 0.080                         # W  Rx LNA
    P_ctrl = 0.150                         # W  controller + clock

    two_stage = {
        'name':  'Two-stage\nT-RIS/R-RIS\n(proposed)',
        'color': cfg.COLORS['two_stage'],
        'power': {
            'PA input':        P_PA,
            'T-RIS bias':      P_bT,
            'R-RIS bias':      P_bR,
            'DDS switching':   P_dds,
            'Rx chain / LNA':  P_lna,
            'DSP + ADC':       P_dsp,
            'Controller':      P_ctrl,
            'Phase shifters':  0.0,         # not applicable
            'Chirp generator': 0.0,
        },
    }

    # ── Single-stage STM-RIS (baseline) ──────────────────────────────────
    # One combined panel; SPI reprogramming per symbol → higher ctrl power.
    # No DDS; slightly more bias (larger current for combined T+R function).
    P_bSTM  = N * 2.0e-3 * 1.5            # 0.768 W  combined RIS panel
    P_spi   = 0.085                        # W  SPI controller (13 kHz × N × B)
    P_ctrl2 = 0.220                        # W  more overhead without DDS

    single_stage = {
        'name':  'Single-stage\nSTM-RIS\n(baseline)',
        'color': cfg.COLORS['single'],
        'power': {
            'PA input':        P_PA,
            'T-RIS bias':      0.0,
            'R-RIS bias':      P_bSTM,      # combined
            'DDS switching':   0.0,
            'Rx chain / LNA':  P_lna,
            'DSP + ADC':       P_dsp,
            'Controller':      P_ctrl2,
            'Phase shifters':  0.0,
            'Chirp generator': 0.0,
            'SPI ctrl':        P_spi,
        },
    }

    # ── Conventional FMCW radar ───────────────────────────────────────────
    fmcw = {
        'name':  'FMCW\nradar',
        'color': cfg.COLORS['fmcw'],
        'power': {
            'PA input':        0.80,        # W  PA (FMCW higher duty cycle)
            'T-RIS bias':      0.0,
            'R-RIS bias':      0.0,
            'DDS switching':   0.0,
            'Rx chain / LNA':  0.45,        # W  LNA + mixer + filter
            'DSP + ADC':       0.75,        # W  ADC + 2-D FFT
            'Controller':      0.12,
            'Phase shifters':  0.0,
            'Chirp generator': 0.35,        # W  VCO + chirp synthesiser
            'SPI ctrl':        0.0,
        },
    }

    # ── AESA phased-array radar ───────────────────────────────────────────
    # Use config calibrated values: N*P_chan + P_RF = 4.71 W
    aesa = {
        'name':  'AESA\nphased array',
        'color': cfg.COLORS['aesa'],
        'power': {
            'PA input':        cfg.P_RF_AESA_W,       # 0.87 W  back-end RF
            'T-RIS bias':      0.0,
            'R-RIS bias':      0.0,
            'DDS switching':   0.0,
            'Rx chain / LNA':  0.0,
            'DSP + ADC':       0.80,
            'Controller':      0.15,
            'Phase shifters':  N * 3.5e-3,            # 0.896 W  analog PS
            'Chirp generator': 0.0,
            'SPI ctrl':        0.0,
            'T/R modules':     N * cfg.P_CHAN_AESA_W,  # 3.84 W (PA + LNA per ch.)
        },
    }

    archs = [two_stage, single_stage, fmcw, aesa]
    for a in archs:
        a['total'] = sum(a['power'].values())

    # ── Performance metrics (raw values, normalised in plot_F18) ─────────
    two_stage['metrics'] = {
        'Symbol rate\n(kHz)':     cfg.F_SYMBOL / 1e3,          # 714.3
        'Power eff.\n(1/W)':      1.0 / two_stage['total'],
        'Range res.\n(1/ΔR)':     1.0 / (c / (M * Df)),        # 1/1.0m
        'Beam agility\n(score)':  1.00,    # DDS → fastest
        'HW cost\n(score)':       0.78,    # two RIS panels, no PA array
    }
    single_stage['metrics'] = {
        'Symbol rate\n(kHz)':     cfg.F_SYMBOL_SINGLE_STAGE / 1e3,   # 13.0
        'Power eff.\n(1/W)':      1.0 / single_stage['total'],
        'Range res.\n(1/ΔR)':     1.0 / (c / (M * Df)),
        'Beam agility\n(score)':  0.02,    # SPI → slowest
        'HW cost\n(score)':       0.85,    # one RIS panel
    }
    fmcw['metrics'] = {
        'Symbol rate\n(kHz)':     1.0 / (50e-6) / 1e3,   # 1/Tsw ≈ 20 kHz
        'Power eff.\n(1/W)':      1.0 / fmcw['total'],
        'Range res.\n(1/ΔR)':     1.0 / 0.15,             # 0.15 m @ 1 GHz BW
        'Beam agility\n(score)':  0.55,
        'HW cost\n(score)':       0.65,
    }
    aesa['metrics'] = {
        'Symbol rate\n(kHz)':     50.0,    # typical PRF
        'Power eff.\n(1/W)':      1.0 / aesa['total'],
        'Range res.\n(1/ΔR)':     1.0 / 0.15,
        'Beam agility\n(score)':  0.90,    # electronic steering
        'HW cost\n(score)':       0.25,    # N active T/R modules
    }

    return archs


def normalise_metrics(archs: list) -> dict:
    """Normalise each metric to [0, 1] across all architectures (max → 1.0)."""
    keys = list(archs[0]['metrics'].keys())
    norm = {}
    for k in keys:
        vals = np.array([a['metrics'][k] for a in archs])
        norm[k] = vals / (vals.max() + 1e-30)
    return norm


def _print_summary(archs):
    sep = '─' * 70
    print(f'\n{sep}')
    print('  Architecture Benchmarking Summary')
    print(sep)
    print(f'  {"Architecture":<28}  {"Power (W)":>9}  '
          f'{"Symbol rate (kHz)":>18}  {"ΔR (m)":>7}')
    print('  ' + '-' * 66)
    for a in archs:
        sr = a['metrics']['Symbol rate\n(kHz)']
        dr = 1.0 / a['metrics']['Range res.\n(1/ΔR)']
        nm = a['name'].replace('\n', ' ')
        print(f'  {nm:<28}  {a["total"]:9.3f}  {sr:18.1f}  {dr:7.2f}')
    print(sep + '\n')


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Figure F17  —  Stacked power-breakdown bar chart
# ═══════════════════════════════════════════════════════════════════════════

def _save_fig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(cfg.OUTPUT_DIR, f'{name}.{ext}'),
                    dpi=(cfg.FIG_DPI if ext == 'png' else None))
    print(f'  Saved: figures/{name}.pdf | .png')


def plot_F17(archs: list, save=True):
    """F17 — Stacked horizontal bar chart (single-column 3.5 × 2.6 in)."""
    # Collect all power keys that appear with non-zero value in any arch
    all_keys = []
    for a in archs:
        for k in a['power']:
            if k not in all_keys:
                all_keys.append(k)
    used = [k for k in all_keys if any(a['power'].get(k, 0) > 0 for a in archs)]

    n_a = len(archs)
    cmap = plt.cm.Set2(np.linspace(0.05, 0.95, len(used)))

    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH_1COL, cfg.FIG_HEIGHT_TALL))
    lefts = np.zeros(n_a)

    for j, key in enumerate(used):
        vals = np.array([a['power'].get(key, 0.0) for a in archs])
        ax.barh(np.arange(n_a), vals, left=lefts, height=0.55,
                color=cmap[j], label=key,
                edgecolor='white', linewidth=0.4)
        lefts += vals

    # Total-power labels
    for i, a in enumerate(archs):
        ax.text(a['total'] + 0.05, i, f'{a["total"]:.2f} W',
                va='center', ha='left', fontsize=8)

    # Power-saving annotation (proposed vs AESA)
    P_prop = archs[0]['total'];  P_aesa = archs[3]['total']
    sav    = (P_aesa - P_prop) / P_aesa * 100
    

    labels = [a['name'].replace('\n', '\n') for a in archs]
    ax.set_yticks(np.arange(n_a));  ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Total power consumption (W)',fontsize=10)
    ax.set_xlim([0, max(a['total'] for a in archs) * 1.22])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.grid(True, axis='x', which='major', alpha=0.30)
    ax.legend(loc='upper right', fontsize=8, ncol=1,
              handlelength=1.2, handletextpad=0.4, borderpad=0.4)
    ax.invert_yaxis()
    fig.tight_layout(pad=0.4)

    if save:
        _save_fig(fig, 'F17_power_comparison')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Figure F18  —  Spider / radar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_F18(archs: list, norm: dict, save=True):
    """F18 — Polar radar chart, 5 metrics × 4 architectures (single-col 3.5 × 3.0)."""
    keys    = list(archs[0]['metrics'].keys())
    N_met   = len(keys)
    angles  = np.linspace(0, 2 * np.pi, N_met, endpoint=False)
    angles_plot = np.append(angles, angles[0])   # close polygon

    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH_1COL, 3.0),
                           subplot_kw=dict(polar=True))

    ls_cycle = ['-', '--', '-.', ':']
    for i_a, (a, ls) in enumerate(zip(archs, ls_cycle)):
        vals = np.array([norm[k][i_a] for k in keys])
        vp   = np.append(vals, vals[0])
        nm   = a['name'].replace('\n', ' ')
        ax.plot(angles_plot, vp, ls=ls, color=a['color'],
                lw=1.6, label=nm)
        ax.fill(angles_plot, vp, color=a['color'], alpha=0.07)
        ax.plot(angles, vals, 'o', color=a['color'], ms=4.5, zorder=5)

    # Spoke labels
    ax.set_xticks(angles)
    ax.set_xticklabels(keys, fontsize=8.0)

    # Radial grid
    ax.set_rgrids([0.25, 0.50, 0.75, 1.00],
                  labels=['0.25', '0.50', '0.75', '1.00'],
                  angle=340, fontsize=8)
    ax.set_ylim([0, 1.08]);  ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.grid(color=cfg.COLORS['gray'], alpha=0.30, lw=0.6)
    ax.set_rlabel_position(310)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              fontsize=8.0, handlelength=2.0, ncol=2,
              borderpad=0.4, columnspacing=0.8)

    if save:
        _save_fig(fig, 'F18_spider_chart')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Tables
# ═══════════════════════════════════════════════════════════════════════════

def export_table_T4(archs):
    """T4 — Two-stage vs single-stage quantitative comparison  [★ critical]."""
    ts  = archs[0];  ss = archs[1]
    c   = cfg.C_LIGHT;  M = cfg.M_TONES;  Df = cfg.DELTA_F

    hdr  = ['Metric', 'Two-stage T-RIS/R-RIS', 'Single-stage STM-RIS', 'Gain']
    rows = [
        ['Symbol rate',
         f'{cfg.F_SYMBOL/1e3:.1f} kHz (DDS-limited)',
         f'{cfg.F_SYMBOL_SINGLE_STAGE/1e3:.1f} kHz (SPI-limited)',
         f'{cfg.F_SYMBOL/cfg.F_SYMBOL_SINGLE_STAGE:.0f}×'],
        ['Range resolution $\\Delta R$',
         f'{c/(M*Df):.2f} m',
         f'{c/(M*Df):.2f} m',
         '1× (identical)'],
        ['Total power',
         f'{ts["total"]:.2f} W',
         f'{ss["total"]:.2f} W',
         f'{(ss["total"]-ts["total"])/ts["total"]*100:.0f}\\% lower'],
        ['T-RIS / R-RIS stages',
         '2 (T-RIS + R-RIS)',
         '1 (combined STM-RIS)',
         '—'],
        ['Modulation control',
         'T-RIS internal DDS',
         'SPI from base-band',
         'DDS: 55× faster'],
        ['R-RIS update rate',
         '1 × per CPI',
         'M × per symbol',
         '$N_{sw}$ fewer events'],
        ['Near-field coupling loss',
         f'$|C_0|^2$ = {cfg.C0_NOM_DB:.1f} dB',
         'N/A (single panel)',
         '—'],
        ['Post-CPI SNR',
         f'{cfg.SNR_CPI_DB:.1f} dB',
         f'$\\approx${cfg.SNR_CPI_DB-0.5:.1f} dB (est.)',
         '—'],
        ['Beam-modulation decoupling',
         'Full (independent stages)',
         'None (coupled)',
         '—'],
        ['Hardware complexity',
         'Moderate (two panels)',
         'Low (one panel)',
         '—'],
    ]
    tex = _tex_wide(
        'Two-Stage vs. Single-Stage Architecture — Quantitative Comparison',
        'tab:arch_comparison', hdr, rows,
        r'DDS clock: $f_{\rm DDS}$ = 714.3 kHz; SPI clock: 10 MHz; '
        r'$N = 256$ elements, 3-bit phase quantisation.')
    _write('T4_two_stage_vs_single', tex, _csv(hdr, rows))


def export_table_T5():
    """T5 — Prior RIS sensing literature comparison  [★ critical]."""
    hdr  = ['Reference', 'System', '$f_c$', '$N$', 'Waveform',
            r'$\Delta R$', 'Symbol rate', 'Power']
    rows = [
        [r'\textbf{This work}',
         r'\textbf{Two-stage T-RIS/R-RIS}',
         r'\textbf{28 GHz}', r'\textbf{256}', r'\textbf{M-FSK (12 tones)}',
         r'\textbf{1.0 m}', r'\textbf{714 kHz}', r'\textbf{2.33 W}'],
        ['Shen \textit{et al.} [12]', 'STM-RIS single-stage',
         '28 GHz', '256', 'M-FSK', '1.0 m', '13 kHz', '2.95 W'],
        ['Buzzi \textit{et al.} [8]', 'RIS-assisted OFDM',
         '5.9 GHz', '64', 'OFDM', '0.3 m', '—', '—'],
        ['Liu \textit{et al.} [9]', 'Passive RIS radar',
         '77 GHz', '128', 'FMCW chirp', '0.15 m', '—', '$>5$ W'],
        ['Aubry \textit{et al.} [10]', 'Active IRS sensing',
         '28 GHz', '64', 'CW probing', '—', '—', '$>8$ W'],
        ['Zhang \textit{et al.} [11]', 'RIS ISAC',
         '24 GHz', '32', 'OFDM', '0.5 m', '—', '—'],
        ['Tang \textit{et al.} [13]', 'Bistatic RIS radar',
         '3.5 GHz', '256', 'FMCW', '0.5 m', '—', '—'],
    ]
    tex = _tex_wide(
        r'Comparison with Prior RIS Sensing Literature',
        'tab:literature', hdr, rows,
        r'$\Delta R$ = range resolution; ``—'' = not reported in source; '
        r'Power = total system power consumption.')
    _write('T5_literature_comparison', tex, _csv(hdr, rows))


def export_table_T6(archs):
    """T6 — Computational complexity comparison."""
    M = cfg.M_TONES;  Nsw = cfg.NSw;  N = cfg.N_ELEM_X

    hdr  = ['Processing step', 'Two-stage RIS', 'FMCW radar', 'AESA radar']
    rows = [
        ['Range processing',
         f'IFFT-$M$ per CPI ($M={M}$)',
         r'2-D FFT per burst',
         r'1-D FFT per CPI'],
        ['Doppler processing',
         f'FFT-$N_{{sw}}$ per range bin ($N_{{sw}}={Nsw}$)',
         f'FFT-$N_{{sw}}$ per range bin',
         f'FFT-$N_{{sw}}$ per beam'],
        ['Angle estimation',
         f'{N}-element beamscan / MUSIC',
         f'{N}-element beamscan',
         f'{N}²-element full steering'],
        ['Range–Doppler coupling',
         r'None (decoupled by design)',
         r'Requires joint 2-D proc.',
         r'None'],
        ['Beamforming update',
         r'1 × per CPI  (R-RIS only)',
         r'N/A',
         r'1 × per dwell'],
        ['Phase reprogramming',
         r'T-RIS: DDS (per symbol);  R-RIS: 1×/CPI',
         r'N/A',
         r'$N$ complex weights per dwell'],
        ['Range profile complexity',
         r'$\mathcal{O}(M \log M)$',
         r'$\mathcal{O}(N_s \log N_s)$',
         r'$\mathcal{O}(N \log N)$'],
    ]
    tex = _tex_wide(
        'Computational Complexity Comparison',
        'tab:complexity', hdr, rows,
        r'$N_s$ = number of ADC samples per FMCW chirp.')
    _write('T6_complexity_comparison', tex, _csv(hdr, rows))


def export_table_T8(archs):
    """T8 — Detailed power budget for the proposed two-stage RIS system."""
    ts    = archs[0]
    total = ts['total']

    _desc = {
        'PA input':       r'Power amplifier ($P_{\rm Tx}=100$ mW, $\eta=20\%$)',
        'T-RIS bias':     r'T-RIS PIN-diode DC bias (256 elem × 2.0 mW)',
        'R-RIS bias':     r'R-RIS PIN-diode DC bias (256 elem × 2.0 mW)',
        'DDS switching':  r'Dynamic: $N E_{\rm sw} M f_{\rm sym}$',
        'Rx chain / LNA': r'Rx low-noise amplifier + down-converter',
        'DSP + ADC':      r'Digital processing + analogue-to-digital chain',
        'Controller':     r'DDS controller + reference clock',
    }

    hdr  = ['Component', 'Description', 'Power (W)', r'Fraction (\%)']
    rows = []
    for comp, pwr in ts['power'].items():
        if pwr > 0:
            rows.append([comp, _desc.get(comp, '—'),
                         f'{pwr:.4f}', f'{pwr/total*100:.1f}'])
    rows.append([r'\midrule \textbf{Total}',
                 r'\textbf{Two-stage RIS system}',
                 f'\\textbf{{{total:.3f}}}', r'\textbf{100.0}'])
    rows.append(['AESA reference (config)',
                 f'$N$={cfg.N_ELEM} active T/R modules',
                 f'{cfg.P_AESA_TOTAL:.2f}',
                 f'{cfg.P_AESA_TOTAL/total*100:.0f}\\% of AESA'])

    tex = _tex_wide(
        r'Detailed Power Budget — Two-Stage T-RIS/R-RIS System',
        'tab:power_budget', hdr, rows,
        r'$E_{\rm sw}$ = 10 pJ/event; PIN bias = 2.0 mW/element; '
        r'PA efficiency $\eta$ = 20\% at 28 GHz.')
    _write('T8_power_budget', tex, _csv(hdr, rows))


# ── LaTeX / CSV helpers ───────────────────────────────────────────────────

def _tex_wide(caption, label, hdr, rows, notes=''):
    """Full-width table* environment (4-column)."""
    nc    = len(hdr)
    cfmt  = 'l ' + ' '.join(['l'] * (nc - 1))
    lines = [r'\begin{table*}[!t]',
             r'\renewcommand{\arraystretch}{1.15}',
             r'\caption{' + caption + '}',
             r'\label{' + label + '}',
             r'\centering',
             r'\begin{tabular}{' + cfmt + '}',
             r'\hline\hline',
             ' & '.join(hdr) + r' \\', r'\hline']
    for r in rows:
        lines.append(' & '.join(str(c) for c in r) + r' \\')
    lines += [r'\hline\hline']
    if notes:
        lines += [r'\multicolumn{' + str(nc) + r'}{l}{\footnotesize '
                  + notes + r'} \\', r'\hline']
    lines += [r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


def _csv(hdr, rows):
    return '\n'.join([','.join(str(h) for h in hdr)] +
                     [','.join(str(c) for c in r) for r in rows])


def _write(name, tex, csv):
    for ext, content in [('.tex', tex), ('.csv', csv)]:
        with open(os.path.join(cfg.TABLE_DIR, name + ext), 'w') as f:
            f.write(content)
    print(f'  Exported: tables/{name}.tex + .csv')


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Data export  +  main entry point
# ═══════════════════════════════════════════════════════════════════════════

def export_data(archs, norm):
    p = os.path.join(cfg.DATA_DIR, 'benchmarking_data.npy')
    np.save(p, {'archs': archs, 'norm': norm}, allow_pickle=True)
    print('  Exported: data/benchmarking_data.npy')


def run(show: bool = False):
    """Run Stage 6b — System Benchmarking.  Returns (archs, norm_metrics)."""
    print('\n' + '═' * 60)
    print('  Stage 6b — System Benchmarking')
    print('  Generating Figures F17, F18 | Tables T4, T5, T6, T8')
    print('═' * 60)

    archs = define_architectures()
    norm  = normalise_metrics(archs)
    _print_summary(archs)

    plot_F17(archs, save=True)
    plot_F18(archs, norm, save=True)
    export_table_T4(archs)
    export_table_T5()
    export_table_T6(archs)
    export_table_T8(archs)
    export_data(archs, norm)

    if show:
        plt.show()
    plt.close('all')
    print('\n   complete — 2 figures + 4 tables + data saved.')
    return archs, norm


if __name__ == '__main__':
    run()
