"""
exp07_benchmarks.py -- Architecture, waveform and power benchmarks
==================================================================
Control-plane and modulation-drive comparison with single-stage designs;
three-target M-FSK versus FMCW at equal bandwidth, CPI and energy;
transmit-side power with identical accounting boundaries.

Paper figures produced: Fig. 12 (F11rev_fmcw_fair), Fig. 13 (F12rev_power)
Outputs: results/exp07_benchmarks/ (data, tables, extra figures), results/figdata/ (figure data),
         paper/figures/ (paper figures)
Run    : python experiments/exp07_benchmarks.py
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
from figures.fig12_fmcw_fair import fig_fmcw
from figures.fig13_power import fig_power

logging.getLogger('fontTools').setLevel(logging.ERROR)

# ---- submitted power-model values (benchmarking.py / config.py) --------------------
PA_EFF = 0.20
P_BIAS_SUB = 2.0e-3            # W per element per layer (benchmarking.py)
P_CTRL = 0.150                 # controller / clock
P_LNA, P_DSP = 0.080, 0.550    # common bistatic receiver (excluded from the fair comparison)
P_LO_AESA = 0.87               # AESA master LO + main amplifier
P_PS_AESA = 3.5e-3             # AESA analog phase shifter per element (submitted F17 bar)
P_DSP_AESA, P_CTRL_AESA = 0.80, 0.150
CJ_48, V_48 = 30e-15, 0.8      # Eq. (48) values
ESW_SUB = 10e-12               # value stated in the paper


# =============================================================================
# A. control plane and modulation drive
# =============================================================================
def control_plane(n=mc.N_ELEM, bits=mc.PHASE_BITS, f_spi=10e6, k_par=16, f_par=100e6):
    fmax = mc.tone_offsets().max()
    t_spi = n * bits / f_spi
    t_par = n * bits / (k_par * f_par)
    L = 2 ** bits
    return dict(
        t_spi_us=t_spi * 1e6, rate_spi_khz=1 / t_spi / 1e3,
        t_par_us=t_par * 1e6, rate_par_khz=1 / t_par / 1e3, k_par=k_par, f_par_mhz=f_par / 1e6,
        rate_two_stage_khz=1 / mc.TSYM / 1e3,
        stc_timing_res_ns=1 / (L * fmax) * 1e9, stc_clock_ghz=L * fmax / 1e9,
        stc_aggregate_gbps=n * L * fmax / 1e9,
        rris_update_kbps=n * bits / mc.TCPI / 1e3, fmax_mhz=fmax / 1e6)


# =============================================================================
# B. FMCW fairness
# =============================================================================
TARGETS_L = np.array([4.0, 7.0, 10.0])
TARGETS_LD = np.array([0.0, 3.0, -2.0])


def peaks_2d(P, n, excl):
    """n strongest peaks, suppressing a (+/-excl[0], +/-excl[1]) neighbourhood (wrapped)."""
    P = P.copy()
    out = []
    for _ in range(n):
        i, j = np.unravel_index(np.argmax(P), P.shape)
        out.append((i, j))
        ii = np.arange(i - excl[0], i + excl[0] + 1) % P.shape[0]
        jj = np.arange(j - excl[1], j + excl[1] + 1) % P.shape[1]
        P[np.ix_(ii, jj)] = 0
    return out


def sim_mfsk(rng, snr_total):
    m, n = mc.M_TONES, mc.NSW
    nu = mc.tone_offsets()
    t = np.arange(n)[None, :] * mc.TSW + np.arange(m)[:, None] * mc.TSYM
    snr = snr_total / (m * n)
    Y = np.zeros((m, n), complex)
    for L, Ld in zip(TARGETS_L, TARGETS_LD):
        Y += np.sqrt(snr) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.exp(
            -1j * 2 * np.pi * (nu[:, None] * L + (mc.FC + nu[:, None]) * Ld * t) / mc.C_LIGHT)
    Y += (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / np.sqrt(2)
    nd, nr = 2 * n, 8 * m
    D = np.fft.fft(Y, n=nd, axis=1)
    R = np.fft.ifft(D, n=nr, axis=0)
    P = np.abs(R) ** 2
    Laxis = np.arange(nr) * mc.C_LIGHT / (nr * mc.DELTA_F)
    fd = np.fft.fftfreq(nd, mc.TSW)
    pk = peaks_2d(P, 3, (4, 8))
    det = sorted([(Laxis[i], -fd[j] * mc.LAMBDA) for i, j in pk])
    prof = P.max(axis=1)
    ops = m * nd / 2 * np.log2(nd) + nd * nr / 2 * np.log2(nr)
    return dict(Laxis=Laxis, prof=prof / prof.max(), det=det, ops=ops)


def sim_fmcw_single(rng, snr_total, B=mc.M_TONES * mc.DELTA_F, T=mc.TCPI, fs=4e3):
    """One CPI-long chirp (the submitted comparison)."""
    ns = int(round(fs * T))
    tt = np.arange(ns) / fs
    slope = B / T
    snr = snr_total / ns
    s = np.zeros(ns, complex)
    for L, Ld in zip(TARGETS_L, TARGETS_LD):
        fb = slope * L / mc.C_LIGHT + Ld / mc.LAMBDA            # beat + Doppler (receding raises it)
        s += np.sqrt(snr) * np.exp(1j * (2 * np.pi * fb * tt + rng.uniform(0, 2 * np.pi)))
    s += (rng.standard_normal(ns) + 1j * rng.standard_normal(ns)) / np.sqrt(2)
    nf = 16 * ns
    S = np.abs(np.fft.fft(s, nf)) ** 2
    f = np.fft.fftfreq(nf, 1 / fs)
    Laxis = f * mc.C_LIGHT / slope
    keep = (Laxis >= 0) & (Laxis <= 16)
    P = S.copy()
    P[~keep] = 0
    det = []
    for _ in range(3):
        i = int(np.argmax(P))
        det.append(Laxis[i])
        P[max(0, i - 40):i + 41] = 0
    order = np.argsort(np.argsort(det))
    prof = S[keep]
    return dict(Laxis=Laxis[keep], prof=prof / prof.max(), det=sorted(det),
                predicted=list(TARGETS_L + TARGETS_LD * mc.FC * T / B))


def sim_fmcw_cs(rng, snr_total, B=mc.M_TONES * mc.DELTA_F, Tc=mc.TSW, nc=mc.NSW, fs=1.5e6):
    """Chirp-sequence FMCW, equal bandwidth, CPI and energy, 2-D FFT."""
    ns = int(round(fs * Tc))
    n_fast = np.arange(ns) / fs
    slope = B / Tc
    snr = snr_total / (ns * nc)
    S = np.zeros((ns, nc), complex)
    p = np.arange(nc)
    for L, Ld in zip(TARGETS_L, TARGETS_LD):
        tau = (L + Ld * p * Tc) / mc.C_LIGHT
        fb = slope * tau
        S += np.sqrt(snr) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.exp(
            1j * 2 * np.pi * (fb[None, :] * n_fast[:, None] - mc.FC * tau[None, :]))
    S += (rng.standard_normal((ns, nc)) + 1j * rng.standard_normal((ns, nc))) / np.sqrt(2)
    nr, nd = 8 * ns, 2 * nc
    R = np.fft.fft(S, n=nr, axis=0)
    D = np.fft.fft(R, n=nd, axis=1)
    P = np.abs(D) ** 2
    Laxis = np.fft.fftfreq(nr, 1 / fs) * mc.C_LIGHT / slope
    fd = np.fft.fftfreq(nd, Tc)
    valid = (Laxis >= 0) & (Laxis <= 16)
    P[~valid, :] = 0
    pk = peaks_2d(P, 3, (4, 8))
    det = sorted([(Laxis[i], -fd[j] * mc.LAMBDA) for i, j in pk])
    prof = P.max(axis=1)[valid]
    order = np.argsort(Laxis[valid])
    ops = nc * nr / 2 * np.log2(nr) + nr * nd / 2 * np.log2(nd)
    return dict(Laxis=Laxis[valid][order], prof=(prof / prof.max())[order], det=det, ops=ops, ns=ns)


# =============================================================================
# C. power
# =============================================================================
def ris_tx_power(n=mc.N_ELEM, p_bias_t=P_BIAS_SUB, p_bias_r=P_BIAS_SUB, e_sw=None, tau_sw=0.27e-9):
    """Transmit-side DC power.  Switching energy from stored charge,
    E_sw = Q_s V = P_bias * tau_sw, with tau_sw from the Phase-1 requirement."""
    fbar = mc.tone_offsets().mean()
    e_sw = p_bias_t * tau_sw if e_sw is None else e_sw
    return {'CW source (PA, 20 %)': 0.1 / PA_EFF, 'T-RIS PIN bias': n * p_bias_t, 'R-RIS bias': n * p_bias_r,
            'T-RIS switching': n * 2 * fbar * e_sw, 'Controller, DDS, clock': P_CTRL}


def aesa_tx_power(n=mc.N_ELEM, p_ch=5e-3):
    return {'T/R channels': n * p_ch, 'LO distribution, main amplifier': P_LO_AESA, 'Controller': P_CTRL_AESA}


# =============================================================================
# figures and tables
# =============================================================================




def table_architecture(out, cp, pw):
    rows = [
        ('Control writes per symbol', 'one DDS word', '$NB$ bits (SPI)', 'pointer (LUT)'),
        ('Symbol rate', f"{cp['rate_two_stage_khz']:.0f} kHz$^\\dagger$", f"{cp['rate_spi_khz']:.1f} kHz",
         f"{cp['rate_par_khz']/1e3:.1f} MHz$^\\ddagger$"),
        ('Elements switched at $f_m$', '$N$ (T-RIS only)', '$N$ (all)', '$N$ (all)'),
        ('Distinct modulation drives', '1 (common)', '$N$', '$N$ or 1 global line'),
        ('Drive timing resolution', 'none', f"{cp['stc_timing_res_ns']:.2f} ns", f"{cp['stc_timing_res_ns']:.2f} ns / none"),
        ('Beamformer technology', 'any (static per CPI)', 'fast switching', 'fast switching'),
        ('Beamformer update', f"{cp['rris_update_kbps']:.0f} kbit/s", 'per symbol', 'per symbol / per CPI'),
        ('Inter-stage coupling loss', '$C_0^2$ (Sec.~\\ref{sec:5b})', 'none', 'none'),
        ('Carrier feedthrough', '$|T_{\\rm mean}|$', 'suppressible', 'suppressed (0/$\\pi$)'),
    ]
    body = "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows)
    tex = r"""\begin{table*}[!t]
\centering
\caption{Two-Stage Versus Single-Stage Architectures for Steered M-FSK Generation}
\label{tab:arch_comparison}
\footnotesize
\begin{tabular}{@{}lccc@{}}
\hline\hline
 & Two-stage T-RIS/R-RIS & Single stage, per-symbol reload & Single stage, FPGA/LUT or global line \\
\hline
""" + body + r"""
\hline\hline
\multicolumn{4}{@{}p{0.98\textwidth}@{}}{\scriptsize $N=256$, $B=3$ bits, $f_{m,M}=300$\,MHz, SPI at 10\,MHz. $^\dagger$Set by $T_{\rm sym}=1.4\,\mu$s; the DDS itself switches in tens of ns. $^\ddagger$%d parallel chains at %d\,MHz. Per-element phase offsets imposed by time coding need a timing resolution of $1/(2^B f_{m,M})$; a global 0/$\pi$ line that toggles every element's phase state avoids it but still switches all beamforming elements at $f_m$.}
\end{tabular}
\end{table*}
""" % (cp['k_par'], cp['f_par_mhz'])
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'T5rev_architecture.tex').write_text(tex, encoding='utf-8')


def table_power(out, ris, aesa5, aesa15, common):
    def rows(d):
        return "\n".join(f"{k} & {v:.3f} \\\\" for k, v in d.items())
    tex = r"""\begin{table}[!t]
\centering
\caption{Transmit-Side DC Power Budget, Identical Accounting ($N=256$)}
\label{tab:power_budget}
\footnotesize
\begin{tabular}{@{}lr@{}}
\hline\hline
Item & W \\
\hline
\multicolumn{2}{@{}l}{\emph{Two-stage RIS (PIN, 2 mW/element/layer, $E_{\rm sw}=P_{\rm bias}\tau_{\rm sw}$)}}\\
""" + rows(ris) + r"""
\textbf{Total} & \textbf{%.3f} \\
\hline
\multicolumn{2}{@{}l}{\emph{AESA, 5\,mW / 15\,mW per channel}}\\
""" % sum(ris.values()) + rows(aesa5) + r"""
\textbf{Total (5 / 15 mW)} & \textbf{%.3f / %.3f} \\
\hline
Common bistatic receiver (excluded) & %.3f \\
\hline\hline
\end{tabular}
\end{table}
""" % (sum(aesa5.values()), sum(aesa15.values()), common)
    (Path(out) / 'TS_power_budget.tex').write_text(tex, encoding='utf-8')


# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description='fair benchmarks')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'exp07_benchmarks'))
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
    log = []

    def say(s=''):
        print(s, flush=True)
        log.append(s)

    say('=' * 74)
    say(' Fair benchmarks')
    say('=' * 74)

    # ---- A ---------------------------------------------------------------------------
    cp = control_plane()
    say('\n[A] Control plane and modulation drive (N = 256, 3-bit, f_m,max = 300 MHz)')
    say(f"    single stage, SPI reload      : {cp['t_spi_us']:.1f} us/symbol -> {cp['rate_spi_khz']:.1f} kHz (submitted)")
    say(f"    single stage, {cp['k_par']} parallel chains @ {cp['f_par_mhz']:.0f} MHz: {cp['t_par_us']:.2f} us -> {cp['rate_par_khz']/1e3:.2f} MHz")
    say(f"    two stage, T_sym = 1.4 us     : {cp['rate_two_stage_khz']:.1f} kHz")
    say(f"    per-element phase via time coding: timing resolution {cp['stc_timing_res_ns']:.2f} ns "
        f"({cp['stc_clock_ghz']:.1f} GHz clock), aggregate {cp['stc_aggregate_gbps']:.0f} Gbit/s")
    say(f"    two-stage beamformer update   : {cp['rris_update_kbps']:.0f} kbit/s (once per CPI)")
    say('    -> the 55x claim holds only against SPI reloading; the invariant is that a single stage switches')
    say('       every beamforming element at f_m, whereas the two stage confines this to the T-RIS.')

    # ---- B ---------------------------------------------------------------------------
    say('\n[B] Three-target scene at equal bandwidth, CPI and energy')
    snr_total = 10 ** (mc.SNR1_DB / 10) * mc.M_TONES * mc.NSW
    rng = np.random.default_rng(mc.RANDOM_SEED)
    a = sim_mfsk(rng, snr_total)
    b = sim_fmcw_single(rng, snr_total)
    c = sim_fmcw_cs(rng, snr_total)
    say('    true (path, rate)          : ' + ', '.join(f'({L:.0f} m, {v:+.0f} m/s)' for L, v in zip(TARGETS_L, TARGETS_LD)))
    say('    M-FSK 2-D                  : ' + ', '.join(f'({L:.2f}, {v:+.2f})' for L, v in a['det']))
    say('    FMCW one CPI-long chirp    : ' + ', '.join(f'{L:.2f}' for L in b['det'])
        + '   predicted ' + ', '.join(f'{L:.2f}' for L in sorted(b['predicted'])))
    say('    chirp-sequence FMCW 2-D    : ' + ', '.join(f'({L:.2f}, {v:+.2f})' for L, v in c['det']))
    say(f"    FFT operations per CPI     : M-FSK {a['ops']:.3g}, chirp-sequence FMCW {c['ops']:.3g} "
        f"(ratio {c['ops']/a['ops']:.2f}); FMCW fast-time samples per chirp {c['ns']}")
    err_a = [max(abs(d[0] - L) / 0.125, abs(d[1] - v) / (mc.LAMBDA / mc.TCPI)) for d, L, v in zip(a['det'], TARGETS_L, TARGETS_LD)]
    err_c = [max(abs(d[0] - L) / 0.125, abs(d[1] - v) / (mc.LAMBDA / mc.TCPI)) for d, L, v in zip(c['det'], TARGETS_L, TARGETS_LD)]
    err_a, err_c = max(err_a), max(err_c)      # in units of (padded path bin, unpadded path-rate bin)
    ghost = [dd - L for dd, L in zip(sorted(b['det']), sorted(b['predicted']))]

    # ---- C ---------------------------------------------------------------------------
    say('\n[C] Power, identical accounting boundaries')
    n = mc.N_ELEM
    sub_two = 0.1 / PA_EFF + 2 * n * P_BIAS_SUB + n * ESW_SUB * mc.M_TONES * (1 / mc.TSYM) + P_LNA + P_DSP + P_CTRL
    sub_aesa_bar = P_LO_AESA + P_DSP_AESA + P_CTRL_AESA + n * P_PS_AESA + n * 15e-3
    sub_aesa_txt = n * 15e-3 + P_LO_AESA
    say(f'    submitted: two-stage {sub_two:.3f} W (includes receiver {P_LNA+P_DSP:.2f} W); AESA text {sub_aesa_txt:.2f} W '
        f'(excludes receiver), AESA bar {sub_aesa_bar:.2f} W (different DSP/controller, extra phase shifters)')
    esw48 = 0.5 * CJ_48 * V_48 ** 2
    say(f'    Eq.(48): 1/2 C_J V^2 = {esw48*1e15:.1f} fJ (paper states 10 pJ); stored-charge estimate '
        f'P_bias*tau_sw = {P_BIAS_SUB*0.27e-9*1e12:.2f} pJ')
    ris = ris_tx_power()
    ris_volt = ris_tx_power(p_bias_r=0.0)
    ris_pess = ris_tx_power(e_sw=ESW_SUB)
    aesa5, aesa15, aesa2 = aesa_tx_power(p_ch=5e-3), aesa_tx_power(p_ch=15e-3), aesa_tx_power(p_ch=2e-3)
    P_r, P_rv, P_rp = sum(ris.values()), sum(ris_volt.values()), sum(ris_pess.values())
    fixed_aesa = P_LO_AESA + P_CTRL_AESA
    be = lambda P: (P - fixed_aesa) / n
    for k, v in ris.items():
        say(f'      RIS  {k:32s} {v:6.3f} W')
    say(f'      RIS total {P_r:.3f} W  | voltage-driven R-RIS {P_rv:.3f} W | pessimistic E_sw=10 pJ {P_rp:.3f} W')
    for lab, d in (('2 mW', aesa2), ('5 mW', aesa5), ('15 mW', aesa15)):
        tot = sum(d.values())
        say(f'      AESA @ {lab:5s}: {tot:.3f} W  -> RIS {100*(1-P_r/tot):+.0f}%  (voltage-driven R-RIS {100*(1-P_rv/tot):+.0f}%)')
    say(f'    break-even P_ch: PIN {be(P_r)*1e3:.2f} mW, voltage-driven {be(P_rv)*1e3:.2f} mW, pessimistic {be(P_rp)*1e3:.2f} mW')
    common = P_LNA + P_DSP
    say(f'    full system incl. common receiver ({common:.2f} W): RIS {P_r+common:.2f} W vs AESA(5 mW) '
        f'{sum(aesa5.values())+common:.2f} W -> {100*(1-(P_r+common)/(sum(aesa5.values())+common)):.0f}% lower')
    say('    per-element costs scale identically with N (2 P_bias vs P_ch): there is no crossover in N.')

    pch = np.linspace(1e-3, 20e-3, 200)
    sweep = dict(p_ch=pch, aesa=n * pch + fixed_aesa, ris_pin=P_r, ris_volt=P_rv, ris_pess=P_rp, be_pin=be(P_r))
    pb = np.linspace(0.2e-3, 5e-3, 120)
    pc = np.linspace(1e-3, 20e-3, 120)
    PB, PC = np.meshgrid(pb, pc)
    ratio = (n * PC + fixed_aesa) / np.vectorize(lambda x: sum(ris_tx_power(p_bias_t=x, p_bias_r=x).values()))(PB)
    paths = [fig_fmcw(fdir, a, b, c), fig_power(fdir, sweep, None, dict(p_bias=PB, p_ch=PC, ratio=ratio))]
    table_architecture(tdir, cp, None)
    table_power(tdir, ris, aesa5, aesa15, common)
    say('\n[out]')
    for p in paths:
        say(f'    {p}')
    say(f"    {tdir / 'T5rev_architecture.tex'}\n    {tdir / 'TS_power_budget.tex'}")

    say('\n[checks] Validation')
    checks = [
        ('SPI baseline reproduces the submitted 13.0 kHz', abs(cp['rate_spi_khz'] - 13.02) < 0.05),
        ('submitted power figures reproduced (2.33 W, 4.71 W, 6.56 W)',
         abs(sub_two - 2.326) < 0.005 and abs(sub_aesa_txt - 4.71) < 0.005 and abs(sub_aesa_bar - 6.556) < 0.005),
        ('one CPI-long FMCW chirp puts the ghosts where the coupling model predicts (< 1/4 resolution cell)',
         max(abs(g) for g in ghost) < 0.25 * mc.path_resolution()),
        ('M-FSK 2-D resolves all three targets (within one bin in path and path rate)', err_a <= 1.0),
        ('chirp-sequence FMCW 2-D resolves all three targets equally, no ghosts', err_c <= 1.0),
        ('break-even consistent: totals equal at P_ch* (< 1e-9 W)', abs(n * be(P_r) + fixed_aesa - P_r) < 1e-9),
    ]
    for n_, ok in checks:
        say(f"    [{'PASS' if ok else 'FAIL'}] {n_}")
    say(f'    time {time.time()-t0:.0f} s')
    res = dict(control=cp, fmcw=dict(mfsk=a['det'], single=b['det'], single_pred=b['predicted'], cs=c['det'],
                                     ops_mfsk=a['ops'], ops_cs=c['ops'], ns=c['ns'], err_mfsk=err_a, err_cs=err_c),
               power=dict(sub_two=sub_two, sub_aesa_txt=sub_aesa_txt, sub_aesa_bar=sub_aesa_bar, esw48_fJ=esw48 * 1e15,
                          esw_est_pJ=P_BIAS_SUB * 0.27e-9 * 1e12, ris=ris, ris_total=P_r, ris_volt_total=P_rv,
                          ris_pess_total=P_rp, aesa2=sum(aesa2.values()), aesa5=sum(aesa5.values()),
                          aesa15=sum(aesa15.values()), be_pin_mW=be(P_r) * 1e3, be_volt_mW=be(P_rv) * 1e3,
                          be_pess_mW=be(P_rp) * 1e3, common=common),
               gate7={n_: bool(o) for n_, o in checks})
    (ddir / 'exp07_results.json').write_text(json.dumps(res, indent=2, default=float), encoding='utf-8')
    (ddir / 'exp07_summary.txt').write_text('\n'.join(log), encoding='utf-8')
    return 0 if all(o for _, o in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
