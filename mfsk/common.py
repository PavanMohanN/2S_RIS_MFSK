"""
mfsk_common.py -- Shared physics for the TVT revision (VT-2026-04842)
=====================================================================
Single source of truth for the Phase 1 and Phase 2 revision scripts.

Every numerical value here is copied from the submitted `config.py` unless a
comment marked [REV] says otherwise.  Every CRB is implemented exactly as the
corresponding (corrected) manuscript equation, so that equations, code and
tables agree -- which the submitted version did not (see PHASE1_2_GATE_REPORT).

[REV] changes relative to the submitted code
--------------------------------------------
1. crb_path()     : implements Eq. (37) exactly.  The submitted code used
                    (c/2)/sqrt(SNR*(2pi)^2*var_f), which omits the factor 2 of
                    the complex-Gaussian FIM and applies a monostatic c/2 to a
                    one-way delay model -> it was sqrt(2) too optimistic.
2. crb_doppler()  : exact FIM over all M*N sample instants.  Equivalent to
                    3 / (2 pi^2 M T_CPI^2 SNR_CPI); the submitted Eq. (39)
                    coincides numerically only because M = 12.
3. crb_angle_ula(): exact ULA bound (Stoica-Nehorai form).  The submitted
                    Eq. (33) and the submitted code disagree with each other
                    and with the exact bound.
4. Position bound : propagated through the manuscript's own bistatic inversion
                    Eqs. (40)-(42) via its Jacobian.  The submitted Eq. (44)
                    ignored the bistatic geometry dilution.
5. Rx placement   : parameterised by the bistatic angle beta at the target,
                    with the Rx kept R3 = 8 m from the target (Table I), so the
                    link budget is unchanged.

Python >= 3.9, numpy >= 1.21.  No other dependency.
"""
from __future__ import annotations

import numpy as np

# =============================================================================
# 1. Constants  (identical to config.py)
# =============================================================================
C_LIGHT = 2.998e8                       # [m/s]  config.C_LIGHT
FC = 28.0e9                             # [Hz]
LAMBDA = C_LIGHT / FC                   # [m]   ~10.71 mm
K_C = 2.0 * np.pi / LAMBDA

# PIN-diode equivalent circuit, Eq. (1)-(2)
RS, CJ, RJ_ON, RJ_OFF, Z0 = 2.0, 25.0e-15, 1.0, 8.0e3, 50.0

# RIS (each stage)
N_X = N_Y = 16
N_ELEM = N_X * N_Y
D_ELEM = LAMBDA / 2.0
PHASE_BITS = 3
GAMMA_MAG = 10.0 ** (-1.0 / 20.0)       # |Gamma| = -1 dB
SIGMA_AMP_FRAC_NOM = 0.02               # config.SIGMA_AMP_FRAC
SIGMA_PHASE_DEG_NOM = 3.0               # config.SIGMA_PHASE_RAD (deg)

# M-FSK waveform
M_TONES = 12
DELTA_F = 25.0e6
TSYM = 1.40e-6
NSW = 1024
TSW = M_TONES * TSYM                    # 16.8 us
TCPI = NSW * TSW                        # 17.2 ms
F_OFF_LEGACY = 0.0                      # tones at k*DELTA_F, k = 1..M

# Operating point
SNR1_DB = -14.5                         # single-symbol SNR (Table I)
SNR_CPI_DB = SNR1_DB + 10.0 * np.log10(NSW)   # +15.6 dB

# Rx array
NRX = 16
D_RX = LAMBDA / 2.0

# Geometry (Table I distances; angle convention of Eq. (40): u = [sin, cos])
R1_TX_RIS = 8.0
R2_RIS_T = 10.0
R3_T_RX = 8.0
THETA1_DEG = 60.0
P_RIS = np.zeros(2)
P_TX = np.array([0.0, -R1_TX_RIS])
P_TARGET = R2_RIS_T * np.array([np.sin(np.deg2rad(THETA1_DEG)),
                                np.cos(np.deg2rad(THETA1_DEG))])
P_RX_LEGACY = np.array([10.46, 7.54])   # submitted config.P_RX (3.1 m from target)
BETA_LEGACY_DEG = 155.3                 # bistatic angle implied by P_RX_LEGACY
BETA_DEFAULT_DEG = 60.0                 # [REV] recommended (gate decision D2)

# Target kinematics
V_TARGET = 2.0
HEADING_DEG = 45.0
V_VEC = V_TARGET * np.array([np.cos(np.deg2rad(HEADING_DEG)),
                             np.sin(np.deg2rad(HEADING_DEG))])

RANDOM_SEED = 42


# =============================================================================
# 2. Tone plan
# =============================================================================
def tone_offsets(f_off: float = F_OFF_LEGACY, m: int = M_TONES,
                 df: float = DELTA_F) -> np.ndarray:
    """Baseband tone offsets nu_k = f_off + k*df, k = 1..M   [revised Eq. (3)]."""
    return f_off + np.arange(1, m + 1, dtype=float) * df


def path_ambiguity(df: float = DELTA_F) -> float:
    """Unambiguous total-path interval of the stepped-frequency IFFT: c/df.
    (The submitted text states c/(2 M df), which is a resolution, not the
    ambiguity interval.)"""
    return C_LIGHT / df


def path_resolution(m: int = M_TONES, df: float = DELTA_F) -> float:
    """Total-path resolution c/(M df) = 1.00 m (Rayleigh)."""
    return C_LIGHT / (m * df)


# =============================================================================
# 3. PIN-diode T-RIS element, Eqs. (1), (2), (5), (8)
# =============================================================================
def z1(f, state: str):
    rj = RJ_ON if state == 'on' else RJ_OFF
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return RS + rj / (1.0 + 1j * w * CJ * rj)


def t_n(f, state: str):
    return 2.0 * Z0 / (2.0 * Z0 + z1(f, state))


def t_sb(f):
    return (t_n(f, 'on') - t_n(f, 'off')) / 2.0


def t_mean(f):
    return (t_n(f, 'on') + t_n(f, 'off')) / 2.0


def db20(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))


def db10(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


# =============================================================================
# 4. Geometry
# =============================================================================
def rx_position(beta_deg: float = BETA_DEFAULT_DEG, r3: float = R3_T_RX,
                p_target: np.ndarray = P_TARGET) -> np.ndarray:
    """Rx placed r3 from the target with bistatic angle beta (angle at the
    target between target->RIS and target->Rx).  The rotation sense matches
    the submitted P_RX (beta = 155.3 deg reproduces its direction)."""
    u = (P_RIS - p_target) / np.linalg.norm(P_RIS - p_target)
    b = np.deg2rad(-beta_deg)
    rot = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
    return p_target + r3 * (rot @ u)


def bearing(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """Global bearing (rad) with the Eq. (40) convention u = [sin, cos]."""
    d = p_to - p_from
    return float(np.arctan2(d[0], d[1]))


def bistatic_path(p_target: np.ndarray, p_rx: np.ndarray) -> float:
    """L = |RIS-target| + |target-Rx|  (the Tx-RIS leg is known and removed)."""
    return float(np.linalg.norm(p_target - P_RIS) + np.linalg.norm(p_target - p_rx))


def bistatic_path_rate(p_target, p_rx, v=V_VEC) -> float:
    """dL/dt.  u_rt points RIS->target, u_rx,t points target->Rx (Eqs. 25-26).
    dL/dt = u_rt.v - u_rx,t.v   ([REV] sign of the 2nd term; Eq. (27) has +)."""
    u_rt = (p_target - P_RIS) / np.linalg.norm(p_target - P_RIS)
    u_rxt = (p_rx - p_target) / np.linalg.norm(p_rx - p_target)
    return float(u_rt @ v - u_rxt @ v)


def invert_position(L, theta, p_rx):
    """Eqs. (40)-(42): Rx line of bearing intersected with the bistatic ellipse
    (foci RIS and Rx, path L).  Vectorised over L and theta."""
    L = np.asarray(L, dtype=float)
    theta = np.asarray(theta, dtype=float)
    a = p_rx - P_RIS
    ux, uy = np.sin(theta), np.cos(theta)
    a_dot_u = a[0] * ux + a[1] * uy
    r = (L ** 2 - a @ a) / (2.0 * (L + a_dot_u))
    return np.stack([p_rx[0] + r * ux, p_rx[1] + r * uy], axis=-1)


def position_jacobian(p_target, p_rx, eps: float = 1e-7) -> np.ndarray:
    """2x2 Jacobian d(x,y)/d(L,theta) of the inversion at the true state."""
    L0 = bistatic_path(p_target, p_rx)
    th0 = bearing(p_rx, p_target)
    dL = (invert_position(L0 + eps, th0, p_rx) - invert_position(L0 - eps, th0, p_rx)) / (2 * eps)
    dT = (invert_position(L0, th0 + eps, p_rx) - invert_position(L0, th0 - eps, p_rx)) / (2 * eps)
    return np.column_stack([dL, dT])


# =============================================================================
# 5. Cramer-Rao bounds (corrected; each maps to a manuscript equation)
# =============================================================================
def snr_cpi_lin(snr1_db, nsw: int = NSW):
    return 10.0 ** (np.asarray(snr1_db, dtype=float) / 10.0) * nsw


def crb_path(snr1_db, nsw: int = NSW, nu=None, weights=None):
    """sqrt CRB of the total bistatic path L [m]  -- Eq. (37):
        CRB(L) = c^2 / (8 pi^2 sigma_f^2 M SNR_CPI).
    With per-tone power weights w_k the frequency moment is weighted.
    Doppler is a nuisance parameter; the range-Doppler cross-information from
    intra-sweep timing is < 1e-6 of the diagonal and is neglected."""
    nu = tone_offsets() if nu is None else np.asarray(nu, dtype=float)
    w = np.ones_like(nu) if weights is None else np.asarray(weights, dtype=float)
    s = snr_cpi_lin(snr1_db, nsw)
    mom = np.sum(w * nu ** 2) - np.sum(w * nu) ** 2 / np.sum(w)   # = M var_f if w = 1
    fim = 2.0 * s * (2.0 * np.pi) ** 2 * mom
    return C_LIGHT / np.sqrt(fim)


def crb_path_legacy(snr1_db, nsw: int = NSW, nu=None):
    """The submitted implementation (for reconciliation only)."""
    nu = tone_offsets() if nu is None else nu
    s = 10.0 ** (np.asarray(snr1_db) / 10.0) * len(nu) * nsw
    return (C_LIGHT / 2.0) / np.sqrt(s * (2 * np.pi) ** 2 * np.var(nu))


def crb_radial_velocity(snr1_db, nsw: int = NSW, m: int = M_TONES,
                        tsw: float = TSW, tsym: float = TSYM):
    """sqrt CRB of the equivalent radial velocity v = lambda f_D / 2 [m/s]
    (paper convention; equals half the bistatic path rate).  Exact FIM over
    the M*N sample instants t = p Tsw + k Tsym."""
    snr1 = 10.0 ** (np.asarray(snr1_db, dtype=float) / 10.0)
    t = (np.arange(nsw)[None, :] * tsw + np.arange(1, m + 1)[:, None] * tsym).ravel()
    fim = 2.0 * snr1 * (2.0 * np.pi) ** 2 * np.sum((t - t.mean()) ** 2)
    return (LAMBDA / 2.0) / np.sqrt(fim)


def crb_radial_velocity_legacy(snr1_db, nsw: int = NSW):
    s = 10.0 ** (np.asarray(snr1_db) / 10.0) * M_TONES * nsw
    n = np.arange(nsw) * TSW
    return (LAMBDA / 2.0) / np.sqrt(s * (2 * np.pi) ** 2 * np.var(n))


def crb_angle_ula(snr_bf_lin, nrx: int = NRX, phi_rad: float = 0.0,
                  d_over_lambda: float = 0.5):
    """sqrt CRB of the AoA [rad] for an N-element ULA, deterministic signal with
    unknown complex amplitude.  snr_bf_lin is the post-beamforming SNR of the
    CPI snapshot (per-element SNR = snr_bf / N):
        CRB = 6 / ( (2 pi d/lambda)^2 cos^2(phi) N(N^2-1)/N... )  (see code)."""
    snr_el = np.asarray(snr_bf_lin, dtype=float) / nrx
    sum_sq = nrx * (nrx ** 2 - 1) / 12.0
    fim = 2.0 * snr_el * (2 * np.pi * d_over_lambda) ** 2 * np.cos(phi_rad) ** 2 * sum_sq
    return 1.0 / np.sqrt(fim)


def snr_snapshot_lin(snr1_db, nsw: int = NSW, m: int = M_TONES):
    """Post-beamforming SNR of the Eq. (30) snapshot after phase-aligned
    combining of all M tones and N_sw sweeps.  SNR1 already contains the
    18 dBi Rx array gain (Table I)."""
    return 10.0 ** (np.asarray(snr1_db, dtype=float) / 10.0) * m * nsw


def crb_position(sig_L, sig_theta, p_target, p_rx):
    """sqrt trace of J diag(sig_L^2, sig_theta^2) J^T  -- [REV] Eq. (44)."""
    J = position_jacobian(p_target, p_rx)
    sL = np.atleast_1d(np.asarray(sig_L, dtype=float))
    sT = np.atleast_1d(np.asarray(sig_theta, dtype=float))
    out = np.sqrt((J[0, 0] ** 2 + J[1, 0] ** 2) * sL ** 2
                  + (J[0, 1] ** 2 + J[1, 1] ** 2) * sT ** 2)
    return out if out.size > 1 else float(out[0])


# =============================================================================
# 6. R-RIS array factor (tone-resolved, with hardware impairments)
# =============================================================================
def rris_element_x() -> np.ndarray:
    """x-coordinates of all 256 R-RIS elements (row-major, centred)."""
    xs = (np.arange(N_X) - (N_X - 1) / 2.0) * D_ELEM
    return np.tile(xs, N_Y)


def quantise_phase(psi, bits):
    if bits is None or bits <= 0:
        return np.asarray(psi, dtype=float)
    step = 2.0 * np.pi / 2 ** bits
    return np.round(np.asarray(psi) / step) * step


def rris_tone_gains(freqs_abs, theta_target_rad, theta_steer_rad,
                    bits=PHASE_BITS, sigma_amp=0.0, sigma_phase_rad=0.0,
                    rng=None):
    """Complex normalised R-RIS array factor at the target for each tone.
    Phases are designed at the carrier (Eq. 9) and optionally quantised
    (B bits) and perturbed by per-element amplitude/phase errors."""
    x = rris_element_x()
    psi = quantise_phase(K_C * x * np.sin(theta_steer_rad), bits)
    amp = np.ones_like(x)
    if rng is not None and (sigma_amp > 0 or sigma_phase_rad > 0):
        amp = amp + sigma_amp * rng.standard_normal(x.size)
        psi = psi + sigma_phase_rad * rng.standard_normal(x.size)
    f = np.atleast_1d(np.asarray(freqs_abs, dtype=float))
    k = 2.0 * np.pi * f[:, None] / C_LIGHT
    af = np.sum(amp[None, :] * np.exp(1j * (psi[None, :] - k * x[None, :]
                                             * np.sin(theta_target_rad))), axis=1)
    return af / x.size


def squint_exact_deg(f_m, theta_deg=THETA1_DEG, f_design=FC):
    """[REV-P3] Exact squint of a phase-only gradient designed at f_design and
    radiated at f_design + f_m:  arcsin(sin(theta1) f_design/(f_design+f_m)) - theta1.
    Small-angle form: -tan(theta1) f_m / f_design.  (The submitted Eq. (14),
    arcsin(sin(theta1) f_m/f_c), underestimates it by 1/cos(theta1) and has the
    wrong sign.)"""
    th = np.deg2rad(theta_deg)
    f = np.asarray(f_m, dtype=float)
    return np.degrees(np.arcsin(np.clip(np.sin(th) * f_design / (f_design + f), -1, 1)) - th)


def hpbw_deg(theta_deg=THETA1_DEG, n=N_X, d_over_lambda=0.5):
    """Half-power beamwidth of an n-element uniform line array steered to theta."""
    return np.degrees(0.886 / (n * d_over_lambda) / np.cos(np.deg2rad(theta_deg)))


def af_line(theta_eval_rad, freqs_abs, theta_steer_rad, n=N_X, f_design=FC, bits=None):
    """Normalised complex array factor of an n-element lambda/2 line with a
    phase-only gradient designed at f_design (optionally B-bit quantised).
    Returns shape (len(freqs), len(theta))."""
    x = (np.arange(n) - (n - 1) / 2.0) * D_ELEM
    k_d = 2 * np.pi * f_design / C_LIGHT
    psi = quantise_phase(k_d * x * np.sin(theta_steer_rad), bits)
    f = np.atleast_1d(np.asarray(freqs_abs, dtype=float))
    th = np.atleast_1d(np.asarray(theta_eval_rad, dtype=float))
    k = 2 * np.pi * f / C_LIGHT
    ph = psi[None, None, :] - k[:, None, None] * x[None, None, :] * np.sin(th)[None, :, None]
    return np.exp(1j * ph).sum(axis=2) / n


# =============================================================================
# 7. Plot style (IEEE TVT) and small utilities
# =============================================================================
IEEE_COL_W = 3.5      # in
IEEE_DBL_W = 7.16     # in

COLORS = {
    'crb': '#5b4a9e', 'legacy': '#8c8c8c', 'coarse': '#d95f02',
    'coherent': '#1b9e77', 'refined': '#1f5fa8', 'accent': '#c0392b',
    'grey': '#555555', 'green': '#2e7d32',
}


def set_ieee_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman No9 L',
                       'Nimbus Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'lines.markersize': 3.5,
        'grid.linewidth': 0.4, 'grid.alpha': 0.35, 'grid.linestyle': '--',
        'legend.framealpha': 0.9, 'legend.edgecolor': '0.7',
        'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def save_fig(fig, out_dir, name):
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.pdf')
    fig.savefig(out / f'{name}.png')
    return out / f'{name}.pdf'


def find_legacy_dir(script_file: str | None = None):
    """Return the folder holding the submitted config.py + estimation_crb.py
    (cwd, the script's folder, or its parent), or None."""
    from pathlib import Path
    cands = [Path.cwd()]
    if script_file:
        here = Path(script_file).resolve().parent
        cands += [here, here.parent]
    for c in cands:
        if (c / 'config.py').exists() and (c / 'estimation_crb.py').exists():
            return c
    return None


class _InDir:
    """Context manager: chdir + sys.path insert, so legacy modules that create
    relative folders on import do so inside the repository root."""
    def __init__(self, d):
        self.d = d

    def __enter__(self):
        import os
        import sys
        self.old = os.getcwd()
        os.chdir(self.d)
        sys.path.insert(0, str(self.d))
        return self

    def __exit__(self, *exc):
        import os
        import sys
        os.chdir(self.old)
        if sys.path and sys.path[0] == str(self.d):
            sys.path.pop(0)
        return False


def check_against_legacy_config(verbose: bool = True, legacy_dir=None) -> list:
    """If the submitted config.py is available, confirm shared values match."""
    if legacy_dir is None:
        if verbose:
            print('  [info] legacy config.py not found -- skipping consistency check')
        return []
    try:
        with _InDir(legacy_dir):
            import config as cfg  # noqa: F401
    except Exception as e:  # pragma: no cover
        if verbose:
            print(f'  [info] could not import legacy config.py ({e}) -- skipping')
        return []
    pairs = [('C_LIGHT', C_LIGHT), ('FC', FC), ('M_TONES', M_TONES),
             ('DELTA_F', DELTA_F), ('TSYM', TSYM), ('NSw', NSW),
             ('N_ELEM_X', N_X), ('RS', RS), ('CJ', CJ), ('RJ_ON', RJ_ON),
             ('RJ_OFF', RJ_OFF), ('SNR_SINGLE_DB', SNR1_DB),
             ('R_RIS_T', R2_RIS_T), ('THETA1_DEG', THETA1_DEG)]
    bad = []
    for name, val in pairs:
        if hasattr(cfg, name) and not np.isclose(getattr(cfg, name), val):
            bad.append((name, getattr(cfg, name), val))
    if verbose:
        if bad:
            for n, a, b in bad:
                print(f'  [WARN] config.{n} = {a}  but revision uses {b}')
        else:
            print(f'  [ok] shared parameters match legacy config.py ({legacy_dir})')
    return bad
