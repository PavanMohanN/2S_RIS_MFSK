"""
tracking.py  —  Stage 5: Closed-Loop Beam Tracking
====================================================
Simulates the closed-loop α-smoother beam-tracking loop of the two-stage
RIS system over N_FRAMES CPI periods under a constant-velocity target model.

At each CPI frame t the loop executes:
  1. Propagate target:   P(t) = P(0) + t·T_CPI·V_vec              [Eq. 28]
  2. Beam gain:          G(t) = |AF(θ_true; θ̃(t−1))|²  (ULA AF)
  3. Effective SNR:      SNR_eff(t) = SNR_CPI · G(t)
  4. Angle CRB noise:    σ_θ(t)  = 1/√[SNR_eff·(2π/λ)²·Var(n·d)] [Eq. 29]
                         θ̂(t)   = θ_true(t) + 𝒩(0, σ_θ²)
  5. α-smoother:         θ̃(t)   = α·θ̂(t) + (1−α)·θ̃(t−1)         [Eq. 30]
  6. Range CRB noise:    σ_R(t)  = (c/2)/√[SNR_eff·(2π)²·Var(k·Δf)]
                         R̂(t)   = R_true(t) + 𝒩(0, σ_R²)

Output figures
--------------
F14 — (a) 2-D bird's-eye scene: true trajectory, beam-steering arrows,
           estimated positions colour-coded by frame index
       (b) Angle commands vs CPI frame: θ_true / θ̂ / θ̃ + ±2σ CRB band
       Double-column IEEE TVT  (7.16 × 2.8 in)
F15 — (a) 2-D position-error scatter (Δx, Δz), colour by frame index
       (b) Empirical CDFs of ‖ΔP‖, |ΔR|, |RΔθ| with CRB reference line
       Double-column IEEE TVT  (7.16 × 2.8 in)

Output table
------------
T3  — Tracking scenario and performance metrics  (tables/T3_tracking_params.*)

Output data
-----------
data/tracking_data.npy

Usage
-----
    python tracking.py          # standalone
    from tracking import run, simulate_tracking_loop
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

# ── Consistent IEEE TVT style (identical to Phases 1–4) ──────────────────
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
# 1.  Pre-computed FIM constants  (same derivation as Phase 4)
# ═══════════════════════════════════════════════════════════════════════════

def _fisher_constants():
    """Var[n·d] for ULA DOA FIM, and Var[k·Δf] for range-delay FIM."""
    n_pos   = np.arange(cfg.N_ELEM_X, dtype=float) * cfg.D_ELEM
    k_tones = np.arange(1, cfg.M_TONES + 1, dtype=float) * cfg.DELTA_F
    return np.var(n_pos), np.var(k_tones)   # [m²], [Hz²]


_VAR_D, _VAR_F = _fisher_constants()

# Operating-point CRB references (used in figures and table)
_SNR_EFF_OP = 10.0 ** (cfg.SNR_SINGLE_DB / 10.0) * cfg.M_TONES * cfg.NSw
CRB_THETA_DEG = np.rad2deg(
    1.0 / np.sqrt(_SNR_EFF_OP * (2 * np.pi / cfg.WAVELENGTH) ** 2 * _VAR_D))
CRB_R_CM = 100.0 * (cfg.C_LIGHT / 2.0) / np.sqrt(
    _SNR_EFF_OP * (2 * np.pi) ** 2 * _VAR_F)


def _sigma_theta(snr_eff_lin: float) -> float:
    """Angle estimation σ [rad] from FIM at given effective (linear) SNR."""
    I = snr_eff_lin * (2.0 * np.pi / cfg.WAVELENGTH) ** 2 * _VAR_D
    return 1.0 / np.sqrt(max(I, 1e-30))


def _sigma_range(snr_eff_lin: float) -> float:
    """Range estimation σ [m] from FIM.
    Formula (identical to estimation_crb.py §Phase 4):
        σ_R = (c/2) / sqrt( SNR_eff · (2π)² · Var[k·Δf] )
    """
    I = snr_eff_lin * (2.0 * np.pi) ** 2 * _VAR_F    # [s⁻²]
    return (cfg.C_LIGHT / 2.0) / np.sqrt(max(I, 1e-30))


def _ula_af_gain(theta_true: float, theta_beam: float) -> float:
    """Normalised ULA power-pattern gain (0 … 1) for beam mispointing.
    Uses the Dirichlet kernel: |sinc(N·Δψ/2) / sinc(Δψ/2)|².
    """
    dphi = (2.0 * np.pi * cfg.D_ELEM / cfg.WAVELENGTH) * (
        np.sin(theta_true) - np.sin(theta_beam))
    if abs(dphi) < 1e-9:
        return 1.0
    N = cfg.N_ELEM_X
    return float((np.sin(N * dphi / 2.0) / (N * np.sin(dphi / 2.0))) ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Constant-velocity target trajectory
# ═══════════════════════════════════════════════════════════════════════════

def simulate_trajectory(n_frames: int = None) -> dict:
    """
    Generate exact target trajectory under P(t) = P(0) + t·T_CPI·V  [Eq. 28].

    All geometry from config:  P_TARGET, V_VEC, TCPI, N_FRAMES.

    Returns
    -------
    traj : dict with arrays of length n_frames
        't'          frame indices  0 … N−1
        'pos'        (N,2) true (x, z) [m]
        'R'          (N,)  true range [m]
        'theta_rad'  (N,)  true azimuth [rad]
        'theta_deg'  (N,)  true azimuth [deg]
    """
    if n_frames is None:
        n_frames = cfg.N_FRAMES
    t   = np.arange(n_frames)
    pos = (cfg.P_TARGET[np.newaxis, :]
           + t[:, np.newaxis] * cfg.TCPI * cfg.V_VEC[np.newaxis, :])
    R   = np.linalg.norm(pos, axis=1)
    th  = np.arctan2(pos[:, 0], pos[:, 1])
    return {'t': t, 'pos': pos, 'R': R,
            'theta_rad': th, 'theta_deg': np.rad2deg(th)}


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Closed-loop α-smoother tracking loop
# ═══════════════════════════════════════════════════════════════════════════

def simulate_tracking_loop(traj: dict, alpha: float = None,
                           seed: int = None) -> dict:
    """
    Run the closed-loop beam-tracking simulation [Eq. 29–30].

    At each frame the beam gain due to mispointing reduces the effective SNR,
    increasing the CRB noise, which feeds back into the smoother.

    Parameters
    ----------
    traj  : from simulate_trajectory()
    alpha : smoother coefficient  (default cfg.ALPHA_SMOOTH = 0.3)
    seed  : RNG seed

    Returns
    -------
    track : dict with (N,) arrays
        'theta_est_deg'    noisy per-frame estimates [deg]
        'theta_smooth_deg' α-smoothed beam command [deg]
        'theta_smooth_rad' α-smoothed beam command [rad]
        'R_est'            range estimates [m]
        'pos_est'          (N,2) estimated positions [m]
        'snr_eff_db'       effective CPI SNR per frame [dB]
        'G_eff'            beam power gain (0 → 1)
        'sigma_theta_deg'  angle noise σ per frame [deg]
        'sigma_R_m'        range noise σ per frame [m]
    """
    rng   = np.random.default_rng(cfg.RANDOM_SEED if seed is None else seed)
    alpha = cfg.ALPHA_SMOOTH if alpha is None else alpha
    N     = len(traj['t'])
    snr_cpi_lin = 10.0 ** (cfg.SNR_CPI_DB / 10.0)

    th_est_d = np.zeros(N); th_sm_d = np.zeros(N); th_sm_r = np.zeros(N)
    R_est    = np.zeros(N); snr_db  = np.zeros(N)
    G_eff    = np.zeros(N); sig_th  = np.zeros(N); sig_R   = np.zeros(N)

    th_prev = traj['theta_rad'][0]     # initialise: beam → true initial direction

    for t in range(N):
        th_true = traj['theta_rad'][t]
        R_true  = traj['R'][t]

        # --- Step 2: beam gain -----------------------------------------------
        g = _ula_af_gain(th_true, th_prev)
        G_eff[t]  = g
        snr_eff   = max(snr_cpi_lin * g, 1e-9)
        snr_db[t] = 10.0 * np.log10(snr_eff)

        # --- Step 4: CRB-based noisy estimates --------------------------------
        s_th = _sigma_theta(snr_eff)       # [rad]
        s_R  = _sigma_range(snr_eff)       # [m]
        sig_th[t] = np.rad2deg(s_th)
        sig_R[t]  = s_R

        th_hat    = th_true + rng.normal(0.0, s_th)
        R_est[t]  = R_true  + rng.normal(0.0, s_R)
        th_est_d[t] = np.rad2deg(th_hat)

        # --- Step 5: α-smoother ----------------------------------------------
        th_s    = alpha * th_hat + (1.0 - alpha) * th_prev
        th_sm_r[t] = th_s
        th_sm_d[t] = np.rad2deg(th_s)
        th_prev = th_s

    pos_est = np.column_stack([
        R_est * np.sin(th_sm_r),
        R_est * np.cos(th_sm_r),
    ])
    return {
        'theta_est_deg':    th_est_d,
        'theta_smooth_deg': th_sm_d,
        'theta_smooth_rad': th_sm_r,
        'R_est':            R_est,
        'pos_est':          pos_est,
        'snr_eff_db':       snr_db,
        'G_eff':            G_eff,
        'sigma_theta_deg':  sig_th,
        'sigma_R_m':        sig_R,
    }


def compute_errors(traj: dict, track: dict) -> dict:
    """Per-frame position, range, and tangential errors."""
    dpos = track['pos_est'] - traj['pos']
    return {
        'dpos':    dpos,
        'err_pos': np.linalg.norm(dpos, axis=1),
        'err_R':   np.abs(track['R_est'] - traj['R']),
        'err_tan': traj['R'] * np.abs(track['theta_smooth_rad'] - traj['theta_rad']),
    }


def _print_summary(traj, track, err):
    sep = '─' * 60
    print(f'\n{sep}')
    print('  Tracking Performance Summary')
    print(sep)
    print(f'  Frames / total time  = {cfg.N_FRAMES} × {cfg.TCPI*1e3:.2f} ms'
          f' = {cfg.N_FRAMES*cfg.TCPI*1e3:.1f} ms')
    print(f'  Range change  ΔR     = {traj["R"][-1]-traj["R"][0]:+.3f} m')
    print(f'  Angle change  Δθ     = {traj["theta_deg"][-1]-traj["theta_deg"][0]:+.3f}°')
    print(f'  Mean beam gain G     = {np.mean(track["G_eff"])*100:.2f} %')
    print(f'  Mean eff. SNR        = {np.mean(track["snr_eff_db"]):.1f} dB '
          f'(target {cfg.SNR_CPI_DB:.1f} dB)')
    print(f'  CRB σ_θ (op. pt.)   = {CRB_THETA_DEG:.4f}°')
    print(f'  CRB σ_R (op. pt.)   = {CRB_R_CM:.3f} cm')
    print(f'  RMSE position        = {np.sqrt(np.mean(err["err_pos"]**2))*100:.3f} cm')
    print(f'  RMSE range           = {np.sqrt(np.mean(err["err_R"]**2))*100:.3f} cm')
    print(f'  RMSE tangential      = {np.sqrt(np.mean(err["err_tan"]**2))*100:.3f} cm')
    print(f'  95-pct pos. error    = {np.percentile(err["err_pos"]*100, 95):.3f} cm')
    print(sep + '\n')


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Figure F14  —  Scene  +  angle-command history
# ═══════════════════════════════════════════════════════════════════════════

def _save_fig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(cfg.OUTPUT_DIR, f'{name}.{ext}'),
                    dpi=(cfg.FIG_DPI if ext == 'png' else None))
    print(f'  Saved: figures/{name}.pdf | .png')


def plot_F14(traj, track, save=True):
    """F14 — (a) 2-D tracking scene  (b) Angle-command history."""
    fig = plt.figure(figsize=(cfg.FIG_WIDTH_2COL, 2.8), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.0])

    t = traj['t'];  pos = traj['pos'];  pe = track['pos_est']

    # ── Panel (a): bird's-eye scene ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    for Rc in [5, 10, 15]:                                  # range circles
        tc = np.linspace(-np.pi / 2, np.pi / 2, 300)
        ax1.plot(Rc * np.sin(tc), Rc * np.cos(tc),
                 lw=0.5, ls=':', color=cfg.COLORS['gray'], alpha=0.45)
        ax1.text(0, Rc + 0.25, f'{Rc} m', fontsize=10,
                 ha='center', color=cfg.COLORS['gray'], va='bottom')

    ax1.plot(pos[:, 0], pos[:, 1], '-', color=cfg.COLORS['sideband'],
             lw=2, zorder=4, label='True trajectory')
    for idx, mk, lbl in [(0, 'o', '$t_0$'), (-1, 's', f'$t_{{{cfg.N_FRAMES}}}$')]:
        ax1.plot(*pos[idx], mk, ms=6, color=cfg.COLORS['sideband'], zorder=5)
        ax1.text(pos[idx, 0] + 0.65, pos[idx, 1], lbl,
                 fontsize=10, color=cfg.COLORS['sideband'], va='center')

    sc = ax1.scatter(pe[:, 0], pe[:, 1], c=t, cmap='cool',
                     s=9, zorder=6, alpha=0.80, label='Estimates')
    cb = fig.colorbar(sc, ax=ax1, pad=0.02, fraction=0.06, shrink=0.80)
    cb.set_label('Frame', fontsize=10);  cb.ax.tick_params(labelsize=10)

    for ti in range(0, cfg.N_FRAMES, 10):                   # beam arrows
        th_b = track['theta_smooth_rad'][ti]
        L    = traj['R'][ti] * 0.75
        ax1.annotate('', xy=(L * np.sin(th_b), L * np.cos(th_b)),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', lw=0.8,
                                     color=cfg.COLORS['off'], alpha=0.55))

    ax1.plot(0, 0, '*', ms=10, color=cfg.COLORS['carrier'],
             zorder=7, label='R-RIS')
    ax1.set_xlabel('Cross-range $x$ (m)',fontsize = 10);  ax1.set_ylabel('Down-range $z$ (m)',fontsize=10)
    
    ax1.set_xlim([-1, 13]);  ax1.set_ylim([-0.5, 14])
    ax1.set_aspect('equal', adjustable='box')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper left', fontsize=8, handlelength=1.5,bbox_to_anchor=(0.4,1))
    #ax1.set_title('(a) Tracking scene', fontsize=cfg.FIG_TITLE,
     #             fontweight='normal', pad=3)

    # ── Panel (b): angle-command history ──────────────────────────────────
    ax2   = fig.add_subplot(gs[1])
    th_t  = traj['theta_deg']
    th_e  = track['theta_est_deg']
    th_sm = track['theta_smooth_deg']
    sig_m = np.mean(track['sigma_theta_deg'])

    ax2.fill_between(t, th_t - 2 * sig_m, th_t + 2 * sig_m,
                     color=cfg.COLORS['sideband'], alpha=0.13,
                     label=rf'±2σ (σ = {sig_m:.3f}°)')
    ax2.plot(t, th_t,  '-',  color=cfg.COLORS['sideband'], lw=1.5,
             label=r'True $\theta_{\rm true}$', zorder=4)
    ax2.plot(t, th_e,  '.',  color=cfg.COLORS['off'], ms=2.5, alpha=0.50,
             label=r'Noisy $\hat{\theta}$', zorder=3)
    ax2.plot(t, th_sm, '-',  color=cfg.COLORS['on'], lw=1.4,
             label=rf'Smooth $\tilde{{\theta}}$  ($\alpha$={cfg.ALPHA_SMOOTH})',
             zorder=5)

    ax2.set_xlabel('CPI frame index $t$',fontsize = 10);  ax2.set_ylabel('Azimuth (degrees)',fontsize= 10)
    ax2.set_xlim([0, cfg.N_FRAMES - 1])
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.grid(True, which='major')
    ax2.grid(True, which='minor', alpha=0.15, lw=0.3)
    ax2.legend(loc='lower left', fontsize=8.0, handlelength=1.8, ncol=2)
    #ax2.set_title('(b) Angle commands vs CPI frame',
     #             fontsize=cfg.FIG_TITLE, fontweight='normal', pad=3)

    if save:
        _save_fig(fig, 'F14_tracking_trajectory')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Figure F15  —  Error scatter  +  empirical CDFs
# ═══════════════════════════════════════════════════════════════════════════

def plot_F15(traj, track, err, save=True):
    """F15 — (a) 2-D error scatter  (b) Empirical CDFs of tracking errors."""
    fig = plt.figure(figsize=(cfg.FIG_WIDTH_2COL, 2.8), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)

    t    = traj['t']
    dpos = err['dpos'] * 100            # cm
    epos = err['err_pos'] * 100
    eR   = err['err_R'] * 100
    etan = err['err_tan'] * 100

    # ── (a) Error scatter ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    sc  = ax1.scatter(dpos[:, 0], dpos[:, 1], c=t, cmap='cool',
                      s=15, alpha=0.80, zorder=4)
    cb  = fig.colorbar(sc, ax=ax1, pad=0.02, fraction=0.07, shrink=0.85)
    cb.set_label('Frame', fontsize=10);  cb.ax.tick_params(labelsize=10)

    for pct, ls in [(50, '-'), (95, '--')]:
        r_c = np.percentile(epos, pct)
        tc  = np.linspace(0, 2 * np.pi, 300)
        ax1.plot(r_c * np.cos(tc), r_c * np.sin(tc), ls=ls, lw=1.0,
                 color=cfg.COLORS['off'],
                 label=f'{pct}th pct: {r_c:.2f} cm')

    ax1.axhline(0, lw=0.5, ls=':', color=cfg.COLORS['gray'], alpha=0.5)
    ax1.axvline(0, lw=0.5, ls=':', color=cfg.COLORS['gray'], alpha=0.5)
    ax1.plot(0, 0, '+', ms=8, color=cfg.COLORS['gray'], zorder=5)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    lim = max(np.abs(dpos).max() * 1.25, 0.5)
    ax1.set_xlim([-lim, lim]);  ax1.set_ylim([-lim, lim])
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel(r'$\Delta x$ (cm)',fontsize = 10);  ax1.set_ylabel(r'$\Delta z$ (cm)',fontsize = 10)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper right', fontsize=8.0, handlelength=1.5)
   # ax1.set_title('(a) Position-error scatter',
    #              fontsize=cfg.FIG_TITLE, fontweight='normal', pad=3)

    # ── (b) CDFs ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    def ecdf(x):
        xs = np.sort(x)
        return xs, np.arange(1, len(xs) + 1) / len(xs)

    for vals, lbl, col, ls in [
        (epos, r'Position $\|\Delta P\|$', cfg.COLORS['on'],       '-'),
        (eR,   r'Range $|\Delta R|$',      cfg.COLORS['off'],      '--'),
        (etan, r'Tang. $|R\Delta\theta|$', cfg.COLORS['sideband'], '-.'),
    ]:
        xs, ys = ecdf(vals)
        ax2.plot(xs, ys, color=col, lw=1.4, ls=ls, label=lbl)

    ax2.axvline(CRB_R_CM, color=cfg.COLORS['crb'], lw=0.9, ls=':',
                label=f'CRB $\\sigma_R$ = {CRB_R_CM:.2f} cm')

    ax2.set_xlabel('Error (cm)',fontsize=10);  ax2.set_ylabel('Empirical CDF',fontsize = 10)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_xlim([0, None]);       ax2.set_ylim([0, 1.02])
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc='lower right', fontsize=8.0, handlelength=1.8)
    #ax2.set_title('(b) Empirical CDF of tracking errors',
     #             fontsize=cfg.FIG_TITLE, fontweight='normal', pad=3)

    if save:
        _save_fig(fig, 'F15_tracking_errors')
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Table T3
# ═══════════════════════════════════════════════════════════════════════════

def export_table_T3(traj, track, err):
    """T3 — Tracking scenario + performance metrics."""
    rmse_p = np.sqrt(np.mean(err['err_pos'] ** 2)) * 100
    rmse_R = np.sqrt(np.mean(err['err_R']  ** 2)) * 100
    rmse_t = np.sqrt(np.mean(err['err_tan'] ** 2)) * 100
    p95    = np.percentile(err['err_pos'] * 100, 95)

    hdr  = ['Parameter', 'Symbol', 'Value']
    rows = [
        ['Initial range',              r'$R_0$',                f'{cfg.R_RIS_T:.0f} m'],
        ['Initial angle',              r'$\theta_0$',           f'{cfg.THETA1_DEG:.0f}°'],
        ['Target speed',               r'$|\mathbf{v}|$',       f'{cfg.V_TARGET:.1f} m/s'],
        ['Target heading',             r'$\psi_v$',             f'{cfg.THETA_T_DEG:.0f}°'],
        ['Final range (frame 50)',     r'$R_{50}$',             f'{traj["R"][-1]:.2f} m'],
        ['Final angle (frame 50)',     r'$\theta_{50}$',        f'{traj["theta_deg"][-1]:.2f}°'],
        ['CPI frames tracked',         r'$N_{\rm fr}$',         f'{cfg.N_FRAMES}'],
        ['CPI duration',               r'$T_{\rm CPI}$',        f'{cfg.TCPI*1e3:.2f} ms'],
        ['Total tracking duration',    r'$T_{\rm tot}$',        f'{cfg.N_FRAMES*cfg.TCPI*1e3:.1f} ms'],
        [r'$\alpha$-smoother coeff.', r'$\alpha$',             f'{cfg.ALPHA_SMOOTH}'],
        ['CRB angle σ (op. pt.)',      r'$\sigma_\theta^{\rm CRB}$', f'{CRB_THETA_DEG:.4f}°'],
        ['Mean track σ_θ',             r'$\bar\sigma_\theta$', f'{np.mean(track["sigma_theta_deg"]):.4f}°'],
        ['Mean track σ_R',             r'$\bar\sigma_R$',      f'{np.mean(track["sigma_R_m"])*100:.3f} cm'],
        ['RMSE position',              r'$\varepsilon_{\rm pos}$', f'{rmse_p:.3f} cm'],
        ['RMSE range',                 r'$\varepsilon_R$',          f'{rmse_R:.3f} cm'],
        ['RMSE tangential',            r'$\varepsilon_{\rm tan}$',  f'{rmse_t:.3f} cm'],
        ['95th-pct position error',    r'$e_{95}$',                 f'{p95:.3f} cm'],
    ]
    tex = _tex3('Tracking Scenario and Performance Parameters',
                'tab:tracking', hdr, rows,
                r'Closed-loop $\alpha$-smoother over $N_{\rm fr}='
                + str(cfg.N_FRAMES) + r'$ CPIs at SNR$_{\rm CPI}='
                + f'{cfg.SNR_CPI_DB:.1f}' + r'$ dB.')
    _write('T3_tracking_params', tex, _csv(hdr, rows))


def _tex3(caption, label, hdr, rows, notes=''):
    L = [r'\begin{table}[!t]', r'\renewcommand{\arraystretch}{1.15}',
         r'\caption{' + caption + '}', r'\label{' + label + '}',
         r'\centering', r'\begin{tabular}{l l r}',
         r'\hline\hline', ' & '.join(hdr) + r' \\', r'\hline']
    for r in rows:
        L.append(' & '.join(str(c) for c in r) + r' \\')
    L += [r'\hline\hline']
    if notes:
        L += [r'\multicolumn{3}{l}{\footnotesize ' + notes + r'} \\',
              r'\hline']
    L += [r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)

def _csv(hdr, rows):
    return '\n'.join([','.join(str(h) for h in hdr)] +
                     [','.join(str(c) for c in r) for r in rows])

def _write(name, tex, csv):
    for ext, content in [('.tex', tex), ('.csv', csv)]:
        with open(os.path.join(cfg.TABLE_DIR, name + ext), 'w',encoding='utf-8') as f:
            f.write(content)
    print(f'  Exported: tables/{name}.tex + .csv')


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Data export  +  main entry point
# ═══════════════════════════════════════════════════════════════════════════

def export_data(traj, track, err):
    p = os.path.join(cfg.DATA_DIR, 'tracking_data.npy')
    np.save(p, {'traj': traj, 'track': track, 'err': err}, allow_pickle=True)
    print('  Exported: data/tracking_data.npy')


def run(show: bool = False):
    """Run Stage 5 — Closed-Loop Beam Tracking.  Returns (traj, track, err)."""
    print('\n' + '═' * 60)
    print('  Stage 5 — Closed-Loop Beam Tracking')
    print('  Generating Figures F14, F15 | Table T3')
    print('═' * 60)
    traj  = simulate_trajectory()
    track = simulate_tracking_loop(traj)
    err   = compute_errors(traj, track)
    _print_summary(traj, track, err)
    plot_F14(traj, track, save=True)
    plot_F15(traj, track, err, save=True)
    export_table_T3(traj, track, err)
    export_data(traj, track, err)
    if show:
        plt.show()
    plt.close('all')
    print('\n  Stage 5 complete — 2 figures + 1 table + data saved.')
    return traj, track, err


if __name__ == '__main__':
    run()
