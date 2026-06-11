"""
multi_target.py  —  Multi-Target Sensing & Range–Doppler Decoupling
==================================================================================
Quantifies the range–Doppler coupling characteristics of multi-target bistatic 
sensing scenes, providing a direct performance contrast between M-FSK wave 
mechanics and single-chirp FMCW architectures.

Core Physical Principles
------------------------
  * FMCW Coupling Dynamics : Evaluates the joint range-Doppler dependency within 
                             the dechirped beat frequency where target velocity 
                             induces a linear shift in the apparent range profile[cite: 7].
  * M-FSK Orthogonality    : Models tone-by-tone continuous-wave transmission blocks 
                             where target range delay is isolated within the tone-to-tone 
                             phase, and Doppler shifts are parsed orthogonally across the 
                             sweep-to-sweep domain to eliminate ghost targets[cite: 7].

Multi-Target Physical Profiles
------------------------------
  * Target 1 : Static clutter reference positioned at a close-range boundary[cite: 7].
  * Target 2 : Receding dynamic profile exhibiting positive radial velocity displacement[cite: 7].
  * Target 3 : Approaching dynamic profile exhibiting negative radial velocity displacement[cite: 7].

Functional Deliverables
-----------------------
  * F16 — Comparative multi-panel plot contrasting the decoupled, true-peak tracking of 
          the M-FSK IFFT profile against the coupled, displacement-distorted peaks 
          (ghost targets) of the conventional FMCW FFT profile[cite: 7].

Data Output
-----------
  * data/multi_target_data.npy : Packaged arrays containing target kinematics, absolute 
                                 M-FSK spectral profiles, and coupled FMCW data structures[cite: 7].

Execution Parameters
--------------------
    python multi_target.py
    from multi_target import run
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------------ #
#  1.  PHYSICAL CONSTANTS AND SYSTEM PARAMETERS       #
# ------------------------------------------------------------------ #
C_LIGHT   = 299_792_458.0          # speed of light [m/s]
FC        = 28e9                   # carrier frequency [Hz]
WAVELEN   = C_LIGHT / FC           # wavelength ~10.71 mm
N_ELEM    = 16                     # elements along the steered (linear) cut
D_ELEM    = WAVELEN / 2            # element spacing (lambda/2)
M_TONES   = 12                     # number of M-FSK tones
DELTA_F   = 25e6                   # tone spacing [Hz]
T_CPI     = 17.2e-3               # coherent processing interval [s]

# Reference operating point (paper Table I / V)
SNR_REF_DB   = 15.6                # post-CPI SNR at the operating point
SIGMA_R_REF  = 0.07                # range meas. std at SNR_REF [m]  (~MC RMSE)
SIGMA_TH_REF = np.deg2rad(0.55)    # angle meas. std at SNR_REF [rad] (~mean track sigma_theta)

# Paper's nominal scenario
R0_INIT      = 10.0                # initial range [m]
THETA0_INIT  = 60.0                # initial angle from boresight [deg]
SPEED_INIT   = 2.0                 # target speed [m/s]
HEADING_INIT = 45.0                # heading from +x axis [deg]
ALPHA_INIT   = 0.30                # alpha-smoother coefficient
FRAMES_SCEN  = 50                  # frames in the paper scenario (loops after)

# Fixed bistatic geometry (for context in the scene)
RIS_POS = np.array([0.0, 0.0])     # RIS stack at the origin (boresight = +y)
TX_POS  = np.array([-8.0, 0.0])    # CW transmitter, 8 m from RIS
RX_POS  = np.array([4.0, -2.5])    # bistatic receiver

# Broadside half-power beamwidth for an N-element lambda/2 array
HPBW_BROADSIDE = 0.886 * WAVELEN / (N_ELEM * D_ELEM)   # [rad] ~6.35 deg


# ------------------------------------------------------------------ #
#  2.  CORE PHYSICS HELPERS                                          #
# ------------------------------------------------------------------ #
def array_factor(theta_eval, theta_steer, n=N_ELEM, d_over_lambda=0.5):
    """Normalised |AF| of a uniform linear array steered to `theta_steer`,
    evaluated at angles `theta_eval` (all angles measured from boresight,
    in radians). Returns linear magnitude in [0, 1]."""
    theta_eval = np.atleast_1d(theta_eval).astype(float)
    psi = 2 * np.pi * d_over_lambda * (np.sin(theta_eval) - np.sin(theta_steer))
    # AF = (1/N) sin(N psi/2)/sin(psi/2), with the removable singularity = 1
    num = np.sin(n * psi / 2.0)
    den = np.sin(psi / 2.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        af = np.where(np.abs(den) < 1e-12, 1.0, num / (n * den))
    return np.abs(af)


def beam_gain(theta_true, theta_steer):
    """Realised normalised beam power gain (|AF|^2) at the true target angle
    when the beam is steered to `theta_steer`."""
    return float(array_factor(theta_true, theta_steer)[0] ** 2)


def hpbw_at(theta_steer):
    """Approximate half-power beamwidth [rad] when steered to `theta_steer`."""
    return HPBW_BROADSIDE / max(np.cos(theta_steer), 0.15)


def meas_noise_std(snr_db):
    """Range and angle measurement std at a given post-CPI SNR.
    Scales as 1/sqrt(SNR_linear) about the reference operating point."""
    scale = 10 ** ((SNR_REF_DB - snr_db) / 20.0)   # std ratio
    return SIGMA_R_REF * scale, SIGMA_TH_REF * scale


def pos_to_range_angle(pos):
    """Cartesian position -> (range, angle-from-boresight). Boresight = +y."""
    r = np.hypot(pos[0], pos[1])
    th = np.arctan2(pos[0], pos[1])      # angle from +y axis
    return r, th


def range_angle_to_pos(r, th):
    """(range, angle-from-boresight) -> Cartesian position."""
    return np.array([r * np.sin(th), r * np.cos(th)])


# ------------------------------------------------------------------ #
#  3.  TARGET + CLOSED-LOOP ALPHA-SMOOTHER TRACKER                    #
# ------------------------------------------------------------------ #
class RISTracker:
    """Holds the true target state and the closed-loop tracker state, and
    advances both by one CPI per `step()` call."""

    def __init__(self):
        self.reset()

    # -- parameters the user can change live -----------------------------
    speed   = SPEED_INIT          # m/s
    heading = HEADING_INIT        # deg (from +x axis)
    alpha   = ALPHA_INIT          # smoother gain
    snr_db  = SNR_REF_DB          # post-CPI SNR
    use_smoother = True

    def reset(self):
        """Restart from the paper's initial condition."""
        self.frame = 0
        self.true_pos = range_angle_to_pos(R0_INIT, np.deg2rad(THETA0_INIT))
        # Tracker initialises on the first measurement; velocity acquired live.
        self.est_pos = self.true_pos.copy()
        self.vel_est = np.zeros(2)
        self.theta_cmd = np.deg2rad(THETA0_INIT)   # current beam command
        # Histories (lists, grown each step)
        self.hist_frame      = []
        self.hist_true_th    = []
        self.hist_meas_th    = []
        self.hist_cmd_th     = []
        self.hist_pos_err_cm = []
        self.hist_gain_pct   = []
        self.hist_true_pos   = []
        self.hist_est_pos    = []
        self._last_meas_pos  = None

    def _velocity_vector(self):
        h = np.deg2rad(self.heading)
        return self.speed * np.array([np.cos(h), np.sin(h)])

    def step(self):
        """Advance true target + tracker by one CPI. Returns a dict snapshot."""
        dt = T_CPI

        # --- 1. advance the true target (constant-velocity over this CPI) ---
        self.true_pos = self.true_pos + self._velocity_vector() * dt
        r_true, th_true = pos_to_range_angle(self.true_pos)

        # --- 2. noisy CPI measurement (variance set by SNR / CRB) ----------
        sig_r, sig_th = meas_noise_std(self.snr_db)
        r_meas  = r_true  + np.random.normal(0, sig_r)
        th_meas = th_true + np.random.normal(0, sig_th)
        meas_pos = range_angle_to_pos(r_meas, th_meas)

        # --- 3. alpha-smoother predict / correct ---------------------------
        if self.use_smoother:
            # predict one CPI ahead from current estimate + estimated velocity
            pred_pos = self.est_pos + self.vel_est * dt
            resid    = meas_pos - pred_pos
            new_est  = pred_pos + self.alpha * resid
            # smooth the velocity estimate (reduces noise amplification)
            inst_vel = (new_est - self.est_pos) / dt
            self.vel_est = (1 - self.alpha) * self.vel_est + self.alpha * inst_vel
            self.est_pos = new_est
            # beam command: point one CPI AHEAD of the smoothed estimate
            lead_pos = self.est_pos + self.vel_est * dt
            self.theta_cmd = np.arctan2(lead_pos[0], lead_pos[1])
        else:
            # no smoothing: steer straight at the raw (noisy) measurement
            self.est_pos = meas_pos
            self.theta_cmd = th_meas

        # --- 4. realised beam gain at the true angle -----------------------
        gain = beam_gain(th_true, self.theta_cmd)

        # --- 5. errors / bookkeeping ---------------------------------------
        pos_err_cm = np.linalg.norm(self.est_pos - self.true_pos) * 100.0

        self.frame += 1
        self.hist_frame.append(self.frame)
        self.hist_true_th.append(np.rad2deg(th_true))
        self.hist_meas_th.append(np.rad2deg(th_meas))
        self.hist_cmd_th.append(np.rad2deg(self.theta_cmd))
        self.hist_pos_err_cm.append(pos_err_cm)
        self.hist_gain_pct.append(gain * 100.0)
        self.hist_true_pos.append(self.true_pos.copy())
        self.hist_est_pos.append(self.est_pos.copy())
        self._last_meas_pos = meas_pos

        # auto-loop when the target leaves the field of view
        if r_true > 18.0 or not (np.deg2rad(5) < th_true < np.deg2rad(85)):
            self.reset()

        return dict(r_true=r_true, th_true=th_true, meas_pos=meas_pos,
                    gain=gain, pos_err_cm=pos_err_cm)

    # -- running statistics (skip first few acquisition frames) ----------
    def running_stats(self, skip=5):
        e = np.array(self.hist_pos_err_cm[skip:]) if len(self.hist_pos_err_cm) > skip else np.array([])
        g = np.array(self.hist_gain_pct[skip:])  if len(self.hist_gain_pct) > skip else np.array([])
        if e.size == 0:
            return dict(rmse=0.0, p95=0.0, mean_gain=100.0)
        return dict(rmse=float(np.sqrt(np.mean(e ** 2))),
                    p95=float(np.percentile(e, 95)),
                    mean_gain=float(np.mean(g)))


# ------------------------------------------------------------------ #
#  4.  FIGURE / PANEL LAYOUT                                          #
# ------------------------------------------------------------------ #
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'figure.facecolor': 'white', 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': '--',
})

COL_TRUE = '#1f6feb'    # true target
COL_EST  = '#d29922'    # estimate
COL_CMD  = '#8957e5'    # beam command
COL_BEAM = '#a371f7'    # beam cone
COL_GOOD = '#2da44e'
COL_BAD  = '#cf222e'

tracker = RISTracker()

fig = plt.figure(figsize=(16, 9.5))
fig.suptitle('Two-Stage RIS M-FSK Bistatic Sensing  —  Closed-Loop Beam Tracking Demonstrator',
             fontsize=14, fontweight='bold', y=0.985)

gs = fig.add_gridspec(3, 3, left=0.05, right=0.975, top=0.93, bottom=0.27,
                      hspace=0.42, wspace=0.28,
                      width_ratios=[1.25, 1.0, 1.0], height_ratios=[1, 1, 1])

ax_scene = fig.add_subplot(gs[0:2, 0])     # bistatic scene (big, left)
ax_beam  = fig.add_subplot(gs[0, 1])       # R-RIS array factor
ax_zoom  = fig.add_subplot(gs[0, 2])       # magnified tracking view
ax_angle = fig.add_subplot(gs[1, 1:3])     # angle tracking time series
ax_err   = fig.add_subplot(gs[2, 0:2])     # position-error history
ax_info  = fig.add_subplot(gs[2, 2])       # live metrics text panel
ax_info.axis('off')

# ---- 4a. Scene (static context) ------------------------------------
def setup_scene():
    ax_scene.clear()
    ax_scene.set_title('Bistatic Scene  (bird\'s-eye, RIS at origin, boresight = +y)')
    ax_scene.set_xlabel('x  [m]'); ax_scene.set_ylabel('y  [m]')
    ax_scene.set_xlim(-10, 14); ax_scene.set_ylim(-5, 16)
    ax_scene.set_aspect('equal', adjustable='box')
    ax_scene.grid(True, alpha=0.3)
    # fixed nodes
    ax_scene.plot(*TX_POS, marker='^', ms=13, color='#57606a', zorder=5)
    ax_scene.annotate('Tx (CW)', TX_POS, textcoords='offset points',
                      xytext=(-6, -16), fontsize=8, color='#57606a')
    ax_scene.plot(*RX_POS, marker='s', ms=11, color='#0969da', zorder=5)
    ax_scene.annotate('Rx (bistatic)', RX_POS, textcoords='offset points',
                      xytext=(6, -4), fontsize=8, color='#0969da')
    # RIS stack drawn as a short bar at the origin
    ax_scene.plot([-0.9, 0.9], [0, 0], lw=6, color='#24292f',
                  solid_capstyle='round', zorder=6)
    ax_scene.annotate('T-RIS / R-RIS', RIS_POS, textcoords='offset points',
                      xytext=(-30, -18), fontsize=8, color='#24292f')

setup_scene()

# dynamic scene artists
beam_patch = Polygon([[0, 0], [0, 1], [0, 1]], closed=True,
                     color=COL_BEAM, alpha=0.22, zorder=2)
ax_scene.add_patch(beam_patch)
beam_axis_line, = ax_scene.plot([], [], color=COL_CMD, lw=1.4, ls='--', zorder=3)
true_dot,  = ax_scene.plot([], [], 'o', ms=10, color=COL_TRUE, zorder=7, label='True target')
est_dot,   = ax_scene.plot([], [], 'x', ms=9, mew=2, color=COL_EST, zorder=8, label='Estimate')
true_trail,= ax_scene.plot([], [], '-', lw=1.2, color=COL_TRUE, alpha=0.5)
tx_ray,    = ax_scene.plot([], [], ':', lw=1, color='#57606a', alpha=0.7)
rx_ray,    = ax_scene.plot([], [], ':', lw=1, color='#0969da', alpha=0.7)
ax_scene.legend(loc='upper right', fontsize=8, framealpha=0.9)

# ---- 4b. Array-factor panel ----------------------------------------
theta_grid = np.linspace(0, np.deg2rad(90), 600)
af_line,   = ax_beam.plot([], [], color=COL_BEAM, lw=1.6)
af_true_v  = ax_beam.axvline(0, color=COL_TRUE, lw=1.3, ls='-',  label='True angle')
af_cmd_v   = ax_beam.axvline(0, color=COL_CMD,  lw=1.3, ls='--', label='Beam command')
ax_beam.axhline(-3, color='0.5', lw=0.8, ls=':')
ax_beam.text(2, -2.4, '-3 dB', fontsize=7, color='0.4')
ax_beam.set_title('R-RIS Array Factor  (16-element cut)')
ax_beam.set_xlabel('Angle from boresight  [deg]')
ax_beam.set_ylabel('Normalised  |AF|$^2$  [dB]')
ax_beam.set_xlim(0, 90); ax_beam.set_ylim(-35, 3)
ax_beam.legend(loc='lower center', fontsize=7, ncol=2)

# ---- 4c. Zoom panel ------------------------------------------------
ax_zoom.set_title('Tracking Zoom  (±0.4 m around target)')
ax_zoom.set_xlabel('x  [m]'); ax_zoom.set_ylabel('y  [m]')
ax_zoom.set_aspect('equal', adjustable='box')
z_true, = ax_zoom.plot([], [], 'o', ms=12, color=COL_TRUE, label='True')
z_meas, = ax_zoom.plot([], [], '.', ms=7, color='#bbbbbb', alpha=0.8, label='Raw meas.')
z_est,  = ax_zoom.plot([], [], 'x', ms=11, mew=2.4, color=COL_EST, label='Tracked est.')
ax_zoom.legend(loc='upper right', fontsize=7)

# ---- 4d. Angle-tracking time series --------------------------------
at_true, = ax_angle.plot([], [], '-',  lw=1.6, color=COL_TRUE, label=r'True $\theta$')
at_meas, = ax_angle.plot([], [], '.',  ms=4, color='#bbbbbb', label=r'Noisy $\hat\theta$')
at_cmd,  = ax_angle.plot([], [], '--', lw=1.6, color=COL_CMD, label=r'Beam cmd $\tilde\theta$')
ax_angle.set_title('Angle Tracking vs CPI Frame')
ax_angle.set_xlabel('CPI frame'); ax_angle.set_ylabel('Angle  [deg]')
ax_angle.legend(loc='upper right', fontsize=8, ncol=3)

# ---- 4e. Position-error history ------------------------------------
err_line, = ax_err.plot([], [], '-', lw=1.5, color=COL_BAD)
err_rmse  = ax_err.axhline(0, color=COL_GOOD, lw=1.2, ls='--', label='running RMSE')
err_p95   = ax_err.axhline(0, color=COL_EST,  lw=1.0, ls=':',  label='95th pct')
ax_err.set_title('Position Tracking Error vs CPI Frame')
ax_err.set_xlabel('CPI frame'); ax_err.set_ylabel('Position error  [cm]')
ax_err.legend(loc='upper right', fontsize=8)

# ---- 4f. Metrics text ----------------------------------------------
metrics_text = ax_info.text(0.0, 1.0, '', va='top', ha='left', fontsize=9.5,
                            family='monospace', transform=ax_info.transAxes)


# ------------------------------------------------------------------ #
#  5.  ANIMATION UPDATE                                              #
# ------------------------------------------------------------------ #
def draw_beam_cone(theta_cmd):
    """Update the shaded R-RIS beam wedge in the scene."""
    hp = hpbw_at(theta_cmd)
    length = 17.0
    edges = []
    for dth in (-hp / 2, 0.0, hp / 2):
        a = theta_cmd + dth
        edges.append([length * np.sin(a), length * np.cos(a)])
    beam_patch.set_xy([[0, 0]] + edges + [[0, 0]])
    beam_axis_line.set_data([0, length * np.sin(theta_cmd)],
                            [0, length * np.cos(theta_cmd)])


def update(_frame):
    if not state['running']:
        return []

    snap = tracker.step()
    tp, ep = tracker.true_pos, tracker.est_pos

    # -- scene --
    draw_beam_cone(tracker.theta_cmd)
    true_dot.set_data([tp[0]], [tp[1]])
    est_dot.set_data([ep[0]], [ep[1]])
    tr = np.array(tracker.hist_true_pos)
    true_trail.set_data(tr[:, 0], tr[:, 1])
    tx_ray.set_data([TX_POS[0], 0], [TX_POS[1], 0])
    rx_ray.set_data([tp[0], RX_POS[0]], [tp[1], RX_POS[1]])

    # -- array factor --
    af_db = 20 * np.log10(np.clip(array_factor(theta_grid, tracker.theta_cmd),
                                  1e-4, None))
    af_line.set_data(np.rad2deg(theta_grid), af_db)
    af_true_v.set_xdata([np.rad2deg(snap['th_true'])] * 2)
    af_cmd_v.set_xdata([np.rad2deg(tracker.theta_cmd)] * 2)

    # -- zoom --
    mp = snap['meas_pos']
    z_true.set_data([tp[0]], [tp[1]])
    z_meas.set_data([mp[0]], [mp[1]])
    z_est.set_data([ep[0]], [ep[1]])
    ax_zoom.set_xlim(tp[0] - 0.4, tp[0] + 0.4)
    ax_zoom.set_ylim(tp[1] - 0.4, tp[1] + 0.4)

    # -- angle time series --
    f = tracker.hist_frame
    at_true.set_data(f, tracker.hist_true_th)
    at_meas.set_data(f, tracker.hist_meas_th)
    at_cmd.set_data(f, tracker.hist_cmd_th)
    ax_angle.relim(); ax_angle.autoscale_view()

    # -- error history --
    err_line.set_data(f, tracker.hist_pos_err_cm)
    stats = tracker.running_stats()
    err_rmse.set_ydata([stats['rmse']] * 2)
    err_p95.set_ydata([stats['p95']] * 2)
    ax_err.relim(); ax_err.autoscale_view()
    ax_err.set_ylim(0, max(12, max(tracker.hist_pos_err_cm) * 1.15))

    # -- metrics --
    gain_col = COL_GOOD if snap['gain'] > 0.9 else COL_BAD
    metrics_text.set_text(
        f"CPI frame      : {tracker.frame:3d}\n"
        f"--------------------------------\n"
        f"True range     : {snap['r_true']:6.2f} m\n"
        f"True angle     : {np.rad2deg(snap['th_true']):6.2f} deg\n"
        f"Beam command   : {np.rad2deg(tracker.theta_cmd):6.2f} deg\n"
        f"Pointing error : {abs(np.rad2deg(snap['th_true']-tracker.theta_cmd)):6.3f} deg\n"
        f"--------------------------------\n"
        f"Beam gain      : {snap['gain']*100:6.2f} %\n"
        f"Position error : {snap['pos_err_cm']:6.2f} cm\n"
        f"--------------------------------\n"
        f"Running RMSE   : {stats['rmse']:6.2f} cm\n"
        f"95th-pct error : {stats['p95']:6.2f} cm\n"
        f"Mean beam gain : {stats['mean_gain']:6.2f} %\n"
        f"--------------------------------\n"
        f"Speed   : {tracker.speed:4.1f} m/s\n"
        f"Heading : {tracker.heading:4.0f} deg\n"
        f"alpha   : {tracker.alpha:4.2f}\n"
        f"SNR_CPI : {tracker.snr_db:4.1f} dB\n"
        f"Smoother: {'ON ' if tracker.use_smoother else 'OFF'}"
    )
    metrics_text.set_color('black')
    return []


# ------------------------------------------------------------------ #
#  6.  WIDGETS                                                        #
# ------------------------------------------------------------------ #
state = {'running': True}

ax_speed = fig.add_axes([0.08, 0.165, 0.34, 0.022])
ax_head  = fig.add_axes([0.08, 0.125, 0.34, 0.022])
ax_alpha = fig.add_axes([0.08, 0.085, 0.34, 0.022])
ax_snr   = fig.add_axes([0.08, 0.045, 0.34, 0.022])

s_speed = Slider(ax_speed, 'Speed [m/s]', 0.5, 6.0, valinit=SPEED_INIT, valstep=0.1)
s_head  = Slider(ax_head,  'Heading [deg]', 0.0, 90.0, valinit=HEADING_INIT, valstep=1.0)
s_alpha = Slider(ax_alpha, 'alpha (smoother)', 0.05, 1.0, valinit=ALPHA_INIT, valstep=0.05)
s_snr   = Slider(ax_snr,   'SNR_CPI [dB]', 0.0, 30.0, valinit=SNR_REF_DB, valstep=0.5)

def on_speed(v):  tracker.speed   = v
def on_head(v):   tracker.heading = v
def on_alpha(v):  tracker.alpha   = v
def on_snr(v):    tracker.snr_db  = v
s_speed.on_changed(on_speed)
s_head.on_changed(on_head)
s_alpha.on_changed(on_alpha)
s_snr.on_changed(on_snr)

ax_play  = fig.add_axes([0.55, 0.10, 0.09, 0.05])
ax_reset = fig.add_axes([0.66, 0.10, 0.09, 0.05])
b_play  = Button(ax_play,  'Play / Pause')
b_reset = Button(ax_reset, 'Reset')

def on_play(_event): state['running'] = not state['running']
def on_reset(_event):
    tracker.reset(); setup_scene_dynamic_reinit()
b_play.on_clicked(on_play)
b_reset.on_clicked(on_reset)

ax_check = fig.add_axes([0.78, 0.085, 0.15, 0.075])
chk = CheckButtons(ax_check, ['alpha-smoother'], [True])
def on_check(_label): tracker.use_smoother = not tracker.use_smoother
chk.on_clicked(on_check)

# A concise on-figure explanation of the algorithm
fig.text(0.55, 0.025,
         "Each CPI:  measure (noisy) -> predict 1 CPI ahead -> blend with gain alpha "
         "-> steer R-RIS beam -> evaluate gain at true angle.\n"
         "Try: drop SNR to see noise grow; lower alpha for smoother-but-laggier "
         "tracking; toggle the smoother off to see raw measurements drive the beam.",
         fontsize=8, color='#444', va='bottom')


def setup_scene_dynamic_reinit():
    """Re-attach dynamic artists after a full scene clear is NOT needed here;
    we keep the static scene and simply let update() refresh dynamic artists."""
    true_trail.set_data([], [])


# ------------------------------------------------------------------ #
#  7.  RUN                                                            #
# ------------------------------------------------------------------ #
anim = FuncAnimation(fig, update, interval=120, blit=False, cache_frame_data=False)

if __name__ == '__main__':
    print(__doc__)
    print(f"  lambda = {WAVELEN*1e3:.2f} mm | broadside HPBW = "
          f"{np.rad2deg(HPBW_BROADSIDE):.2f} deg | HPBW@60deg = "
          f"{np.rad2deg(hpbw_at(np.deg2rad(60))):.2f} deg")
    plt.show()
