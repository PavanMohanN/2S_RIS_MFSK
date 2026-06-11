"""
config.py — RIS M-FSK Sensing: Central Configuration
=====================================================
Establishes the centralized physical, hardware, and operational parameter
bounds for a two-stage transmitter-RIS (T-RIS) and receiver-RIS (R-RIS)
bistatic multi-frequency shift keying (M-FSK) radar sensing suite.

System Parameters Established
-----------------------------
  * Electromagnetic Boundaries   : Definitive specifications for the 28 GHz carrier, 
                                   wavenumber vector properties, and free-space wavelength.
  * PIN Diode Equivalent Circuit : Physical forward and reverse junction resistance constraints, 
                                   switching energy thresholds, and parasitic contact boundaries.
  * Array & Aperture Metrics     : Structural layout constraints for a 256-element half-wavelength 
                                   spatially organized uniform planar array.
  * M-FSK Radar Signal Waveform  : Waveform temporal partitions including multi-tone offsets, 
                                   symbol blocks, block sweep patterns, and serialization clock caps.
  * Inter-Stage Evanescent Gap   : Mechanical spacing boundaries determining near-field evanescent 
                                   wave decay coefficients and coupling limits.
  * Radar Tracking Framework     : Geometric node allocations, target kinematics, tracking frame caps, 
                                   and feedback filtering constants.
  * Numerical & Graphic Standards : Statistical parameters for Monte Carlo evaluation loops alongside 
                                   normalized vector color parameters.

Dependencies
------------
  * Imported by all functional processing blocks within the simulation chain to enforce 
    system-wide variable consistency.

Execution Parameters
--------------------
    python config.py
    from config import print_config_summary
"""

import numpy as np
import os

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Physical constants
# ═══════════════════════════════════════════════════════════════════════════════
C_LIGHT  = 2.998e8          # Speed of light            [m/s]
K_BOLTZ  = 1.380649e-23     # Boltzmann constant         [J/K]
T0       = 290.0            # Standard noise temperature [K]

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Carrier frequency
# ═══════════════════════════════════════════════════════════════════════════════
FC         = 28.0e9                 # Carrier frequency  [Hz]
WAVELENGTH = C_LIGHT / FC           # Free-space λ ≈ 10.71 mm  [m]
K_C        = 2.0 * np.pi / WAVELENGTH  # Free-space wavenumber  [rad/m]

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PIN diode equivalent-circuit parameters (28 GHz)
#     Source: Pozar (2011) [13], Caverly & Hiller (1989) [15],
#             Shen et al. (2025) [12]
# ═══════════════════════════════════════════════════════════════════════════════
RS      = 2.0           # Series resistance (bond-wire + contact)  [Ω]
CJ      = 25.0e-15      # Junction capacitance                      [F]  = 25 fF
RJ_ON   = 1.0           # Forward-bias junction resistance           [Ω]
RJ_OFF  = 8.0e3         # Reverse-bias junction resistance           [Ω]  = 8 kΩ
Z0      = 50.0          # Feed-line characteristic impedance         [Ω]
V_BIAS  = 0.8           # PIN diode switching voltage                [V]
ESW     = 10.0e-12      # Switching energy per event (Cj=30fF, V=0.8V) [J]

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  T-RIS / R-RIS array parameters  (TABLE II)
# ═══════════════════════════════════════════════════════════════════════════════
N_ELEM_X    = 16                            # Elements along x (each stage)
N_ELEM_Y    = 16                            # Elements along y (each stage)
N_ELEM      = N_ELEM_X * N_ELEM_Y          # Total elements per stage = 256
D_ELEM      = WAVELENGTH / 2               # Element spacing [m] ≈ 5.357 mm
PHASE_BITS  = 3                             # R-RIS phase quantisation bits (default)
GAMMA_DB    = -1.0                          # R-RIS reflection loss         [dB]
GAMMA_MAG   = 10.0 ** (GAMMA_DB / 20.0)    # R-RIS reflection magnitude    (≈ 0.891)

# R-RIS physical aperture dimensions
ARIS_X = N_ELEM_X * D_ELEM                 # Aperture width  [m]
ARIS_Y = N_ELEM_Y * D_ELEM                 # Aperture height [m]
ARIS   = ARIS_X * ARIS_Y                   # Physical aperture area [m²] ≈ 73.5 cm²

# Manufacturing tolerances (element-to-element Gaussian spread)
SIGMA_AMP_FRAC = 0.02                       # Amplitude variation  (2 % of mean)
SIGMA_PHASE_RAD = np.deg2rad(3.0)           # Phase variation      (3°)

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  M-FSK waveform parameters  (TABLE I / TABLE II)
# ═══════════════════════════════════════════════════════════════════════════════
M_TONES    = 12                                          # Number of FSK tones
DELTA_F    = 25.0e6                                      # Tone spacing          [Hz]
FM_OFFSETS = np.arange(1, M_TONES + 1) * DELTA_F        # {25,50,…,300} MHz above fc

TSYM       = 1.40e-6                                     # Symbol period         [s] ≈ 1.40 μs
F_SYMBOL   = 1.0 / TSYM                                  # Symbol rate           [Hz] ≈ 714 kHz
TSW        = M_TONES * TSYM                              # Sweep duration        [s] ≈ 16.8 μs
NSw        = 1024                                        # Sweeps per CPI
TCPI       = NSw * TSW                                   # Coherent processing interval [s]

BW_MFSK    = M_TONES * DELTA_F                          # M-FSK signal bandwidth ≈ 300 MHz [Hz]
BW_NOISE   = (M_TONES + 0.5) * DELTA_F                  # Noise bandwidth ≈ 312.5 MHz [Hz]

# BPF passband (5th-order Butterworth centred on upper sideband band)
BPF_LO_HZ  = FC + 12.5e6                                # BPF lower edge [Hz]
BPF_HI_HZ  = FC + 312.5e6                               # BPF upper edge [Hz]

# SPI symbol rate bottleneck (single-stage limitation)
N_BITS_SPI = N_ELEM * PHASE_BITS                        # Bits per reprogramming event
F_SPI_CLK  = 10.0e6                                     # SPI clock frequency [Hz]
T_SPI_REPROGRAM = N_BITS_SPI / F_SPI_CLK               # Reprogramming time [s] ≈ 76.8 μs
F_SYMBOL_SINGLE_STAGE = 1.0 / T_SPI_REPROGRAM          # Single-stage symbol rate [Hz] ≈ 13 kHz

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Inter-stage gap and near-field coupling  (Eq. 11)
# ═══════════════════════════════════════════════════════════════════════════════
D_GAP_NOM   = 2.0e-3                        # Nominal inter-stage gap        [m] = 2 mm
ALPHA_EV    = 2.0 * np.pi / WAVELENGTH      # Evanescent decay constant      [1/m]
C0_NOM      = np.exp(-ALPHA_EV * D_GAP_NOM) # Coupling amplitude at nom. gap (≈ 0.308)
C0_NOM_DB   = 20.0 * np.log10(C0_NOM)       # Coupling loss at nominal gap   [dB] ≈ −10.2 dB

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Bistatic radar geometry  (TABLE I, Fig. 1)
# ═══════════════════════════════════════════════════════════════════════════════
PTX_DBM    = 20.0                            # Tx radiated power              [dBm]
PTX_W      = 10.0 ** ((PTX_DBM - 30.0) / 10.0)  # Tx power                  [W] = 0.1 W
GTX_DB     = 0.0                             # Tx antenna gain                [dBi]
GTX        = 10.0 ** (GTX_DB / 10.0)        # Tx antenna gain (linear)

SIGMA_RCS_DBSM = 0.0                         # Target RCS                    [dBsm]
SIGMA_RCS      = 10.0 ** (SIGMA_RCS_DBSM / 10.0)  # Target RCS              [m²] = 1 m²

D_TX_RIS   = 8.0                             # Tx → RIS distance             [m]
THETA1_DEG = 60.0                            # RIS → target azimuth          [°]
THETA1_RAD = np.deg2rad(THETA1_DEG)          # RIS → target azimuth          [rad]
R_RIS_T    = 10.0                            # RIS → target distance         [m]
R_T_RX     = 8.0                             # Target → Rx distance          [m]

# Rx array
NRX        = 16                              # Rx array elements
DRX        = WAVELENGTH / 2                  # Rx element spacing             [m]
GRX_DB     = 18.0                            # Rx gain                       [dBi]
GRX        = 10.0 ** (GRX_DB / 10.0)
NF_DB      = 5.0                             # Rx noise figure               [dB]
NF         = 10.0 ** (NF_DB / 10.0)

# Bistatic scene positions (RIS at origin)
P_RIS    = np.array([0.0, 0.0])
P_TARGET = np.array([R_RIS_T * np.sin(THETA1_RAD),
                      R_RIS_T * np.cos(THETA1_RAD)])   # ≈ (8.66, 5.0) m
P_RX     = np.array([10.46, 7.54])                      # Rx position [m]  (β = 165°)

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Target tracking parameters  (TABLE III)
# ═══════════════════════════════════════════════════════════════════════════════
V_TARGET   = 2.0                             # Target speed                   [m/s]
THETA_T_DEG = 45.0                           # Initial target heading angle   [°]
THETA_T_RAD = np.deg2rad(THETA_T_DEG)
N_FRAMES   = 50                              # Tracking frames (CPIs)

# Initial velocity vector
V_VEC = V_TARGET * np.array([np.cos(THETA_T_RAD), np.sin(THETA_T_RAD)])

ALPHA_SMOOTH = 0.3                           # α-smoother coefficient for beam tracking

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Link budget — derived quantities  (Section IV-B)
# ═══════════════════════════════════════════════════════════════════════════════
# These are computed analytically; exact values confirmed by bistatic_radar.py.
SNR_SINGLE_DB = -14.5   # Single-symbol SNR at operating point                [dB]
SNR_CPI_DB    = +15.6   # Post-CPI SNR (10·log10(M·Nsw) ≈ +30.1 dB gain)    [dB]
NOISE_FLOOR_DBM = -84.6 # Thermal noise floor (B=275MHz, NF=5dB, T=290K)     [dBm]

# R-RIS coherent beam gain (N²·d²/λ² × cos²(θ1) approximation):
GRIS_DB = 10.0 * np.log10(N_ELEM ** 2 * D_ELEM ** 2 * np.cos(THETA1_RAD) ** 2
                           / WAVELENGTH ** 2)           # ≈ 48.2 dBi

# ═══════════════════════════════════════════════════════════════════════════════
# 10. AESA reference (power benchmarking, Section V-A)
# ═══════════════════════════════════════════════════════════════════════════════
P_CHAN_AESA_W = 15.0e-3   # Power per AESA element chain [W] = 15 mW
P_RF_AESA_W   = 0.87      # AESA master LO + PA power    [W]
P_AESA_TOTAL  = N_ELEM * P_CHAN_AESA_W + P_RF_AESA_W    # ≈ 4.71 W

# T-RIS / R-RIS power budget (Eq. 51–52)
P_TX_W        = PTX_W                        # CW Tx power
P_BIAS_ELEM_W = 1.5e-3                       # Per-element bias (∼ 6 mA @ 0.25 V ON)  [W]
P_BIAS_TRIS_W = N_ELEM * P_BIAS_ELEM_W       # T-RIS total bias                       [W]
P_BIAS_RRIS_W = N_ELEM * P_BIAS_ELEM_W       # R-RIS total bias                       [W]
P_DYN_BASE_W  = N_ELEM * DELTA_F * ESW       # Dynamic switching power (per tone, base) [W]

# ═══════════════════════════════════════════════════════════════════════════════
# 11. Monte Carlo simulation settings
# ═══════════════════════════════════════════════════════════════════════════════
MC_TRIALS     = 1000        # Monte Carlo realisations per SNR point
SNR_SWEEP_DB  = np.arange(-30, 21, 2)   # SNR sweep range (single-symbol) [dB]
RANDOM_SEED   = 42          # For reproducibility


FIG_WIDTH_1COL  = 3.5       # Single-column width  [inches]
FIG_WIDTH_2COL  = 7.16      # Double-column width  [inches]
FIG_HEIGHT_STD  = 2.6       # Standard height      [inches]
FIG_HEIGHT_TALL = 3.4       # Taller panels        [inches]
FIG_DPI         = 300       # Output DPI

FIG_FONT        = 8         # Axis tick labels     [pt]
FIG_LABEL       = 9         # Axis labels          [pt]
FIG_LEGEND      = 7         # Legend entries       [pt]
FIG_TITLE       = 9         # Panel titles         [pt]
FIG_LW          = 1.5       # Default line width   [pt]
FIG_MS          = 4         # Default marker size  [pt]


COLORS = {
    'on':        '#1a6faf',   # Deep blue    — on-state / two-stage
    'off':       '#c1272d',   # Deep red     — off-state / image
    'sideband':  '#2e8b57',   # Dark green   — sideband / desired
    'carrier':   '#d97706',   # Amber        — carrier feedthrough
    'two_stage': '#1a6faf',   # Blue         — proposed two-stage
    'single':    '#c1272d',   # Red          — single-stage STM-RIS
    'fmcw':      '#2e8b57',   # Green        — FMCW radar
    'aesa':      '#d97706',   # Amber        — AESA radar
    'crb':       '#6b4ea0',   # Purple       — CRB curve
    'target1':   '#1a6faf',   # Blue         — target 1
    'target2':   '#c1272d',   # Red          — target 2
    'target3':   '#2e8b57',   # Green        — target 3
    'gray':      '#555555',   # Gray         — reference lines
}

LINESTYLES = ['-', '--', '-.', ':']   # For multi-curve plots

# ═══════════════════════════════════════════════════════════════════════════════
# 13. Directory structure
# ═══════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = 'figures'
TABLE_DIR  = 'tables'
DATA_DIR   = 'data'
MATLAB_DIR = 'matlab'

for _d in [OUTPUT_DIR, TABLE_DIR, DATA_DIR, MATLAB_DIR]:
    os.makedirs(_d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 14. Derived consistency checks  (printed on import if run directly)
# ═══════════════════════════════════════════════════════════════════════════════
def print_config_summary():
    """Print a concise system parameter summary for verification."""
    print("\n" + "═" * 60)
    print("  RIS M-FSK Sensing — Configuration Summary")

    print("═" * 60)
    print(f"  Carrier frequency fc         = {FC/1e9:.1f} GHz")
    print(f"  Wavelength λ                 = {WAVELENGTH*1e3:.2f} mm")
    print(f"  Array: {N_ELEM_X}×{N_ELEM_Y} = {N_ELEM} elements, d = λ/2")
    print(f"  Physical aperture ARIS       = {ARIS*1e4:.1f} cm²")
    print(f"  M-FSK tones M                = {M_TONES}, Δf = {DELTA_F/1e6:.0f} MHz")
    print(f"  Tone range                   = {FM_OFFSETS[0]/1e6:.0f}–{FM_OFFSETS[-1]/1e6:.0f} MHz above fc")
    print(f"  Symbol period Tsym           = {TSYM*1e6:.2f} μs")
    print(f"  Symbol rate (two-stage)      = {F_SYMBOL/1e3:.1f} kHz  [DDS-limited]")
    print(f"  Symbol rate (single-stage)   = {F_SYMBOL_SINGLE_STAGE/1e3:.1f} kHz  [SPI-limited]")
    print(f"  Speed-up factor              = {F_SYMBOL/F_SYMBOL_SINGLE_STAGE:.0f}×")
    print(f"  Sweep duration Tsw           = {TSW*1e6:.1f} μs")
    print(f"  CPI duration TCPI            = {TCPI*1e3:.1f} ms  ({NSw} sweeps)")
    print(f"  Nominal inter-stage gap dg   = {D_GAP_NOM*1e3:.0f} mm  ({D_GAP_NOM/WAVELENGTH:.2f}λ)")
    print(f"  Near-field coupling C0       = {C0_NOM:.4f}  ({C0_NOM_DB:.1f} dB)")
    print(f"  Operating SNR (single sym)   = {SNR_SINGLE_DB:.1f} dB")
    print(f"  Post-CPI SNR                 = {SNR_CPI_DB:.1f} dB")
    print(f"  RIS beam gain GRIS           = {GRIS_DB:.1f} dBi")
    print(f"  AESA total power             = {P_AESA_TOTAL:.2f} W")
    print("═" * 60 + "\n")


if __name__ == '__main__':
    print_config_summary()
