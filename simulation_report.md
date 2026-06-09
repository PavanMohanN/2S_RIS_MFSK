# RIS M-FSK Sensing — Simulation Report
**Target journal:** IEEE Transactions on Vehicular Technology
**Generated:** 2026-05-30 13:22
**System:** Two-stage T-RIS/R-RIS at 28 GHz  |  M = 12 tones  |  Δf = 25 MHz  |  N = 256 elements

---

## System Configuration

| Parameter | Symbol | Value |
|---|---|---|
| Carrier frequency | fc | 28 GHz |
| Wavelength | λ | 10.71 mm |
| Array size (each stage) | N | 16 × 16 = 256 elements |
| Element spacing | d | λ/2 = 5.354 mm |
| M-FSK tones | M | 12 |
| Tone spacing | Δf | 25 MHz |
| Symbol period | Tsym | 1.40 μs |
| Symbol rate (two-stage) | fs | 714.3 kHz [DDS] |
| Symbol rate (single-stage) | fs | 13.0 kHz [SPI] |
| Speed-up factor | — | 55× |
| Sweeps per CPI | Nsw | 1024 |
| CPI duration | TCPI | 17.20 ms |
| Inter-stage gap (nominal) | dg | 2 mm (0.19λ) |
| Near-field coupling C0 | C0 | 0.3092 (-10.2 dB) |
| Single-symbol SNR | SNR₁ | -14.5 dB |
| Post-CPI SNR | SNR_CPI | 15.6 dB |

---

## Phase 1 — PIN Diode Device Physics

| Metric | Value |
|---|---|
| Carrier frequency | 28 GHz |
| Series resistance Rs | 2 Ω |
| Junction capacitance CJ | 25 fF |
| ON-state junction resistance RJ_on | 1 Ω |
| OFF-state junction resistance RJ_off | 8 kΩ |
| Switching energy Esw | 10 pJ |

---

## Phase 2 — M-FSK Waveform

| Metric | Value |
|---|---|
| BPF passband | 28.0125 – 28.3125 GHz |
| Signal bandwidth | 300 MHz |
| Noise bandwidth | 312.5 MHz |
| SPI reprogramming time | 76.8 μs |

---

## Phase 3 — Bistatic Radar Signal Processing

| Metric | Value |
|---|---|
| Tx power | 20 dBm (100 mW) |
| Tx → RIS distance | 8 m |
| RIS → target distance | 10 m |
| Target → Rx distance | 8 m |
| Target RCS | 0 dBsm |
| Range resolution ΔR | 1.00 m |
| Max unambiguous range | 12.0 m |
| Detected range (sim.) | 9.99 m (true: 10.00 m, error: 7 mm) |
| Detected velocity | 2.49 m/s (true: 2.00 m/s, within 1 bin) |

---

## Phase 4 — CRB + Monte Carlo Estimation

| Metric | Value |
|---|---|
| Effective post-CPI SNR | 15.6 dB |
| CRB range σ_R | 1.32 cm |
| CRB velocity σ_v | 0.78 cm/s |
| CRB 2-D position σ_pos | 3.36 cm |
| MC RMSE range (1000 trials) | 7.14 cm |
| MC RMSE velocity | 56.87 cm/s (0.46 FFT bins) |
| MC RMSE position | 9.49 cm |
| FFT velocity bin width Δv | 31.12 cm/s |
| MC trials | 1000 |
| SNR sweep | -30 to 20 dB |

---

## Phase 5 — Closed-Loop Beam Tracking

| Metric | Value |
|---|---|
| Initial position | (8.66, 5.00) m |
| Initial range R₀ | 10.0 m |
| Initial angle θ₀ | 60° |
| Target velocity | 2.0 m/s at 45° |
| CPI frames | 50 |
| Total tracking time | 860.2 ms |
| α-smoother coefficient | 0.3 |
| RMSE position | 5.78 cm |
| RMSE range | 3.26 cm |
| 95th-pct position error | 9.18 cm |
| Mean beam gain G_eff | 99.87% |

---

## Phase 6 — Multi-Target + Benchmarking

### Multi-target comparison
| Target | True range | FMCW apparent | FMCW displacement |
|---|---|---|---|
| T1 (static) | 4.0 m | 4.0 m | 0.0 m |
| T2 (v = +3 m/s) | 7.0 m | 11.82 m | +4.82 m |
| T3 (v = −2 m/s) | 10.0 m | 6.79 m | −3.21 m |

FMCW displacement coefficient: 1.606 m per m/s

### Power comparison
| Architecture | Total power | Symbol rate | ΔR |
|---|---|---|---|
| Two-stage T-RIS/R-RIS | 2.33 W | 714 kHz | 1.0 m |
| Single-stage STM-RIS | 2.20 W | 13 kHz | 1.0 m |
| FMCW radar | 2.47 W | 20 kHz | 0.15 m |
| AESA phased array | 6.56 W | 50 kHz | 0.15 m |

Power saving vs AESA: **−65%**

---

## Phase 7 — Robustness and Sensitivity

| Impairment | Nominal | SNR loss | Tolerance for <1 dB |
|---|---|---|---|
| Amplitude error σ_A | 2% | 0.000 dB | ≫ 20% (negligible) |
| Phase error σ_φ | 3° | 0.012 dB | 27.5° (9× margin) |
| Gap deviation Δdg | 0 mm | 0.00 dB | ±0.196 mm (**tight**) |
| Quantisation B=3 bits | — | 0.224 dB | B ≥ 2 bits |
| **Total budget (nominal)** | — | **0.236 dB** | — |

Gap sensitivity: **5.10 dB per mm** (critical manufacturing spec)

---

## Figure and Table Index

### Figures (16 total)
| ✓ | Ph.1 | `F4_transmission_amplitude` | T-RIS PIN-diode transmission amplitude |T_n| vs frequency. On-state (V_f > 0… |
| ✓ | Ph.1 | `F5_phase_response` | T-RIS PIN-diode phase response \angle T_n(f) in the on and off states. Phase d… |
| ✓ | Ph.1 | `F6_impedance_magnitude` | PIN-diode impedance magnitude |Z_n(f)| for forward and reverse bias. |
| ✓ | Ph.2 | `F7_mfsk_spectrum` | M-FSK tone spectrum at T-RIS output. Twelve sidebands at f_c + kΔ f, k=1… |
| ✓ | Ph.2 | `F8_mfsk_spectrogram` | Time--frequency spectrogram of the T-RIS M-FSK waveform. Each row shows one symb… |
| ✓ | Ph.3 | `F10_link_budget_waterfall` | Bistatic link budget waterfall. Per-stage contributions (bars) and cumulative si… |
| ✓ | Ph.3 | `F11_range_doppler` | (a) IFFT range profile from N_{ sw}=1024 sweeps: target peak at 9.99 m (tru… |
| ✓ | Ph.4 | `F12_rmse_range_position` | (a) Range RMSE and CRB vs single-symbol SNR. (b) 2-D position RMSE and CRB. Vert… |
| ✓ | Ph.4 | `F13_rmse_velocity` | Velocity RMSE, CRB, and FFT estimation floor vs SNR. Three regimes: noise-domina… |
| ✓ | Ph.5 | `F14_tracking_trajectory` | (a) Closed-loop beam-tracking scene over 50 CPIs (T_{ tot}=860 ms). Beam-st… |
| ✓ | Ph.5 | `F15_tracking_errors` | (a) Position-error scatter: 50th-percentile circle = 5.3 cm, 95th = 9.2 cm. … |
| ✓ | Ph.6 | `F16_multitarget_vs_fmcw` | Multi-target range profiles. (a) M-FSK IFFT: three clean peaks at true ranges 4,… |
| ✓ | Ph.6 | `F17_power_comparison` | Stacked power breakdown for four sensing architectures. The two-stage T-RIS/R-RI… |
| ✓ | Ph.6 | `F18_spider_chart` | Normalised performance spider chart (five axes). The proposed two-stage architec… |
| ✓ | Ph.7 | `FS1_element_tolerance` | Element-level manufacturing tolerance robustness. (a) Beam-gain change vs amplit… |
| ✓ | Ph.7 | `FS2_gap_and_quantisation` | Hardware sensitivity. (a) SNR_{ CPI} and CRB_R vs inter-stage gap d_g: … |

### Tables (10 total)
| ✓ | Ph.1 | `T1_system_params` | System Parameters |
| ✓ | Ph.1 | `T2_ris_params` | RIS Parameters |
| ✓ | Ph.5 | `T3_tracking_params` | Tracking Scenario and Performance |
| ✓ | Ph.6 | `T4_two_stage_vs_single` | Two-Stage vs Single-Stage Comparison |
| ✓ | Ph.6 | `T5_literature_comparison` | Prior RIS Sensing Literature Comparison |
| ✓ | Ph.6 | `T6_complexity_comparison` | Computational Complexity Comparison |
| ✓ | Ph.4 | `T7_estimation_performance` | Estimation Performance at Operating Point |
| ✓ | Ph.6 | `T8_power_budget` | Detailed Power Budget |
| ✓ | Ph.7 | `T9_sensitivity_analysis` | Sensitivity Analysis |
| ✓ | Ph.7 | `T10_tolerance_budget` | Tolerance Budget |

---

## Reproduction Instructions

```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run all phases
python main.py

# Run individual phases
python main.py --phases 1 2 3 4 5 6 7 8

# Run MATLAB figures (requires MATLAB R2021a+)
cd matlab && matlab -batch "run_all_matlab"
```

---
*Generated by `figures_tables.py` — RIS M-FSK Sensing Simulation Suite*
*IEEE TVT submission — 2026-05-30 13:22*
