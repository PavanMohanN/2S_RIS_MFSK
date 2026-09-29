# A Two-Stage Reconfigurable Intelligent Surface Architecture for Decoupled M-FSK Modulation and Beamforming in 28-GHz Bistatic Sensing

Simulation code, data and figure scripts for the paper

> J. Lota, P. M. Neelamraju, W. Whittow, A. Demosthenous, and A. Bansal,
> "A Two-Stage Reconfigurable Intelligent Surface Architecture for Decoupled M-FSK Modulation and Beamforming in 28-GHz Bistatic Sensing," submitted to *IEEE Transactions on Vehicular Technology*.

A transmissive RIS (T-RIS), driven by a single direct digital synthesizer,
modulates a 28 GHz continuous-wave carrier into stepped M-FSK tones; a
reflective RIS (R-RIS), updated once per coherent processing interval, steers
all tones toward the target; a 16-element bistatic receiver estimates the
bistatic path, velocity and angle, and tracks the target.

Every figure in the paper is produced by its own Python file in `figures/`,
and every number in the paper comes from the experiments in `experiments/`.

---

## Quick start

Python 3.10 or newer.

```bash
pip install -r requirements.txt
python run_all.py            # all experiments + all figures (paper results)
```

| Command | What it does | Time* |
|---|---|---|
| `python run_all.py` | Runs the seven experiments with the paper's Monte Carlo sizes, then every figure script | ~10 min |
| `python run_all.py --quick` | Reduced trial counts, for a smoke test | ~2 min |
| `python run_all.py --figures-only` | Re-plots all figures from the stored data in `results/figdata/` | ~30 s |
| `python run_all.py --pdf` | Additionally compiles `paper/` with `pdflatex` | +30 s |

\*Single laptop core. `--quick` results differ from the paper's; run without
it to reproduce the paper exactly.

---

## Repository layout

```
.
├── run_all.py              reproduce everything
├── requirements.txt
├── mfsk/
│   ├── common.py           system parameters, signal model, PIN-diode model,
│   │                       exact CRBs, geometry, bistatic inversion, plot style
│   └── figio.py            figure data storage and re-plotting
├── experiments/            the seven simulation studies (compute + validate)
├── figures/                one script per paper figure (plot only)
├── results/
│   ├── figdata/            exact inputs of every figure (.json + .npz)
│   └── exp0N_*/            per-experiment data, LaTeX tables, extra figures
└── paper/
    ├── RIS_MFSK_TVT.tex                 manuscript
    ├── RIS_MFSK_TVT_supplementary.tex   supplementary material
    ├── IEEEtran.cls
    └── figures/            the figures, named exactly as in \includegraphics
```

---

## Figures

Each script writes `paper/figures/<name>.pdf` (and a `.png` preview), where
`<name>` is the file name used by `\includegraphics` in the manuscript.

| Figure | File | Script | Data |
|---|---|---|---|
| Fig. 1a | `F1a_scene.pdf` | `figures/fig01a_scene.py` | computed in the script |
| Fig. 1b | `F1b_process_flow.pdf` | `figures/fig01b_process_flow.py` | computed in the script |
| Fig. 2 | `F4rev_multitone_beam.pdf` | `figures/fig02_multitone_beam.py` | experiments/exp03_beam_squint.py |
| Fig. 3 | `F3rev_baseband_spectrum.pdf` | `figures/fig03_baseband_spectrum.py` | experiments/exp01_receiver_sideband.py |
| Fig. 4a | `F2a_transmission_amplitude.pdf` | `figures/fig04a_transmission_amplitude.py` | computed in the script (PIN-diode model) |
| Fig. 4b | `F2b_phase_response.pdf` | `figures/fig04b_phase_response.py` | computed in the script (PIN-diode model) |
| Fig. 4c | `F2c_impedance_magnitude.pdf` | `figures/fig04c_impedance_magnitude.py` | computed in the script (PIN-diode model) |
| Fig. 5 | `F5rev_link_budget.pdf` | `figures/fig05_link_budget.py` | computed in the script (+ exp04 results) |
| Fig. 6 | `F6rev_range_doppler.pdf` | `figures/fig06_range_doppler.py` | computed in the script |
| Fig. 7 | `F7rev_rmse_path_position.pdf` | `figures/fig07_rmse_path_position.py` | experiments/exp02_estimation_crb.py |
| Fig. 8 | `F8rev_rmse_velocity.pdf` | `figures/fig08_rmse_velocity.py` | experiments/exp02_estimation_crb.py |
| Fig. 9 | `FA_estimator_efficiency.pdf` | `figures/fig09_estimator_efficiency.py` | experiments/exp02_estimation_crb.py |
| Fig. 10 | `F9rev_tracking_cv.pdf` | `figures/fig10_tracking_cv.py` | experiments/exp05_tracking.py |
| Fig. 11 | `FG_tracking_manoeuvre.pdf` | `figures/fig11_tracking_manoeuvre.py` | experiments/exp05_tracking.py |
| Fig. 12 | `F11rev_fmcw_fair.pdf` | `figures/fig12_fmcw_fair.py` | experiments/exp07_benchmarks.py |
| Fig. 13 | `F12rev_power.pdf` | `figures/fig13_power.py` | experiments/exp07_benchmarks.py |
| Fig. 14 | `FH_interlayer_coupling.pdf` | `figures/fig14_interlayer_coupling.py` | experiments/exp04_hardware_coupling.py |
| Fig. 15 | `FF_phase_noise.pdf` | `figures/fig15_phase_noise.py` | experiments/exp04_hardware_coupling.py |
| Fig. 16 | `FI_rx_aperture.pdf` | `figures/fig16_rx_aperture.py` | experiments/exp06_rx_aperture.py |
| Suppl. Fig. S1 | `FC_sideband_separation.pdf` | `figures/figS01_sideband_separation.py` | experiments/exp01_receiver_sideband.py |

**To reproduce or edit one figure**, run its script, for example

```bash
python figures/fig07_rmse_path_position.py
```

Scripts marked "experiments/..." re-plot from `results/figdata/<name>.json`
and `.npz`, which the experiment writes each time it runs. Editing the plotting
code and re-running the script therefore changes the figure without re-running
the simulation. The other scripts compute their data directly in a few
seconds. The Fig. 5 script reads the inter-stage coupling computed by
`experiments/exp04_hardware_coupling.py`.

---

## Experiments

| Script | Content | Paper | Checks |
|---|---|---|---|
| `exp01_receiver_sideband.py` | Complex-baseband receiver, image-rejection ratio versus I/Q imbalance (analytic and time-domain), modulator bandwidth and tone plans | Sec. II-E, Fig. 3, Suppl. Fig. S1, Table S1 | 7 |
| `exp02_estimation_crb.py` | Estimators versus exact CRBs for path, velocity, angle and position; efficiency decomposition; zero-padding; hardware mechanisms | Secs. III, IV-B, IV-C, Figs. 7-9 | 8 |
| `exp03_beam_squint.py` | Exact squint of every tone, per-tone gain, CRB increase and bias, validity map | Sec. II-C, Fig. 2, Suppl. Table S2 | 6 |
| `exp04_hardware_coupling.py` | Element and oscillator phase noise; Rayleigh-Sommerfeld inter-stage coupling and assembly tolerances | Secs. V-B, V-C, Figs. 14, 15 | 9 |
| `exp05_tracking.py` | Closed-loop tracking (alpha-smoother, alpha-beta, CV/CA EKF, IMM) on straight, turning and accelerating trajectories | Secs. III-E, IV-D, Figs. 10, 11 | 7 |
| `exp06_rx_aperture.py` | Position accuracy versus receive array size; far-field versus range-focused scanning | Sec. V-D, Fig. 16, Suppl. Table S3 | 6 |
| `exp07_benchmarks.py` | Single-stage architectures, M-FSK versus FMCW at equal CPI, transmit-side power | Sec. V-A, Figs. 12, 13, Suppl. Table S4 | 6 |

Every experiment ends with self-validation checks (for example, time-domain
simulation against the analytic image-rejection ratio, Monte Carlo against the
CRB, the angular-spectrum model against its fundamental-mode limit) and exits
with a non-zero code if any fails. Useful options:

```bash
python experiments/exp02_estimation_crb.py --quick      # fewer trials
python experiments/exp02_estimation_crb.py --snr1 -10   # another operating SNR (dB)
python experiments/exp02_estimation_crb.py --beta 90    # another bistatic angle (deg)
```

---

## Reproducibility

- All random streams are seeded, so a run reproduces the paper's numbers
  exactly on the same NumPy version.
- System parameters are defined once, in `mfsk/common.py`; the operating point
  is SNR<sub>1</sub> = -14.5 dB at a 60° bistatic angle.
- Each experiment writes its full console log to
  `results/exp0N_*/data/exp0N_summary.txt` and its results to
  `exp0N_results.json`.

`Created in Aug 2026`

`@author: Pavan Mohan Neelamraju`

`Affiliation: Loughborough University`

**Email**: npavanmohan3@gmail.com

**Personal Website 🔴🔵**: [https://pavanmohan.netlify.app/](https://pavanmohan.netlify.app/)   


---
