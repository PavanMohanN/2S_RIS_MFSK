<img width="1918" height="1077" alt="image" src="https://github.com/user-attachments/assets/14c4fc7a-235e-4a3c-921c-4450524210c0" />

# About the Project

Welcome to the **RIS M-FSK Sensing Simulation Suite** repository. This project provides a mathematically rigorous, self-contained **Monte Carlo numerical physics simulator** designed to model an advanced **28 GHz two-stage Reconfigurable Intelligent Surface (RIS)** bistatic radar framework.

If you are new to the project, this guide will walk you through the core physics, the codebase layout, setup requirements, and execution steps needed to understand and extend this system.

---

## 1. What We Are Simulating: Core Engineering Principles

This suite simulates a closed-loop, physical radar sensing chain that leverages a novel **two-stage RIS architecture** to decouple beam-steering and radar waveform modulation at millimeter-wave (mmWave) frequencies.

### The Problem with Conventional Single-Stage RIS

In classic Space-Time-Modulated RIS (STM-RIS) systems, a single panel must handle both beamforming (pointing a narrow beam at a target) and fast waveform modulation (switching phases to create distinct radar tones). This creates a severe hardware bottleneck: sending high-speed phase updates from a baseband processor to hundreds of array elements over serial interfaces like SPI chokes the system speed, limiting the radar symbol rate to roughly **13 kHz**.

### Our Solution: The Two-Stage Decoupled System

This repository simulates a system that splits these functions across two separate layers spaced an evanescent wave's distance apart:

```
  [ Continuous-Wave Transmitter ] ── (8m) ──> [ T-RIS Layer ]
                                                    │  (2mm Near-Field Gap)
                                                    ▼
  [ Isotropic Rx Array ] <── (8m) ── [ R-RIS Layer ] <── (10m) ── [ Target ]

```

1. **Transmitter-RIS (T-RIS):** Dedicated exclusively to high-speed waveform modulation. Instead of serial control, it uses a fast, localized Direct Digital Synthesis (DDS) internal clock boundary to step through reflection configurations at an accelerated symbol rate of **714.3 kHz** (a **55× speedup** over the serial baseline).
2. **Receiver-RIS (R-RIS):** Dedicated exclusively to spatial beamforming and target tracking. Because target positions change slowly relative to a radar pulse, this stage only updates its narrow beam commands once per Coherent Processing Interval (CPI), completely freeing it from high-speed switching requirements.

### The Radar Waveform: M-FSK vs. FMCW

The simulator builds a **Multi-Frequency Shift Keying (M-FSK)** waveform.

* **Orthogonality:** The T-RIS transmits $M=12$ discrete, sequential continuous-wave tones spaced exactly $\Delta f = 25\text{ MHz}$ apart.
* **Decoupling Advantage:** For a moving target, standard linear Frequency Modulated Continuous Wave (FMCW) chirps suffer from *range-Doppler coupling*, where target velocity shifts the received beat frequency and creates misleading spatial displacement or "ghost targets". In contrast, M-FSK isolates target range within the tone-to-tone phase variations, while tracking Doppler entirely in the sweep-to-sweep temporal domain—completely eliminating ghost targets.

---

## 2. Comprehensive Codebase Manifest

The simulator is highly structured and modular. Each file represents a dedicated physics, waveform, tracking, or benchmarking block:

### Infrastructure & Orchestration

* **`config.py`**
* **What it does:** The single source of truth for all physical, mechanical, geometric, and graphical parameters. Holds physical constants, carrier parameters, PIN diode equivalent circuits, array dimensions, signal boundaries, target trajectories, and plotting style definitions.


* **`main.py`**
* **What it does:** The primary master orchestrator script. It contains a phased pipeline execution map (`Phase 1` through `Phase 8`) and uses `argparse` to let you execute the entire sequence sequentially or isolate a localized stage with interactive canvas rendering toggled on.



### Physics & Signal Processing Modules

* **`ris_waveform.py`**
* **What it does:** Simulates the time-domain multi-tone baseband signal steps driven by DDS phase-profile switching. It maps desired upper sidebands and mirror lower image sidebands, and evaluates frequency-dependent spatial beam squint.


* **`bistatic_radar.py`**
* **What it does:** Executes the complete 4-hop physical link budget matrix ($\text{Tx} \rightarrow \text{T-RIS} \rightarrow \text{Target} \rightarrow \text{R-RIS} \rightarrow \text{Rx}$) evaluating path loss and aperture captures. Synthesizes raw complex beat-signal data arrays injected with noise, processing them through IFFT and 2D Fourier shift maps to output 1D range peaks and 2D Range-Doppler spectrum records.


* **`estimation_crb.py`**
* **What it does:** Derives the analytical Cramér-Rao Bounds (CRB) for range, radial velocity, and 2D position by constructing the Fisher Information Matrix (FIM) from signal partial derivatives. Runs a statistical loop over an SNR sweep to validate the numerical root-mean-square error (RMSE) efficiency of our Fourier estimators.


* **`tracking.py`**
* **What it does:** Models a dynamic closed-loop beam tracking environment over 50 tracking frames. Propagates a constant-velocity target, calculates immediate mispointing power losses via a Dirichlet array-factor kernel, dynamically inflates noise variances, and implements an alpha-smoother loop to continuously steer the array.



### System Evaluation & Verification

* **`multi_target.py`**
* **What it does:** Populates a complex radar scene containing three distinct co-located targets. It visualizes how the M-FSK processing chain separates these profiles cleanly, contrasting it against a single-chirp FMCW radar framework that exhibits severe peak displacements.


* **`benchmarking.py`**
* **What it does:** Performs architectural system benchmarking across four distinct radar topologies. It evaluates component-level hardware power consumption draw equations and outputs multi-axis normalized performance scores.


* **`robustness.py`**
* **What it does:** Sweeps various physical hardware imperfections to establish a robust manufacturing error budget, modeling amplitude errors, phase tracking variances, air gap misalignments, and phase-quantization bounds.


* **`figures_tables.py`**
* **What it does:** Automated packaging block. Cross-checks that all expected figures and structural LaTeX tables are present, compiles them into structured LaTeX mapping arrays, merges data into a centralized database file, and auto-generates a summary markdown report.



---

## 3. Interactive Visualization Engine

* **`ris_mfsk_tracking_demo.py`**
* **What it does:** A standalone, real-time GUI simulator designed for rapid prototyping and intuition building. It runs an animated target tracking loop on a live canvas equipped with five synchronous diagnostic panels and user adjustment widgets.
* **How you can interact with it:** You can drag live sliders to manipulate target speed, change trajectory angles on the fly, adjust the alpha-smoother filter parameter, or scale linear power to inject heavier measurement noise. The animated beam cone actively re-steers and tracks the target across the grid in real time.



---

## 4. Setup and Quickstart Instructions

The entire simulator is engineered to run out of the box using core scientific Python libraries. No custom dependencies or external toolboxes are required.

### Prerequisites

Ensure you have Python 3.8+ installed along with the core numerical packages:

```bash
pip install numpy matplotlib scipy

```

### Running the System Simulation Pipeline

To verify the system end-to-end, run the master runner script:

```bash
python main.py

```

This single command will print out a structured system configuration overview, run all phases sequentially, perform the required Monte Carlo trials, and save publication-quality vector figures and standalone LaTeX tables directly into your local directory tree.

### Running Specific Phase Steps

If you are developing or editing a localized component (e.g., tweaking the tracking filter loops or reviewing link budget limits), you can run isolated phase commands:

```bash
# Execute only the link budget physics and estimation bounds
python main.py --phases 3 4

```

### Launching the Live Tracking App

To interactively explore the system tracking limits, open the desktop simulation dashboard:

```bash
python ris_mfsk_tracking_demo.py

```

---

## 5. Repository Directory Tree Layout

```text
.
├── main.py                     # Central master runner orchestrating all phases
├── config.py                   # Global physical, structural, and graphic constants
├── ris_waveform.py             # Phase 2: DDS tone stepping and spectral lines
├── bistatic_radar.py           # Phase 3: 4-hop physical link budget & Fourier parsing
├── estimation_crb.py           # Phase 4: Fisher Information Matrix & CRB loops
├── tracking.py                 # Phase 5: Dynamic constant-velocity alpha-tracking
├── multi_target.py             # Phase 6a: Multi-target scene & FMCW ghost comparison
├── benchmarking.py             # Phase 6b: Topology electricity draw & metrics grading
├── robustness.py               # Phase 7: Hardware tolerance & evanescent decay sweeps
├── figures_tables.py           # Phase 8a: Output validation and artifact collation
└── ris_mfsk_tracking_demo.py   # Live animated desktop beam-steering dashboard

```

All figures are automatically formatted using customized LaTeX typography.

If you want to tweak parameters (such as testing different element counts, shifting carrier frequencies, or setting heavier noise figures), simply adjust the values inside `config.py` and run `main.py` again. Welcome to the team! Feel free to raise any architectural questions as you begin exploring the code.

---

*Created in June 2026*

*@author: Pavan Mohan Neelamraju*

*Affiliation: Loughborough University*

**Email**: npavanmohan3@gmail.com

**Personal Website 🔴🔵**: [pavanmohan.netlify.app](https://pavanmohan.netlify.app/)
