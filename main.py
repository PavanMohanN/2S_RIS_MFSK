"""
main.py — RIS M-FSK Sensing Simulation Suite
=============================================
Acts as the central master orchestrator and runner for the multi-disciplinary, 
stage-wise radar simulation pipeline.

Pipeline Flow and Architecture Mapping
--------------------------------------
  * Phase 1: Device Physics           — Equivalent circuit modeling of PIN-diode variations
  * Phase 2: RIS Waveform Layer       — Multi-tone frequency generation and spectral modeling
  * Phase 3: Bistatic Radar Chains    — Wave path link budget calculation and range-Doppler map generation
  * Phase 4: Statistical Estimation  — Fisher information evaluation and Cramér-Rao bound loops
  * Phase 5: Closed-Loop Tracking     — Target kinematics and feedback-driven array redirection
  * Phase 6: System Benchmarking      — Coherent profile multi-target extraction and comparative analysis
  * Phase 7: Hardware Robustness      — Multi-parameter tolerance degradation sweeps
  * Phase 8: Deployment Assembly      — Central database package construction and artifact collation

Execution Parameters
--------------------
    # Run the complete sequence:
    python main.py

    # Execute specific sequential stages:
    python main.py --phases 3 4 5

    # Run a localized stage with interactive canvas display active:
    python main.py --phases 5 --show
"""

import argparse
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner(phase_num, title):
    width = 62
    print("\n" + "█" * width)
    print(f"  PHASE {phase_num} — {title}")
    print("█" * width)


def run_phase_1(show=False):
    banner(1, "Device Physics — PIN Diode Model")
    from pin_diode_model import run as p1_run
    return p1_run(show=show)


def run_phase_2(show=False):
    banner(2, "RIS Waveform Layer — M-FSK Spectrum & Spectrogram")
    from ris_waveform import run as p2_run
    return p2_run(show=show)


# ____________ Placeholder stubs for phases 3–8 ____________________________ Need to correct from here
def run_phase_3(show=False):
    banner(3, "Bistatic Radar Signal Processing — F10, F11, T1, T2")
    from bistatic_radar import run as p3_run
    return p3_run(show=show)


def run_phase_4(show=False):
    banner(4, "CRB + Monte Carlo RMSE — F12, F13, T7")
    from estimation_crb import run as p4_run
    return p4_run(show=show)

def run_phase_5(show=False):
    banner(5, "Closed-Loop Beam Tracking — F14, F15, T3")
    from tracking import run as p5_run
    return p5_run(show=show)


def run_phase_6(show=False):
    banner(6, "Multi-Target + Benchmarking — F16, F17, F18, T4–T6, T8")
    from multi_target  import run as mt_run
    from benchmarking  import run as bm_run
    mt_results = mt_run(show=show)
    bm_results = bm_run(show=show)
    return mt_results, bm_results

def run_phase_7(show=False):
    banner(7, "Robustness and Sensitivity Analysis — F-S1, F-S2, T9, T10")
    from robustness import run as p7_run
    return p7_run(show=show)


def run_phase_8(show=False):
    banner(8, "Publication Assembly — LaTeX snippets + report + package")
    from figures_tables import run as p8_run
    return p8_run(show=show)


PHASE_MAP = {
    1: run_phase_1,
    2: run_phase_2,
    3: run_phase_3,
    4: run_phase_4,
    5: run_phase_5,
    6: run_phase_6,
    7: run_phase_7,
    8: run_phase_8,
}


def main():
    parser = argparse.ArgumentParser(
        description='RIS M-FSK Sensing Simulation Suite ')
    parser.add_argument(
        '--phases', nargs='+', type=int,
        default=list(PHASE_MAP.keys()),
        help='Which phases to run (default: all). Example: --phases 1 2')
    parser.add_argument(
        '--show', action='store_true',
        help='Display figures interactively (requires a display)')
    args = parser.parse_args()

    # Print config summary
    import config as cfg
    cfg.print_config_summary()

    t_total = time.time()
    results = {}

    for phase_num in sorted(args.phases):
        if phase_num not in PHASE_MAP:
            print(f"\n  [WARNING] Phase {phase_num} not recognised — skipping.")
            continue
        t0 = time.time()
        results[phase_num] = PHASE_MAP[phase_num](show=args.show)
        elapsed = time.time() - t0
        print(f"\n  ↳ Phase {phase_num} finished in {elapsed:.1f} s")

    print("\n" + "═" * 62)
    print(f"  All requested phases complete.")
    print(f"  Total wall time : {time.time() - t_total:.1f} s")
    print(f"  Figures saved to: {os.path.abspath(cfg.OUTPUT_DIR)}/")
    print(f"  Data saved to   : {os.path.abspath(cfg.DATA_DIR)}/")
    print("═" * 62 + "\n")

    return results


if __name__ == '__main__':
    main()
