"""
main.py — RIS M-FSK Sensing Simulation Suite
=============================================
Master runner for all simulation phases.

Usage
-----
    # Run all phases sequentially:
    python main.py

    # Run specific phases only:
    python main.py --phases 1 2

    # Run a single phase and show figures interactively:
    python main.py --phases 1 --show

Dependencies
------------
    pip install numpy matplotlib scipy

Phase map
---------
    Phase 1 — Device physics   → F4, F5, F6   (pin_diode_model.py)
    Phase 2 — RIS waveform     → F7, F8        (ris_waveform.py)
    Phase 3 — Bistatic radar   → F10, F11, T1, T2  (bistatic_radar.py)
    Phase 4 — CRB / MC RMSE   → F12, F13, T7      (estimation_crb.py)
    Phase 5 — Beam tracking    → F14, F15, T3      (tracking.py)
    Phase 6 — Benchmarking     → F16–F18, T4–T6,T8 (multi_target.py + benchmarking.py)
    Phase 7 — Robustness       → F-S1, F-S2, T9, T10  (robustness.py)
    Phase 8 — Publication pack → LaTeX + report + package (figures_tables.py)
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
        description='RIS M-FSK Sensing Simulation Suite — IEEE TVT target')
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
