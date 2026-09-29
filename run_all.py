#!/usr/bin/env python
"""
run_all.py -- reproduce every experiment and every figure of the paper.

    python run_all.py                  # full Monte Carlo (the paper's results)
    python run_all.py --quick          # reduced trials, for a fast smoke test
    python run_all.py --figures-only   # re-plot all figures from results/figdata
    python run_all.py --pdf            # also compile paper/ with pdflatex

Experiments write their data to results/ and the paper figures to
paper/figures/.  The run stops if any step's self-validation checks fail.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ['exp01_receiver_sideband', 'exp02_estimation_crb', 'exp03_beam_squint',
               'exp04_hardware_coupling', 'exp05_tracking', 'exp06_rx_aperture', 'exp07_benchmarks']
HAS_QUICK = {'exp02_estimation_crb', 'exp04_hardware_coupling', 'exp05_tracking', 'exp06_rx_aperture'}


def run(cmd, cwd=ROOT):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd)
    print(f'   -> exit {r.returncode}, {time.time() - t0:.0f} s', flush=True)
    if r.returncode != 0:
        sys.exit(f'failed: {" ".join(str(c) for c in cmd)}')


def figure_names():
    tex = ''.join((ROOT / 'paper' / f).read_text(encoding='utf-8')
                  for f in ('RIS_MFSK_TVT.tex', 'RIS_MFSK_TVT_supplementary.tex'))
    return sorted(set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='reduced Monte Carlo trials')
    ap.add_argument('--figures-only', action='store_true', help='re-plot the figures from stored data only')
    ap.add_argument('--pdf', action='store_true', help='compile the paper with pdflatex')
    args = ap.parse_args()

    if not args.figures_only:
        for e in EXPERIMENTS:
            print(f'== experiments/{e}.py', flush=True)
            extra = ['--quick'] if args.quick and e in HAS_QUICK else []
            run([sys.executable, str(ROOT / 'experiments' / f'{e}.py')] + extra)

    for f in sorted((ROOT / 'figures').glob('fig*.py')):
        print(f'== figures/{f.name}', flush=True)
        run([sys.executable, str(f)])

    names = figure_names()
    missing = [n for n in names if not (ROOT / 'paper' / 'figures' / f'{n}.pdf').exists()]
    print(f'== {len(names) - len(missing)} of {len(names)} manuscript figures present in paper/figures/')
    if missing:
        sys.exit(f'missing figures: {missing}')

    if args.pdf:
        for doc, passes in (('RIS_MFSK_TVT', 3), ('RIS_MFSK_TVT_supplementary', 2)):
            for _ in range(passes):
                subprocess.run(['pdflatex', '-interaction=nonstopmode', f'{doc}.tex'],
                               cwd=ROOT / 'paper', stdout=subprocess.DEVNULL)
            print(f'== paper/{doc}.pdf')


if __name__ == '__main__':
    main()
