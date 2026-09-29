"""
mfsk.figio -- figure input/output shared by the experiments and the figure scripts.

Every figure of the paper is produced by a function decorated with
``@paper_figure('<name>')`` in ``figures/``.  When an experiment calls it, the
decorator

  1. stores the function's arguments in ``results/figdata/<name>.json`` (with the
     numpy arrays in ``<name>.npz``), and
  2. draws the figure into ``paper/figures/<name>.pdf`` (and ``.png``).

Running a figure script on its own calls :func:`replot`, which reloads the stored
arguments and draws the figure again -- so figures can be edited without
re-running the simulations.  ``<name>`` is the file name used by
``\\includegraphics`` in the manuscript.
"""
from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / 'paper' / 'figures'
DATADIR = ROOT / 'results' / 'figdata'


def _enc(x, arrays):
    if isinstance(x, np.ndarray):
        key = f'a{len(arrays)}'
        arrays[key] = x
        return {'__nd__': key}
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (complex, np.complexfloating)):
        return {'__complex__': [float(np.real(x)), float(np.imag(x))]}
    if isinstance(x, tuple):
        return {'__tuple__': [_enc(v, arrays) for v in x]}
    if isinstance(x, list):
        return [_enc(v, arrays) for v in x]
    if isinstance(x, dict):
        if all(isinstance(k, str) for k in x):
            return {k: _enc(v, arrays) for k, v in x.items()}
        return {'__items__': [[_enc(k, arrays), _enc(v, arrays)] for k, v in x.items()]}
    if isinstance(x, Path):
        return str(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return {'__unserializable__': type(x).__name__}


def _dec(x, arrays):
    if isinstance(x, list):
        return [_dec(v, arrays) for v in x]
    if isinstance(x, dict):
        if '__nd__' in x:
            return arrays[x['__nd__']]
        if '__tuple__' in x:
            return tuple(_dec(v, arrays) for v in x['__tuple__'])
        if '__items__' in x:
            return {_dec(k, arrays): _dec(v, arrays) for k, v in x['__items__']}
        if '__complex__' in x:
            return complex(*x['__complex__'])
        if '__unserializable__' in x:
            return None
        return {k: _dec(v, arrays) for k, v in x.items()}
    return x


def save(name, data):
    DATADIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    enc = _enc(data, arrays)
    (DATADIR / f'{name}.json').write_text(json.dumps(enc, indent=1), encoding='utf-8')
    np.savez_compressed(DATADIR / f'{name}.npz', **arrays)


def load(name):
    js, nz = DATADIR / f'{name}.json', DATADIR / f'{name}.npz'
    if not js.exists():
        raise SystemExit(f'no stored data for {name}: run the experiment that produces it first '
                         f'(see README, "Figures")')
    with np.load(nz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return _dec(json.loads(js.read_text(encoding='utf-8')), arrays)


def paper_figure(name):
    """Decorator: store the arguments, then draw into paper/figures/<name>.pdf."""
    def deco(fn):
        sig = inspect.signature(fn)
        first = next(iter(sig.parameters))

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            data = {k: v for k, v in bound.arguments.items() if k != first}
            save(name, data)
            return fn(FIGDIR, **data)
        wrapper.plot = fn
        wrapper.figname = name
        return wrapper
    return deco


def replot(fig_fn):
    """Re-draw a figure from its stored arguments."""
    from mfsk import common as mc
    mc.set_ieee_style()
    return fig_fn.plot(FIGDIR, **load(fig_fn.figname))
