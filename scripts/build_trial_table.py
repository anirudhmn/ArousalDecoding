"""Build data/trial_table.pkl, the table every analysis script reads.

Identical to the build cell of notebooks/03_yerkes_dodson.ipynb, extracted into
a script so that a plain checkout can regenerate the table without opening a
notebook. `analyses/run_fast.sh` is unusable until this file exists.

The decoder output read here is `results_online_simple_physio.pkl`. That choice
is not arbitrary: it is the file whose per-trial lengths match the
`ring_events_online.pkl` on disk, and the one that reproduces every published
count and coefficient.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arousal import config as cfg
from arousal import data as D

CACHE = cfg.DATA / "trial_table.pkl"
DECODER = cfg.RESULTS / "results_online_simple_physio.pkl"

# The decoded index is expected on a 0-100 scale: arousal.control hardcodes
# AROUSAL_BIN = 5 and a band half-width floor MIN_STD = 5.0, and every deviation
# metric is a percentage of that range. A decoder output on a 0-1 scale does not
# raise anywhere - it silently produces a band wider than the whole signal, so
# nothing ever exceeds it and every association comes back null. Hence the check.
EXPECTED_MAX = 100.0


def check_scale(arousal_results, rescale):
    """Verify the decoder output is on the 0-100 scale the analyses assume."""
    vals = np.concatenate([np.asarray(t).ravel()[::997]
                           for s in arousal_results for t in s])
    n_nan = int(np.isnan(vals).sum())
    hi = float(np.nanmax(vals))
    print(f"  decoder output range: max {hi:.4f}, "
          f"{n_nan} NaN in the sampled values")
    if rescale is None:
        if hi <= 1.5:
            raise SystemExit(
                f"\nRefusing to build: the decoder output tops out at {hi:.4f}, "
                f"so it is on a 0-1 scale.\nEvery downstream constant assumes "
                f"0-100 (MIN_STD = 5.0, AROUSAL_BIN = 5). Building from this "
                f"would\nnot error - it would silently return null findings "
                f"everywhere.\nRe-run with --rescale 100 if that is what you "
                f"intend.")
        return 1.0
    print(f"  rescaling by x{rescale:g}")
    return rescale


def drop_bad_trials(arousal_results):
    """Remove traces containing NaN, reporting exactly what was dropped."""
    out, dropped = [], []
    for s, subj in enumerate(arousal_results):
        keep = []
        for i, t in enumerate(subj):
            if np.isnan(np.asarray(t)).any():
                dropped.append((s, i, np.asarray(t).shape[1]))
            else:
                keep.append(t)
        out.append(keep)
    if dropped:
        print(f"  dropped {len(dropped)} trace(s) containing NaN:")
        for s, i, n in dropped:
            print(f"    subject index {s}, trial {i}, {n} samples")
    return out


def main(overwrite=False, decoder=None, rescale=None, out=None):
    global DECODER, CACHE
    if decoder:
        DECODER = cfg.RESULTS / decoder
    if out:
        CACHE = cfg.DATA / out
    if CACHE.exists() and not overwrite:
        df = pd.read_pickle(CACHE)
        print(f"{CACHE.name} exists ({len(df)} trials); use --overwrite to rebuild")
        return

    t0 = time.time()
    with open(DECODER, "rb") as f:
        arousal_results = pickle.load(f)["final_results"]
    print(f"decoder output: {DECODER.name}, "
          f"{sum(len(x) for x in arousal_results)} trial traces")
    scale = check_scale(arousal_results, rescale)
    arousal_results = drop_bad_trials(arousal_results)
    if scale != 1.0:
        arousal_results = [[np.asarray(t) * scale for t in s]
                           for s in arousal_results]
    print(f"  {sum(len(x) for x in arousal_results)} traces retained")

    ring_online = D.load_ring_epochs_online()
    print(f"  loaded {len(ring_online)} ring epochs [{time.time()-t0:.0f}s]")

    trials = D.build_trial_table(arousal_results, ring_online=ring_online)
    print(f"  trial table: {len(trials)} trials [{time.time()-t0:.0f}s]")

    baselines = D.baseline_features(ring_online)
    df = D.attach_baselines(trials, baselines)
    df["arousal_old_mean"] = df["old_arousal"].apply(np.mean)
    df.to_pickle(CACHE)

    print(f"built trial table: {len(df)} trials [{time.time()-t0:.0f}s]")
    print(f"  subjects   : {df['subject'].nunique()}")
    print(f"  conditions : {df['condition'].value_counts().sort_index().to_dict()}")
    print(f"  difficulty : {df['difficulty'].value_counts().to_dict()}")
    print(f"wrote {CACHE}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--decoder", default=None,
                    help="decoder output filename inside data/results/")
    ap.add_argument("--rescale", type=float, default=None,
                    help="multiply the decoded index by this factor")
    ap.add_argument("--out", default=None, help="output filename inside data/")
    a = ap.parse_args()
    main(overwrite=a.overwrite, decoder=a.decoder, rescale=a.rescale, out=a.out)
