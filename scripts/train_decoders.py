#!/usr/bin/env python3
"""Train the decoders headlessly. Equivalent to notebook 01.

    python scripts/train_decoders.py --stage cv --feature-set physio
    python scripts/train_decoders.py --stage online --subjects 1 2
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arousal import config as cfg
from arousal import data as D
from arousal import signals as S
from arousal import training as T


def run_cv(args, device):
    ring = D.load_ring_epochs()
    ring = ring[ring["condition"] == cfg.CALIBRATION_CONDITION]
    if args.subjects:
        ring = ring[ring["subj_idx"].isin(args.subjects)]

    for name in args.feature_set:
        out = cfg.RESULTS / f"results_offline_simple_{name}{args.suffix}.pkl"
        if out.exists() and not args.overwrite:
            print(f"{out.name} exists; skipping (use --overwrite)")
            continue

        print(f"\n=== {name}: {len(ring)} epochs, "
              f"{ring['subj_idx'].nunique()} subjects")
        df, modalities = D.prepare_features(ring, name)
        t0 = time.time()
        final_results, imps = T.within_subject_cv(
            df, modalities, device, n_repetitions=args.repetitions)
        elapsed = time.time() - t0

        per_subject = final_results[:, :, -1, 1].mean(axis=1)
        print(f"{name}: AUC {per_subject.mean()*100:.2f} +/- "
              f"{per_subject.std()/np.sqrt(len(per_subject))*100:.2f} SEM  "
              f"({elapsed/60:.1f} min)")
        with open(out, "wb") as f:
            pickle.dump({"final_results": final_results, "imps": imps,
                         "time": elapsed}, f)
        print(f"wrote {out}")


def run_online(args, device):
    out = cfg.RESULTS / f"results_online_simple_physio{args.suffix}.pkl"
    if out.exists() and not args.overwrite:
        print(f"{out.name} exists; skipping (use --overwrite)")
        return

    online = D.load_ring_epochs_online()
    fixed = D.load_ring_epochs()
    subjects = args.subjects or cfg.SUBJECTS

    final_results, t0 = [], time.time()
    for s in subjects:
        calib = fixed[(fixed["subj_idx"] == s)
                      & (fixed["condition"] == cfg.CALIBRATION_CONDITION)
                      & (fixed["label"] >= 0)]
        X_train = np.stack(calib["data"].to_numpy())[:, cfg.PERIPHERAL_CHANNELS, :]
        y_train = calib["label"].to_numpy()

        subj = online[(online["subj_idx"] == s) & (online["label"] >= 0)]
        trials, _, _ = D.concatenate_trials(
            subj[subj["condition"] != cfg.CALIBRATION_CONDITION],
            channels=cfg.PERIPHERAL_CHANNELS)

        X_train_n, trials_n = S.normalize_train_val(X_train, trials)
        traces = T.decode_subject_trials(
            X_train_n, y_train, trials_n, cfg.MODALITIES_PERIPHERAL, device,
            n_repetitions=args.repetitions, seed_base=100 * s)
        final_results.append(traces)
        print(f"  S{s:02d}: {len(traces)} trials  "
              f"({(time.time()-t0)/60:.1f} min elapsed)")

    with open(out, "wb") as f:
        pickle.dump({"final_results": final_results, "time": time.time() - t0}, f)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["cv", "online", "both"], default="both")
    p.add_argument("--feature-set", nargs="+", default=list(D.FEATURE_SETS),
                   choices=list(D.FEATURE_SETS))
    p.add_argument("--subjects", nargs="+", type=int, default=None)
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--suffix", default="", help="appended to output filenames")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = T.get_device(prefer_gpu=not args.cpu)
    print(f"device: {device}")

    if args.stage in ("cv", "both"):
        run_cv(args, device)
    if args.stage in ("online", "both"):
        run_online(args, device)


if __name__ == "__main__":
    main()
