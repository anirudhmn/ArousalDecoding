"""Leave-one-subject-out decoder validation.

Sixteen folds. Each fold trains on the calibration epochs of fifteen subjects
and tests on the sixteenth, so no subject contributes to both sides. Channel
normalisation is fitted on the training subjects only.

Both feature sets are run, because the headline result is a comparison: if
peripheral signals beat EEG within subjects but not across them, that changes
what can be claimed about modality.

Results are written after every fold, so a partial run is still usable and an
interrupted run resumes from the saved CSV.

Outputs: results/loso_decoder.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from common import D, OUT, cfg
from arousal import training as T
from arousal.signals import normalize_train_val

N_REPS_DEFAULT = 3
CSV = OUT / "loso_decoder.csv"


def loso_fold(X_tr, y_tr, X_te, y_te, modalities, device, n_reps, seed_base):
    """Train on the held-in subjects, score the held-out one.

    Probabilities are averaged across repetitions before scoring, matching how
    the within-subject pipeline averages its repeated decoders.
    """
    Xtr_n, Xte_n = normalize_train_val(X_tr, X_te)
    Xtr = torch.tensor(Xtr_n, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_tr, dtype=torch.long).to(device)
    Xte = torch.tensor(Xte_n, dtype=torch.float32).to(device)

    probs = np.zeros((n_reps, len(y_te)))
    for rep in range(n_reps):
        model = T.fit_decoder(Xtr, ytr, modalities, device, seed=seed_base + rep)
        probs[rep] = T.predict_probs(model, Xte, device)[0][:, 1]
        del model

    mean_p = probs.mean(axis=0)
    return {
        "auc": roc_auc_score(y_te, mean_p) * 100,
        "acc": accuracy_score(y_te, (mean_p > 0.5).astype(int)) * 100,
        "auc_per_rep": [roc_auc_score(y_te, p) * 100 for p in probs],
    }


def run(feature_sets=("physio",), n_reps=N_REPS_DEFAULT):
    device = T.get_device()
    print(f"device: {device} | {n_reps} repetitions per fold\n", flush=True)

    rows = []
    if CSV.exists():                      # resume a partial run
        rows = pd.read_csv(CSV).to_dict("records")
        done = {(r["feature_set"], r["subject"]) for r in rows}
        print(f"resuming: {len(done)} folds already on disk\n", flush=True)
    else:
        done = set()

    ring_raw = D.load_ring_epochs()
    ring_raw = ring_raw[(ring_raw.condition == cfg.CALIBRATION_CONDITION)
                        & (ring_raw.label >= 0)]

    for fs in feature_sets:
        print("=" * 78)
        print(f"feature set: {fs}")
        print("=" * 78, flush=True)

        prep, modalities = D.prepare_features(ring_raw, fs)
        X_all = np.stack(prep["data"].to_numpy())
        y_all = prep["label"].to_numpy()
        s_all = prep["subj_idx"].to_numpy()
        print(f"  {X_all.shape[0]} epochs, {X_all.shape[1]} channels, "
              f"{X_all.shape[2]} samples", flush=True)

        for sidx, s in enumerate(cfg.SUBJECTS):
            if (fs, s) in done:
                continue
            te = s_all == s
            tr = ~te
            t0 = time.time()
            r = loso_fold(X_all[tr], y_all[tr], X_all[te], y_all[te],
                          modalities, device, n_reps,
                          seed_base=7000 + 100 * sidx)
            rows.append({"feature_set": fs, "subject": int(s),
                         "n_train": int(tr.sum()), "n_test": int(te.sum()),
                         "auc": r["auc"], "acc": r["acc"],
                         "auc_sd_reps": float(np.std(r["auc_per_rep"], ddof=1))})
            pd.DataFrame(rows).to_csv(CSV, index=False)
            print(f"  S{s:02d}  AUC {r['auc']:5.1f}  acc {r['acc']:5.1f}   "
                  f"[{time.time()-t0:.0f}s]", flush=True)

        df = pd.DataFrame(rows)
        v = df[df.feature_set == fs]["auc"].to_numpy()
        print(f"\n  {fs}: LOSO AUC {v.mean():.1f} +/- "
              f"{v.std(ddof=1)/np.sqrt(len(v)):.2f} (SEM), "
              f"range {v.min():.1f}-{v.max():.1f}\n", flush=True)

    summarise(pd.DataFrame(rows))
    return pd.DataFrame(rows)


def summarise(df):
    from scipy.stats import ttest_1samp, ttest_rel

    print("=" * 78)
    print("Leave-one-subject-out decoder AUC (%)")
    print("=" * 78)
    print(f"\n{'feature set':<14}{'mean':>8}{'SEM':>7}{'min':>7}{'max':>7}"
          f"{'n>60%':>7}{'vs chance':>14}")
    for fs, g in df.groupby("feature_set"):
        v = g["auc"].to_numpy()
        t, p = ttest_1samp(v, 50)
        print(f"{fs:<14}{v.mean():>8.1f}{v.std(ddof=1)/np.sqrt(len(v)):>7.2f}"
              f"{v.min():>7.1f}{v.max():>7.1f}{int((v > 60).sum()):>7d}"
              f"   t={t:5.2f} p={p:.1e}")

    sets = sorted(df.feature_set.unique())
    if len(sets) > 1:
        print()
        wide = df.pivot(index="subject", columns="feature_set", values="auc")
        for i, a in enumerate(sets):
            for b in sets[i + 1:]:
                both = wide[[a, b]].dropna()
                t, p = ttest_rel(both[a], both[b])
                print(f"  {a} vs {b}: diff {both[a].mean()-both[b].mean():+.1f} "
                      f"points, t={t:.2f}, p={p:.4g}")

    # The comparison that matters: does the within-subject advantage
    # survive when no data from the test subject is available?
    try:
        ws = pd.read_csv(OUT / "negative_controls.csv")[["subject", "physio"]]
        m = df[df.feature_set == "physio"].merge(ws, on="subject")
        if len(m):
            t, p = ttest_rel(m["physio"], m["auc"])
            print(f"\n  within-subject {m['physio'].mean():.1f} vs "
                  f"LOSO {m['auc'].mean():.1f} "
                  f"(drop {m['physio'].mean()-m['auc'].mean():.1f} points, "
                  f"t={t:.2f}, p={p:.2e})")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-sets", nargs="+", default=["physio"])
    ap.add_argument("--repetitions", type=int, default=N_REPS_DEFAULT)
    a = ap.parse_args()
    run(tuple(a.feature_sets), a.repetitions)
