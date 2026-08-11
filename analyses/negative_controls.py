"""Negative controls for the decoder.

A decoder trained on a task-demand label may be learning task phase, motor or
ocular effort, or any other difficulty-correlated feature rather than a
generalisable arousal state. Four conditions are run with the same architecture
and training recipe throughout:

  physio    the peripheral decoder, as a reference
  motor     joystick and head-motion channels only
  shuffled  peripheral channels, labels permuted within subject
  shifted   peripheral channels, each ring given a later ring's label

The shifted condition is retained for completeness only. ``label_controls.py``
shows that it changes too few labels to be informative, and why no label shift
can do better in this design.

Integrated Gradients is skipped here. Only AUC matters.

Outputs: results/negative_controls.csv
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import D, OUT, cfg
from arousal import training as T
from arousal.config import CHANNEL_GROUPS, CH
from arousal.signals import normalize_train_val

MOTOR_CHANNELS = [CH[c] for c in (CHANNEL_GROUPS["Joystick Controls"]
                                  + CHANNEL_GROUPS["Head Motion"])]
MODALITIES_MOTOR = {"joystick": list(range(12)), "head": list(range(12, 18))}
N_REPS = 2
SHIFT = 2          # rings
RNG = np.random.default_rng(11)


def cv_auc(X, y, modalities, device, n_reps=N_REPS, seed_base=0):
    """Repeated 5-fold CV, AUC on the full epoch only. No attribution."""
    kf = StratifiedKFold(5, shuffle=True, random_state=13)
    splits = list(kf.split(X, y))
    aucs = []
    for rep in range(n_reps):
        proba = np.zeros(len(y))
        for fold, (tr, va) in enumerate(splits):
            Xtr, Xva = normalize_train_val(X[tr], X[va])
            Xtr = torch.tensor(Xtr, dtype=torch.float32).to(device)
            ytr = torch.tensor(y[tr], dtype=torch.long).to(device)
            Xva = torch.tensor(Xva, dtype=torch.float32).to(device)
            model = T.fit_decoder(Xtr, ytr, modalities, device,
                                  seed=seed_base + 10 * rep + fold)
            proba[va] = T.predict_probs(model, Xva, device)[0][:, 1]
        aucs.append(roc_auc_score(y, proba))
    return np.array(aucs)


def run():
    device = T.get_device()
    ring = D.load_ring_epochs()
    ring = ring[(ring.condition == cfg.CALIBRATION_CONDITION) & (ring.label >= 0)]
    print(f"device: {device} | {len(ring)} calibration epochs | {N_REPS} reps\n")

    rows = []
    for sidx, s in enumerate(cfg.SUBJECTS):
        subj = ring[ring.subj_idx == s].sort_values(["trial_idx", "ring_idx"])
        data = np.stack(subj["data"].to_numpy())
        y = subj["label"].to_numpy()

        X_phys = data[:, cfg.PERIPHERAL_CHANNELS, :]
        X_motor = data[:, MOTOR_CHANNELS, :]

        # Labels permuted within subject: destroys any real relationship while
        # preserving class balance.
        y_shuf = RNG.permutation(y)

        # Each ring inherits the label of the ring SHIFT positions later in the
        # same trial. Tests whether slow drift alone carries the signal.
        trials = subj["trial_idx"].to_numpy()
        y_shift = y.copy().astype(float)
        for t in np.unique(trials):
            m = np.where(trials == t)[0]
            if len(m) > SHIFT:
                y_shift[m[:-SHIFT]] = y[m[SHIFT:]]
                y_shift[m[-SHIFT:]] = np.nan
            else:
                y_shift[m] = np.nan
        keep = ~np.isnan(y_shift)

        t0 = time.time()
        res = {
            "physio": cv_auc(X_phys, y, cfg.MODALITIES_PERIPHERAL, device,
                             seed_base=1000 * sidx),
            "motor": cv_auc(X_motor, y, MODALITIES_MOTOR, device,
                            seed_base=2000 + 1000 * sidx),
            "shuffled": cv_auc(X_phys, y_shuf, cfg.MODALITIES_PERIPHERAL, device,
                               seed_base=3000 + 1000 * sidx),
        }
        if keep.sum() > 40 and len(np.unique(y_shift[keep])) == 2:
            res["shifted"] = cv_auc(X_phys[keep], y_shift[keep].astype(int),
                                    cfg.MODALITIES_PERIPHERAL, device,
                                    seed_base=4000 + 1000 * sidx)
        else:
            res["shifted"] = np.array([np.nan])

        row = {"subject": s, **{k: v.mean() * 100 for k, v in res.items()}}
        rows.append(row)
        print(f"  S{s:02d}  physio {row['physio']:5.1f}  motor {row['motor']:5.1f}  "
              f"shuffled {row['shuffled']:5.1f}  shifted {row['shifted']:5.1f}   "
              f"[{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "negative_controls.csv", index=False)

    print("\n" + "=" * 74)
    print("Negative controls (within-subject AUC, %)")
    print("=" * 74)
    print(f"{'condition':<12}{'mean':>8}{'SEM':>7}{'min':>7}{'max':>7}"
          f"{'n>60%':>7}")
    for c in ("physio", "motor", "shuffled", "shifted"):
        v = df[c].dropna().to_numpy()
        print(f"{c:<12}{v.mean():>8.1f}{v.std(ddof=1)/np.sqrt(len(v)):>7.2f}"
              f"{v.min():>7.1f}{v.max():>7.1f}{int((v>60).sum()):>7d}")

    from scipy.stats import ttest_rel, ttest_1samp
    print()
    for c in ("motor", "shuffled", "shifted"):
        both = df[["physio", c]].dropna()
        t, p = ttest_rel(both["physio"], both[c])
        t1, p1 = ttest_1samp(both[c], 50)
        print(f"  physio vs {c:<9} t={t:5.2f} p={p:.2e}   |   "
              f"{c} vs chance: t={t1:5.2f} p={p1:.3g}")
    return df


if __name__ == "__main__":
    run()
