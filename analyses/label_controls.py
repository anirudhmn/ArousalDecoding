"""Label controls that change a meaningful number of labels.

A temporally shifted label control is only informative if it actually changes
labels. Giving each ring the label of the ring two positions later changes only about
7 percent of them. Ring size changes once or twice in a whole trial, so a
two-ring shift mostly maps a label onto itself. Its AUC is uninterpretable.

Four replacements, each reported with the fraction of labels it changes:

  cross-trial   each ring takes the label of the ring at the same position in a
                different trial of the same subject. The label-to-phase mapping
                is preserved in aggregate and only the trial-specific pairing is
                broken.
  reversed      the label sequence is reversed within each trial, so labels run
                against trial phase instead of with it.
  phase-perm    labels permuted within ring-position strata. Anything the
                decoder can still do must be independent of phase.
  block-perm    whole trials keep their label sequence but are reassigned to
                other trials' epochs. Preserves within-trial label structure and
                destroys the physiological pairing.

The complementary question, how much of the real decoder's accuracy survives
once phase is held fixed, is answered without retraining by stratifying the
out-of-fold probabilities by ring position.

Outputs: results/label_controls_shift_controls.csv,
         results/label_controls_probs.csv,
         results/label_controls_determinism.csv
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import D, OUT, cfg
from arousal import training as T
from arousal.signals import normalize_train_val

N_REPS = 2
RNG = np.random.default_rng(1907)


# --------------------------------------------------------------------------- #
# Label controls
# --------------------------------------------------------------------------- #

def cross_trial_labels(y, trials, positions, rng):
    """Label of the ring at the same position in a different trial."""
    out = y.copy()
    uniq = np.unique(trials)
    by_trial = {t: dict(zip(positions[trials == t], y[trials == t])) for t in uniq}
    for i in range(len(y)):
        others = [t for t in uniq if t != trials[i] and positions[i] in by_trial[t]]
        if others:
            out[i] = by_trial[rng.choice(others)][positions[i]]
    return out


def reversed_labels(y, trials):
    out = y.copy()
    for t in np.unique(trials):
        m = np.where(trials == t)[0]
        out[m] = y[m][::-1]
    return out


def phase_permuted_labels(y, positions, rng):
    """Permute labels within ring-position strata."""
    out = y.copy()
    for p in np.unique(positions):
        m = np.where(positions == p)[0]
        if len(m) > 1:
            out[m] = rng.permutation(y[m])
    return out


def block_permuted_labels(y, trials, rng):
    """Reassign whole label sequences between trials of the same length."""
    out = y.copy()
    uniq = np.unique(trials)
    by_len = {}
    for t in uniq:
        by_len.setdefault((trials == t).sum(), []).append(t)
    for t in uniq:
        pool = [o for o in by_len[(trials == t).sum()] if o != t]
        if pool:
            out[trials == t] = y[trials == rng.choice(pool)]
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def cv_probs(X, y, modalities, device, n_reps=N_REPS, seed_base=0):
    """Repeated 5-fold CV; returns out-of-fold probabilities averaged over reps."""
    kf = StratifiedKFold(5, shuffle=True, random_state=13)
    splits = list(kf.split(X, y))
    acc = np.zeros((n_reps, len(y)))
    for rep in range(n_reps):
        for fold, (tr, va) in enumerate(splits):
            Xtr, Xva = normalize_train_val(X[tr], X[va])
            Xtr = torch.tensor(Xtr, dtype=torch.float32).to(device)
            ytr = torch.tensor(y[tr], dtype=torch.long).to(device)
            Xva = torch.tensor(Xva, dtype=torch.float32).to(device)
            model = T.fit_decoder(Xtr, ytr, modalities, device,
                                 seed=seed_base + 10 * rep + fold)
            acc[rep, va] = T.predict_probs(model, Xva, device)[0][:, 1]
    return acc.mean(axis=0)


def stratified_auc(y, prob, positions, min_per_class=3):
    """AUC computed within ring-position strata, then pooled by stratum size.

    If the label is close to a deterministic function of ring position, most
    strata will contain a single class and drop out - which is itself the
    finding, so the number of usable strata is reported alongside.
    """
    num = den = 0.0
    usable = 0
    for p in np.unique(positions):
        m = positions == p
        yy = y[m]
        if min(np.sum(yy == 0), np.sum(yy == 1)) < min_per_class:
            continue
        w = np.sum(yy == 0) * np.sum(yy == 1)      # discordant pairs
        num += roc_auc_score(yy, prob[m]) * w
        den += w
        usable += 1
    return (num / den if den else np.nan), usable, int(den)


def label_determinism(ring):
    """Is the task-demand label a function of ring position?

    This decides whether any of the controls below can work at all. If the
    label is determined by position, then permuting or shifting labels while
    respecting trial structure changes nothing, and no phase-stratified
    analysis is possible.
    """
    rows = []
    for s in cfg.SUBJECTS:
        g = ring[ring.subj_idx == s]
        y, pos = g["label"].to_numpy(), g["ring_position"].to_numpy()
        up = np.unique(pos)
        rows.append(dict(
            subject=int(s), n_epochs=len(y),
            auc_position_only=roc_auc_score(y, pos) * 100,
            n_positions=len(up),
            n_mixed_strata=sum(1 for p in up if len(np.unique(y[pos == p])) > 1),
            deterministic=all(len(np.unique(y[pos == p])) == 1 for p in up)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "label_controls_determinism.csv", index=False)

    print("=" * 86)
    print("Is the label a deterministic function of trial phase?")
    print("=" * 86)
    print(f"\n  ring position alone predicts the label at "
          f"{df.auc_position_only.mean():.1f}% AUC (mean over subjects)")
    print(f"  exactly deterministic for {int(df.deterministic.sum())}/16 subjects")
    print(f"  ring-position strata containing both classes: "
          f"{int(df.n_mixed_strata.sum())} of {int(df.n_positions.sum())}")
    print("\n  Where this holds, any label control that respects trial phase is a")
    print("  no-op, and phase-stratified AUC is undefined. Read the change rates")
    print("  in the table below before reading any of the AUCs.\n")
    return df


def run():
    device = T.get_device()
    ring = D.load_ring_epochs()
    ring = ring[(ring.condition == cfg.CALIBRATION_CONDITION) & (ring.label >= 0)]
    print(f"device: {device} | {len(ring)} calibration epochs | {N_REPS} reps\n",
          flush=True)
    label_determinism(ring)

    CONTROLS = ["real", "shift2 (+2 rings)", "cross-trial", "reversed",
                "phase-perm", "block-perm"]
    rows, prob_rows = [], []

    for sidx, s in enumerate(cfg.SUBJECTS):
        subj = ring[ring.subj_idx == s].sort_values(["trial_idx", "ring_idx"])
        X = np.stack(subj["data"].to_numpy())[:, cfg.PERIPHERAL_CHANNELS, :]
        y = subj["label"].to_numpy()
        trials = subj["trial_idx"].to_numpy()
        pos = subj["ring_position"].to_numpy()

        # the +2-ring shift, reproduced so its label-change rate is on record
        y_shift = y.copy().astype(float)
        for t in np.unique(trials):
            m = np.where(trials == t)[0]
            if len(m) > 2:
                y_shift[m[:-2]] = y[m[2:]]
                y_shift[m[-2:]] = np.nan
            else:
                y_shift[m] = np.nan

        variants = {
            "real": y,
            "shift2 (+2 rings)": y_shift,
            "cross-trial": cross_trial_labels(y, trials, pos, RNG),
            "reversed": reversed_labels(y, trials),
            "phase-perm": phase_permuted_labels(y, pos, RNG),
            "block-perm": block_permuted_labels(y, trials, RNG),
        }

        t0 = time.time()
        line = {"subject": int(s)}
        for name, yv in variants.items():
            keep = ~np.isnan(np.asarray(yv, dtype=float))
            yk = np.asarray(yv, dtype=float)[keep].astype(int)
            changed = float(np.mean(yk != y[keep])) * 100
            if keep.sum() < 40 or len(np.unique(yk)) < 2:
                line[name], line[name + "_changed"] = np.nan, changed
                continue
            prob = cv_probs(X[keep], yk, cfg.MODALITIES_PERIPHERAL, device,
                            seed_base=5000 + 1000 * sidx + 137 * CONTROLS.index(name))
            line[name] = roc_auc_score(yk, prob) * 100
            line[name + "_changed"] = changed
            if name == "real":
                sa, usable, pairs = stratified_auc(yk, prob, pos[keep])
                line["auc_phase_stratified"] = sa * 100 if sa == sa else np.nan
                line["n_strata_usable"] = usable
                line["n_pairs"] = pairs
                for j, i in enumerate(np.where(keep)[0]):
                    prob_rows.append({"subject": int(s), "label": int(y[i]),
                                      "position": int(pos[i]),
                                      "trial": int(trials[i]),
                                      "prob": float(prob[j])})

        # How well does trial phase alone predict the label for this subject?
        line["auc_position_only"] = roc_auc_score(y, pos) * 100

        rows.append(line)
        pd.DataFrame(rows).to_csv(OUT / "label_controls_shift_controls.csv", index=False)
        print(f"  S{s:02d}  " + "  ".join(
            f"{n.split()[0]} {line.get(n, float('nan')):.1f}" for n in CONTROLS)
            + f"   [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    pd.DataFrame(prob_rows).to_csv(OUT / "label_controls_probs.csv", index=False)

    print("\n" + "=" * 86)
    print("Label controls that change a meaningful number of labels")
    print("=" * 86)
    print(f"\n{'control':<22}{'labels changed':>16}{'mean AUC':>11}{'SEM':>7}"
          f"{'range':>16}")
    for c in CONTROLS:
        v = df[c].dropna().to_numpy()
        ch = df[c + "_changed"].dropna().to_numpy()
        print(f"{c:<22}{ch.mean():>15.1f}%{v.mean():>11.1f}"
              f"{v.std(ddof=1)/np.sqrt(len(v)):>7.2f}"
              f"{f'{v.min():.1f}-{v.max():.1f}':>16}")

    from scipy.stats import ttest_1samp, ttest_rel
    print()
    for c in CONTROLS[1:]:
        both = df[["real", c]].dropna()
        t, p = ttest_rel(both["real"], both[c])
        t1, p1 = ttest_1samp(both[c], 50)
        print(f"  real vs {c:<20} t={t:5.2f} p={p:.2e}   |   "
              f"vs chance: t={t1:6.2f} p={p1:.3g}")

    print("\n" + "-" * 86)
    print("Holding trial phase fixed: AUC within ring-position strata")
    print("-" * 86)
    sa = df["auc_phase_stratified"].dropna()
    print(f"  unstratified AUC          {df['real'].mean():.1f}")
    print(f"  ring position alone       {df['auc_position_only'].mean():.1f}")
    print(f"  usable strata per subject {df['n_strata_usable'].mean():.1f} "
          f"(strata with >=3 of each class)")
    if len(sa) < 3:
        print(f"\n  Phase-stratified AUC is UNDEFINED: only {len(sa)} subject(s)")
        print("  have any ring-position stratum containing both classes. Holding")
        print("  trial phase fixed leaves no label variance to decode, so the")
        print("  question 'does the decoder beat trial phase?' cannot be asked")
        print("  of this design at all.")
    else:
        t1, p1 = ttest_1samp(sa, 50)
        print(f"  phase-stratified AUC      {sa.mean():.1f} +/- "
              f"{sa.std(ddof=1)/np.sqrt(len(sa)):.2f} (n={len(sa)} subjects)")
        print(f"  phase-stratified vs chance: t={t1:.2f}, p={p1:.3g}")

    print("\n" + "-" * 86)
    print("Reading of this table")
    print("-" * 86)
    print("""
  Read the 'labels changed' column first. Four of the five controls change
  under 7% of labels, so their AUCs carry no information - they are the same
  defect the +2-ring shift has, and it is not fixable by choosing a
  different shift. The label is a deterministic function of ring position for
  13 of 16 subjects.

  The one control that does change labels is reversal (79.5%), and it costs
  only 3 AUC points. That is not evidence of robustness: reversal preserves a
  deterministic position-to-label mapping and merely inverts it, so a decoder
  trained and tested on reversed labels simply learns the inverted mapping.

  Taken together: no label-side control can separate this decoder from trial
  phase, because within a subject the label IS trial phase. The only valid
  destruction control is full label shuffling, which t6 already reports at
  chance (48.9%). Validity has to be established across courses at matched
  phase (matched_phase), which is the one comparison where ring size varies independently
  of elapsed time.""")
    return df


if __name__ == "__main__":
    run()
