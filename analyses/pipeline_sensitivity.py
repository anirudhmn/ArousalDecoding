"""Sensitivity of the main results to three pipeline choices.

(a) Half-sham contamination. Condition 2 delivers 50 percent veridical feedback,
    so the control trials are not a clean control. Refit on condition 1 only.
(b) Gamma power. The baseline feature is the mean of a band-passed signal, which
    is close to zero by construction. Recompute it as actual band power and redo
    everything downstream.
(c) Warm-up transient. The first second of every arousal trace is zero by
    construction, so trial-mean arousal correlates with trial length.

Outputs: printed tables only
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress

from common import C, D, OUT, Y, cfg, hard_control, trial_table
from common import FS, ROOT


# --------------------------------------------------------------------------- #
# (a) condition 1 only
# --------------------------------------------------------------------------- #

def part_a(df):
    print("=" * 74)
    print("(a) Dropping the half-sham condition from the trajectory fit")
    print("=" * 74)
    for label, sub in [("cond 1+2 (primary)", df[(df.difficulty == 1)
                                                   & df.condition.isin([1, 2])]),
                       ("cond 1 only (silence)", df[(df.difficulty == 1)
                                                    & (df.condition == 1)])]:
        sub = sub.reset_index(drop=True)
        traj = C.optimal_trajectory(C.performance_surface(sub))
        m = C.deviation_metrics(sub, traj, 1.0)
        r = linregress(m.pct_in_band, m.performance)
        d, dp = C.cohens_d_good_bad(m)
        fitted = (traj["confidence"] >= C.MIN_CONFIDENCE_FOR_FIT).sum()
        print(f"  {label:<24} n={len(sub):3d}  fitted bins={fitted:2d}  "
              f"r={r.rvalue:+.3f} (p={r.pvalue:.4f})  d={d:.2f}")

    # Cross-application: fit on silence, evaluate on half-sham.
    fit = df[(df.difficulty == 1) & (df.condition == 1)].reset_index(drop=True)
    test = df[(df.difficulty == 1) & (df.condition == 2)].reset_index(drop=True)
    traj = C.optimal_trajectory(C.performance_surface(fit))
    m = C.deviation_metrics(test, traj, 1.0)
    r = linregress(m.pct_in_band, m.performance)
    print(f"  {'fit silence -> test half':<24} n={len(test):3d}  "
          f"{'':<15}r={r.rvalue:+.3f} (p={r.pvalue:.4f})")


# --------------------------------------------------------------------------- #
# (b) gamma as actual band power
# --------------------------------------------------------------------------- #

def band_power_baselines():
    """Same as data.baseline_features, but averaging squared amplitude."""
    from arousal.signals import extract_bandpowers_timeseries
    from arousal.config import CH
    ro = D.load_ring_epochs_online()
    calib = ro[ro.condition == cfg.CALIBRATION_CONDITION]
    rows = []
    for s, subj_id in enumerate(sorted(ro.subj_idx.unique())):
        g = calib[calib.subj_idx == subj_id]
        banded = np.concatenate(
            [extract_bandpowers_timeseries(x[None, ...])[0] for x in g["data"]],
            axis=1)
        p = banded ** 2                       # power, not signed amplitude
        rows.append({"subject": s, "subj_id": subj_id,
                     "theta": p[0:64].mean(), "alpha": p[64:128].mean(),
                     "beta": p[128:192].mean(), "gamma": p[192:256].mean(),
                     "hr": banded[256 + CH["HR"] - 64].mean(),
                     "hrv": banded[256 + CH["HRV-pNN35"] - 64].mean(),
                     "resp": banded[256 + CH["RESP"] - 64].mean(),
                     "eda_p": banded[256 + CH["EDA-phasic"] - 64].mean(),
                     "eda_t": banded[256 + CH["EDA-tonic"] - 64].mean(),
                     "pup": banded[[256 + CH["PUP-L"] - 64,
                                    256 + CH["PUP-R"] - 64]].mean()})
    return pd.DataFrame(rows)


def part_b(df):
    print("\n" + "=" * 74)
    print("(b) Recomputing gamma as band power instead of filtered-signal mean")
    print("=" * 74)

    base_new = band_power_baselines()
    print("\nper-subject gamma, old (signal mean) vs new (power):")
    old = df.groupby("subject")["gamma"].first()
    print(f"  old: range [{old.min():.2e}, {old.max():.2e}]")
    print(f"  new: range [{base_new.gamma.min():.4f}, {base_new.gamma.max():.4f}]")
    print(f"  rank correlation between them: "
          f"{pd.Series(old.to_numpy()).corr(base_new.gamma, method='spearman'):+.3f}")

    trials = df.drop(columns=[c for c in df.columns
                              if c.endswith("_z") or c in base_new.columns
                              and c not in ("subject",)], errors="ignore")
    merged = D.attach_baselines(trials, base_new)
    merged["arousal_old_mean"] = df["arousal_old_mean"].to_numpy()

    print("\nYerkes-Dodson interactions with the recomputed features:")
    for f in ("hrv", "gamma"):
        m = Y.fit_interaction_model(merged, f)
        term = f"I(arousal ** 2):{f}_z"
        print(f"  arousal^2 x {f:<6} beta={m.params[term]:+.4f}  "
              f"p={m.pvalues[term]:.4f}")

    hc = hard_control(merged)
    traj = C.optimal_trajectory(C.performance_surface(hc))
    metrics = C.deviation_metrics(hc, traj, 1.0)
    score = -hc.hrv_z.to_numpy() + hc.gamma_z.to_numpy()
    med = np.median(score)
    s_i, t_i = np.where(score > med)[0], np.where(score <= med)[0]
    print(f"\n  group sizes: sensitive {hc.iloc[s_i].subject.nunique()} subjects, "
          f"tolerant {hc.iloc[t_i].subject.nunique()} subjects")
    for label, idx in [("sensitive", s_i), ("tolerant", t_i)]:
        g = metrics.iloc[idx]
        r = linregress(g.pct_in_band, g.performance)
        ds, _ = C.band_width_sweep(hc, traj, idx)
        best = int(np.nanargmax(ds))
        print(f"  {label:<10} r={r.rvalue:+.3f} (p={r.pvalue:.4f})  "
              f"best m={C.BAND_MULTIPLIERS[best]} (d={ds[best]:.2f})")

    # Does the membership change?
    old_score = -df.groupby("subject")["hrv_z"].first() + df.groupby("subject")["gamma_z"].first()
    old_sens = set(old_score[old_score > old_score.median()].index)
    new_sens = set(hc.iloc[s_i].subject.unique())
    print(f"\n  subjects changing group: "
          f"{len(old_sens ^ new_sens)} of 16")


# --------------------------------------------------------------------------- #
# (c) warm-up transient
# --------------------------------------------------------------------------- #

def part_c(df):
    print("\n" + "=" * 74)
    print("(c) Trimming the decoder warm-up from trial-mean arousal")
    print("=" * 74)
    print(f"{'dropped':<10}{'r(len,arousal)':>16}{'beta quad':>12}"
          f"{'p easy':>10}{'p hard':>10}{'opt E/H':>14}")
    for drop_s in (0, 1, 2, 3, 5):
        n = int(drop_s * FS)
        d = df.copy()
        d["arousal"] = [t[n:].mean() if len(t) > n + FS else np.nan
                        for t in d["new_arousal"]]
        d = d.dropna(subset=["arousal"])
        m = Y.fit_yd_model(d, "arousal")
        pv = Y.quadratic_pvalues(m)
        o0, _ = Y.optimum_with_ci(m, 0)
        o1, _ = Y.optimum_with_ci(m, 1)
        r = np.corrcoef(d.performance, d.arousal)[0, 1]
        print(f"{drop_s:<10}{r:>+16.3f}{m.params[Y.QUAD]:>+12.3f}"
              f"{pv[0]:>10.1e}{pv[1]:>10.1e}   {o0:5.1f}/{o1:5.1f}")


if __name__ == "__main__":
    df = trial_table()
    part_a(df)
    part_b(df)
    part_c(df)
