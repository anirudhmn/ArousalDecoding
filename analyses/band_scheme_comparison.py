"""Every control-band scheme scored against the same baseline, out of sample.

Two scripts here produce held-out band results and neither can be compared with
the other as written. ``loso_control_bands.py`` evaluates the group-specific
band widths of the original submission, whose group assignment comes from the
composite sensitivity score. ``individual_reactivity.py`` evaluates bands
derived from each subject's own fitted curve, keyed on the calibration marker
that survives the screen. Both report a correlation between time in band and
flight time on 113 held-out trials, but only the second carries a bootstrap, so
quoting one script's point estimate next to the other's interval would mix an
out-of-sample difference with an in-sample one.

This script recomputes all of them on the same trials and applies one
subject-level bootstrap to every contrast. Three questions follow:

  1  Does any scheme beat the default universal band?
  2  Does any scheme beat a universal band retuned to its best single
     multiplier, which uses no per-subject information at all?
  3  Does the group-width result depend on the withdrawn composite score? The
     same leave-one-subject-out procedure is rerun with the groups defined by
     the calibration marker instead, so the answer does not rest on a grouping
     the manuscript no longer stands behind.

Run after loso_control_bands, calibration_profile and individual_reactivity.

Outputs: results/band_scheme_comparison.csv,
         results/band_scheme_deltas.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from common import C as ctrl
from common import OUT, hard_control, subject_scores, trial_table

RNG = np.random.default_rng(4242)
N_BOOT = 10000
RETUNED = 1.75              # best single multiplier, from band_width_limits
MARKER = "cal_sd"           # the measure that survives the calibration screen


def _profile():
    return pd.read_csv(OUT / "calibration_profile.csv")


def _loso_group_bands(hc, score_by_subject, label):
    """Group-specific widths, with everything refitted per fold.

    ``score_by_subject`` maps subject to a sensitivity score. The median that
    splits the groups, the two multipliers and the trajectory are all fitted on
    the training subjects; the held-out subject is assigned to a group by its
    own score against the training median.
    """
    subjects = sorted(hc.subject.unique())
    held, folds = [], []
    for s in subjects:
        train = hc[hc.subject != s].reset_index(drop=True)
        test = hc[hc.subject == s].reset_index(drop=True)
        if not len(test):
            continue

        traj = ctrl.optimal_trajectory(ctrl.performance_surface(train))
        tr_scores = np.array([score_by_subject[t] for t in train.subject])
        median = np.median([score_by_subject[t] for t in train.subject.unique()])
        sens_i = np.where(tr_scores > median)[0]
        tol_i = np.where(tr_scores <= median)[0]

        m_sens = _best_multiplier(train, traj, sens_i)
        m_tol = _best_multiplier(train, traj, tol_i)
        if not np.isfinite(m_sens) or not np.isfinite(m_tol):
            continue
        m = m_sens if score_by_subject[s] > median else m_tol

        d = ctrl.deviation_metrics(test, traj, m)
        d["subject"] = s
        held.append(d)
        folds.append(dict(scheme=label, subject=s, multiplier=m,
                          m_sens=m_sens, m_tol=m_tol, median=median))
    return pd.concat(held, ignore_index=True), pd.DataFrame(folds)


def _best_multiplier(train, traj, idx):
    """Multiplier maximising Cohen's d between good and bad training trials.

    Same objective and same grid as ``loso_control_bands.py``, so that the two
    scripts select identically when given identical groups.
    """
    ds, _ = ctrl.band_width_sweep(train, traj, idx)
    if not np.any(np.isfinite(ds)):
        return np.nan
    return ctrl.BAND_MULTIPLIERS[int(np.nanargmax(ds))]


def _fixed_multiplier_band(hc, mult):
    """Leave-one-subject-out scoring of a single multiplier for everyone."""
    held = []
    for s in sorted(hc.subject.unique()):
        train = hc[hc.subject != s].reset_index(drop=True)
        test = hc[hc.subject == s].reset_index(drop=True)
        if not len(test):
            continue
        traj = ctrl.optimal_trajectory(ctrl.performance_surface(train))
        d = ctrl.deviation_metrics(test, traj, mult)
        d["subject"] = s
        held.append(d)
    return pd.concat(held, ignore_index=True)


def _bootstrap(frames, contrasts):
    """Subject-level bootstrap of the difference in r for each contrast."""
    base = next(iter(frames.values()))
    subj = base.subject.to_numpy()
    uniq = np.unique(subj)
    index = {s: np.where(subj == s)[0] for s in uniq}

    picks = [RNG.choice(uniq, uniq.size, replace=True) for _ in range(N_BOOT)]
    rows = []
    for name, a, b in contrasts:
        fa, fb = frames[a], frames[b]
        ax, ay = fa.pct_in_band.to_numpy(), fa.performance.to_numpy()
        bx, by = fb.pct_in_band.to_numpy(), fb.performance.to_numpy()
        out = np.empty(N_BOOT)
        for i, pick in enumerate(picks):
            k = np.concatenate([index[s] for s in pick])
            out[i] = (np.corrcoef(bx[k], by[k])[0, 1]
                      - np.corrcoef(ax[k], ay[k])[0, 1])
        lo, hi = np.percentile(out, [2.5, 97.5])
        p = 2 * min((out <= 0).mean(), (out >= 0).mean())
        rows.append(dict(contrast=name, delta_r=out.mean(), ci_lo=lo, ci_hi=hi,
                         p=min(p, 1.0)))
    return pd.DataFrame(rows)


def run():
    df = trial_table()
    hc = hard_control(df)
    prof = _profile()

    print("=" * 82)
    print("Control-band schemes, all scored on the same held-out trials")
    print("=" * 82)
    print(f"\n{len(hc)} hard-course control trials from "
          f"{hc.subject.nunique()} subjects.")

    frames = {}

    # 1. the two universal bands
    frames["universal x1.0"] = _fixed_multiplier_band(hc, 1.0)
    frames[f"universal x{RETUNED} (retuned)"] = _fixed_multiplier_band(hc, RETUNED)

    # 2. group widths under the composite score of the original submission
    composite = subject_scores(df).to_dict()
    frames["group widths (composite score)"], f1 = _loso_group_bands(
        hc, composite, "composite")

    # 3. the same scheme with the surviving calibration marker. A low
    #    calibration variability goes with the sharply peaked curve, so the
    #    score is negated to keep "higher means more sensitive".
    marker = {int(s): -float(v) for s, v in zip(prof.subject, prof[MARKER])}
    frames["group widths (calibration marker)"], f2 = _loso_group_bands(
        hc, marker, "calibration marker")

    # 4. the curve-derived bands, read back from individual_reactivity
    for key, label in [("centre", "centre from fitted curve"),
                       ("width", "width from fitted curve")]:
        path = OUT / f"individual_reactivity_band_{key}.csv"
        if path.exists():
            frames[label] = pd.read_csv(path)

    print(f"\n{'scheme':<38}{'n':>5}{'r':>9}{'p':>10}{'Cohen d':>10}")
    summary = []
    for name, frame in frames.items():
        r, p = pearsonr(frame.pct_in_band, frame.performance)
        d = ctrl.cohens_d_good_bad(frame)[0]
        print(f"{name:<38}{len(frame):>5}{r:>+9.3f}{p:>10.4f}{d:>10.2f}")
        summary.append(dict(scheme=name, n=len(frame), r=r, p=p, cohens_d=d))

    print("\nMultipliers selected per fold:")
    for f, tag in [(f1, "composite score"), (f2, "calibration marker")]:
        print(f"  {tag:<20} sensitive "
              f"{sorted(set(f.m_sens))}\n{'':<23}tolerant  {sorted(set(f.m_tol))}")

    default = "universal x1.0"
    retuned = f"universal x{RETUNED} (retuned)"
    contrasts = []
    for name in frames:
        if name != default:
            contrasts.append((f"{name} vs default", default, name))
    for name in frames:
        if name not in (default, retuned):
            contrasts.append((f"{name} vs retuned", retuned, name))

    deltas = _bootstrap(frames, contrasts)
    print(f"\n{'contrast':<52}{'delta r':>9}{'95% CI':>21}{'p':>8}")
    for row in deltas.itertuples():
        print(f"{row.contrast:<52}{row.delta_r:>+9.3f}"
              f"   [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]{row.p:>8.3f}")

    print("\nReading. Compare the two blocks. Schemes that beat the default")
    print("band do not separate from a universal band retuned to a single")
    print("multiplier, which uses no per-subject information. The two group")
    print("rows also show whether the result depends on the withdrawn")
    print("composite score.")

    pd.DataFrame(summary).to_csv(OUT / "band_scheme_comparison.csv", index=False)
    deltas.to_csv(OUT / "band_scheme_deltas.csv", index=False)
    pd.concat([f1, f2], ignore_index=True).to_csv(
        OUT / "band_scheme_folds.csv", index=False)
    print(f"\nwrote three CSVs to {OUT}")


if __name__ == "__main__":
    run()
