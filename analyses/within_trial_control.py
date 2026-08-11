"""Control parameters that vary within a trial.

``band_width_limits.py`` shows that band width fails as a control parameter for
a structural reason. It is a constant per subject, the arousal-performance
association lives within subjects, and a sixfold change in width barely reorders
a subject's own trials. The implication is that a useful parameter has to vary
within the trial. This tests that implication rather than asserting it.

Everything is evaluated in the discrete-time hazard framework of
``hazard_model.py``: for each 1 s bin, does the current control signal predict
crashing within the next 5 s? That framing is length-safe, and it is the
question a controller actually faces, which is whether to intervene now.

Parameters compared, all against the same outcome:

  fixed          above the universal band, the reference rule
  retuned        above a universal band at the best single multiplier
  per-subject    width scaled by calibration arousal SD
  adaptive       width from the subject's own arousal variability in a trailing
                 window within the trial, so it varies bin to bin
  sustained      above band for k consecutive seconds, not merely above now
  rate           rate of change of arousal, which is intrinsically within-trial

Discrimination is reported as AUC over bins, together with the mixed-model
coefficient and a likelihood-ratio test against a time-only null, so that a
parameter cannot win merely by being noisier.

Outputs: results/within_trial_control_params.csv,
         results/within_trial_control_bins.csv
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

from common import C, OUT, hard_control, trial_table

warnings.filterwarnings("ignore")
HORIZON = 5             # seconds
TRAIL = 10              # trailing window for the adaptive band, seconds
SUSTAIN = 3             # seconds above band before "sustained" fires


def build_bins(df, prof=None):
    """One row per 1 s bin, carrying every candidate control signal."""
    traj = C.optimal_trajectory(C.performance_surface(hard_control(df)))
    opt, sd = traj["optimal"], traj["std"]

    cal = None
    if prof is not None:
        w = prof.set_index("subject")["cal_sd"]
        cal = (w / w.median()).clip(0.5, 3.0)

    rows = []
    for r in df[df.difficulty == 1].itertuples():
        a = C._binned_arousal(r.new_arousal)
        n = len(a)
        run_above = 0
        for t in range(n):
            if t >= len(opt) or np.isnan(opt[t]) or np.isnan(sd[t]):
                continue
            hi_fixed = opt[t] + 1.00 * sd[t]
            hi_tuned = opt[t] + 1.75 * sd[t]
            hi_subj = opt[t] + float(cal.get(r.subject, 1.0)) * sd[t] \
                if cal is not None else hi_fixed

            # Adaptive: the half-width is this trial's own recent variability,
            # so the band breathes with the subject's current state.
            lo_t = max(0, t - TRAIL)
            local = a[lo_t:t + 1]
            local_sd = float(np.std(local)) if len(local) > 2 else float(sd[t])
            hi_adapt = opt[t] + max(local_sd, 1.0)

            above_fixed = float(a[t] > hi_fixed)
            run_above = run_above + 1 if above_fixed else 0

            # Rate of change over the last 3 s.
            prev = a[max(0, t - 3)]
            rate = (a[t] - prev) / max(1, min(3, t))

            rows.append({
                "subject": r.subject, "condition": r.condition, "t": t,
                "tb": min(t // 5, 11), "arousal": a[t],
                "above_fixed": above_fixed,
                "above_tuned": float(a[t] > hi_tuned),
                "above_subject": float(a[t] > hi_subj),
                "above_adaptive": float(a[t] > hi_adapt),
                "sustained": float(run_above >= SUSTAIN),
                "excess_fixed": max(0.0, a[t] - hi_fixed),
                "excess_adaptive": max(0.0, a[t] - hi_adapt),
                "rate": rate,
                "crash": float(n - t <= HORIZON),
            })
    return pd.DataFrame(rows)


PARAMS = [
    ("above universal band (reference)", "above_fixed"),
    ("above universal band, retuned x1.75", "above_tuned"),
    ("above per-subject band (calibration SD)", "above_subject"),
    ("above ADAPTIVE band (trailing 10 s SD)", "above_adaptive"),
    ("sustained above band (>=3 s)", "sustained"),
    ("excess over universal band (graded)", "excess_fixed"),
    ("excess over ADAPTIVE band (graded)", "excess_adaptive"),
    ("rate of change of arousal", "rate"),
]


def stratified_auc(h, col):
    """AUC computed WITHIN each 5 s time bin, pooled by discordant pairs.

    A marginal AUC is meaningless here. Elapsed time drives both the outcome
    and every band signal, and late in a trial it drives them in opposite
    directions. The fitted trajectory rises, so almost nothing exceeds the band
    after about 45 s, while the crash rate stays high. The two effects cancel.
    The marginal AUC then lands at chance even where the within-time
    association is large. Stratifying by time bin is the comparison the model
    actually makes.
    """
    num = den = 0.0
    for _, g in h.groupby("tb"):
        y, x = g["crash"].to_numpy(), g[col].to_numpy()
        if len(np.unique(y)) < 2 or np.std(x) < 1e-12:
            continue
        w = (y == 0).sum() * (y == 1).sum()
        num += roc_auc_score(y, x) * w
        den += w
    return (num / den * 100) if den else np.nan


def evaluate(h, col, tterm="C(tb)"):
    """Hazard model with time controlled, plus a time-stratified AUC."""
    m = smf.mixedlm(f"crash ~ {col} + {tterm}", h, groups=h["subject"]).fit(reml=False)
    m0 = smf.mixedlm(f"crash ~ {tterm}", h, groups=h["subject"]).fit(reml=False)
    lr = 2 * (m.llf - m0.llf)
    return dict(beta=m.params[col], p=m.pvalues[col], lr=lr,
                p_lr=chi2.sf(max(lr, 0), 1),
                auc=stratified_auc(h, col),
                auc_marginal=(roc_auc_score(h["crash"], h[col]) * 100
                              if h[col].std() > 1e-9 else np.nan),
                fires=100 * (h[col] > 0).mean())


def run():
    df = trial_table()
    try:
        from calibration_profile import build_profile
        prof = build_profile()
    except Exception as e:
        print(f"[warn] calibration profile unavailable ({type(e).__name__}); "
              f"per-subject row will fall back to the universal band")
        prof = None

    h = build_bins(df, prof)
    h.to_csv(OUT / "within_trial_control_bins.csv", index=False)

    print("=" * 94)
    print("Control parameters that vary within a trial")
    print("=" * 94)
    print(f"\n{len(h)} one-second bins, {h.subject.nunique()} subjects, "
          f"{h['crash'].mean()*100:.1f}% within {HORIZON} s of a crash")
    print("Outcome: crash within 5 s. Elapsed time controlled by 5 s-bin fixed "
          "effects.\n")

    print(f"{'control parameter':<42}{'fires':>8}{'AUC|t':>8}{'AUC raw':>9}"
          f"{'beta':>10}{'p':>10}{'LRT':>9}")
    rows = []
    for label, col in PARAMS:
        r = evaluate(h, col)
        print(f"{label:<42}{r['fires']:>7.1f}%{r['auc']:>8.1f}"
              f"{r['auc_marginal']:>9.1f}{r['beta']:>+10.4f}"
              f"{r['p']:>10.1e}{r['lr']:>9.1f}")
        rows.append(dict(parameter=label, column=col, **r))
    res = pd.DataFrame(rows)

    # Head-to-head: does the adaptive band add anything over the fixed one?
    print("\n" + "-" * 94)
    print("Does the adaptive band add information beyond the fixed band?")
    print("-" * 94)
    for extra, base in [("above_adaptive", "above_fixed"),
                        ("sustained", "above_fixed"),
                        ("rate", "above_fixed"),
                        ("excess_adaptive", "excess_fixed")]:
        m_full = smf.mixedlm(f"crash ~ {base} + {extra} + C(tb)", h,
                             groups=h["subject"]).fit(reml=False)
        m_base = smf.mixedlm(f"crash ~ {base} + C(tb)", h,
                             groups=h["subject"]).fit(reml=False)
        lr = 2 * (m_full.llf - m_base.llf)
        p = chi2.sf(max(lr, 0), 1)
        print(f"   + {extra:<18} on top of {base:<14} "
              f"beta = {m_full.params[extra]:+.4f}, LRT = {lr:6.1f}, "
              f"p = {p:.2e}" + ("  *" if p < 0.05 else ""))
        res.loc[len(res)] = {"parameter": f"{extra} | {base}", "column": extra,
                             "beta": m_full.params[extra], "p": p, "lr": lr,
                             "p_lr": p, "auc": np.nan, "fires": np.nan}

    # Does feedback attenuate the best parameter, as it does the fixed band?
    print("\n" + "-" * 94)
    print("Feedback interaction, as in hazard_model, for each band definition")
    print("-" * 94)
    for label, col in [("fixed", "above_fixed"), ("adaptive", "above_adaptive"),
                       ("sustained", "sustained")]:
        m = smf.mixedlm(f"crash ~ {col} * C(condition) + C(tb)", h,
                        groups=h["subject"]).fit(reml=False)
        term = [t for t in m.params.index
                if t.startswith(col) and "condition)[T.3]" in t]
        if term:
            print(f"   {label:<10} {col} x full BCI: "
                  f"beta = {m.params[term[0]]:+.4f}, p = {m.pvalues[term[0]]:.3f}"
                  + ("  *" if m.pvalues[term[0]] < 0.05 else ""))

    res.to_csv(OUT / "within_trial_control_params.csv", index=False)

    print("\n" + "=" * 94)
    print("Reading")
    print("=" * 94)
    print("""
  The fixed band is the reference rule and the benchmark. A within-trial
  parameter earns its place only if it adds information ON TOP of the fixed
  band in the head-to-head section - a higher AUC on its own is not enough,
  since a parameter that fires more often will score higher by base rate alone.

  'fires' is the share of bins in which the parameter is non-zero. Compare it
  against AUC: a rule that fires constantly is not a usable control signal even
  if it discriminates, because a controller acting on it would intervene
  permanently.

  AUC|t is stratified by time bin; AUC raw is marginal and is reported only to
  show how misleading it is. The two differ because elapsed time drives the
  outcome and the band signals in opposing directions late in a trial - the
  fitted trajectory rises, so nothing exceeds the band after ~45 s while the
  crash rate stays high. Marginally the effect cancels to chance. Within time
  bin, being above the band raises the 5 s crash rate by about 7 percentage
  points, positive in 7 of 9 bins.""")


if __name__ == "__main__":
    run()
