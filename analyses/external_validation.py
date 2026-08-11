"""External validation of the decoded index against raw physiology.

A correlation between decoded arousal and a raw physiological channel can be
computed in several defensible ways, and they do not agree. This enumerates the
plausible methods, applies each to all five signals, and reports them side by
side so that one method can be chosen and stated explicitly.

The candidate methods differ in what they treat as an observation:

  within-trial       correlate the two traces inside each trial, then average
                     the per-trial r values
  fisher-z           the same, but average after a Fisher z transform
  pooled-samples     concatenate every sample from every trial and correlate
                     once, which ignores clustering entirely and inflates n to
                     roughly 5e6
  trial-means        one point per trial, correlating the trial means
  subject-means      one point per subject
  within-subject     correlate within each subject across trials, then average
  mixed-model        trial means with a random intercept per subject, converted
                     to a partial correlation

Raw signals are min-max scaled within each trial by ``_raw_metric_traces``, so
the trial-mean methods inherit the level-stripping documented in
``scaling_parity.py``. That is reported rather than worked around, because it
constrains which methods are meaningful at all.

Outputs: results/external_validation.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

from common import OUT, trial_table

SIGNALS = {
    "HR": "hr_signal",
    "HRV-pNN35": "hrv_signal",
    "EDA-phasic": "eda_phasic",
    "EDA-tonic": "eda_tonic",
    "Pupil": "pupil_signal",
}
TARGET = {"HR": 0.71, "HRV-pNN35": -0.58}     # reference values under check


def _pairs(df, col):
    """Per-trial (arousal, signal) trace pairs, warm-up samples dropped."""
    out = []
    for r in df.itertuples():
        a, b = np.asarray(r.new_arousal, float), np.asarray(getattr(r, col), float)
        n = min(len(a), len(b))
        a, b = a[256:n], b[256:n]             # first second is decoder warm-up
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 512 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9:
            out.append((r.subject, a[m], b[m]))
    return out


def within_trial(pairs, fisher=False):
    rs = np.array([pearsonr(a, b)[0] for _, a, b in pairs])
    if not fisher:
        return rs.mean(), len(rs)
    z = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    return float(np.tanh(z.mean())), len(rs)


def pooled_samples(pairs):
    a = np.concatenate([x for _, x, _ in pairs])
    b = np.concatenate([y for _, _, y in pairs])
    return pearsonr(a, b)[0], len(a)


def trial_means(pairs):
    a = np.array([x.mean() for _, x, _ in pairs])
    b = np.array([y.mean() for _, _, y in pairs])
    return pearsonr(a, b)[0], len(a)


def subject_means(pairs):
    d = pd.DataFrame({"s": [s for s, _, _ in pairs],
                      "a": [x.mean() for _, x, _ in pairs],
                      "b": [y.mean() for _, _, y in pairs]}).groupby("s").mean()
    return pearsonr(d["a"], d["b"])[0], len(d)


def within_subject(pairs):
    d = pd.DataFrame({"s": [s for s, _, _ in pairs],
                      "a": [x.mean() for _, x, _ in pairs],
                      "b": [y.mean() for _, _, y in pairs]})
    rs = [pearsonr(g["a"], g["b"])[0] for _, g in d.groupby("s") if len(g) > 3]
    return float(np.mean(rs)), len(rs)


def mixed_model(pairs):
    """Partial correlation from a random-intercept model on trial means."""
    d = pd.DataFrame({"s": [s for s, _, _ in pairs],
                      "a": [x.mean() for _, x, _ in pairs],
                      "b": [y.mean() for _, _, y in pairs]})
    m = smf.mixedlm("a ~ b", d, groups=d["s"]).fit(reml=False)
    t = m.tvalues["b"]
    dfree = len(d) - d["s"].nunique() - 1
    return float(np.sign(t) * np.sqrt(t ** 2 / (t ** 2 + dfree))), len(d)


METHODS = {
    "within-trial (current)": lambda p: within_trial(p, False),
    "within-trial, Fisher z": lambda p: within_trial(p, True),
    "pooled samples": pooled_samples,
    "trial means": trial_means,
    "subject means": subject_means,
    "within-subject across trials": within_subject,
    "mixed model (partial r)": mixed_model,
}


def run():
    df = trial_table()
    print("=" * 92)
    print("External validation of the decoded index against raw physiology")
    print("=" * 92)
    print(f"\n{len(df)} trials. Reference values under check: HR r = +0.71, HRV r = -0.58.")
    print("Raw traces are min-max scaled within each trial by _raw_metric_traces,")
    print("so any trial-mean method sees shape, not level (see scaling_parity).")

    rows = []
    print(f"\n{'method':<30}" + "".join(f"{k:>13}" for k in SIGNALS))
    for mname, fn in METHODS.items():
        cells = []
        for sig, col in SIGNALS.items():
            r, n = fn(_pairs(df, col))
            cells.append(r)
            rows.append(dict(method=mname, signal=sig, r=r, n=n))
        print(f"{mname:<30}" + "".join(f"{c:>+13.3f}" for c in cells))

    res = pd.DataFrame(rows)

    print("\n" + "-" * 92)
    print("Does any method reproduce the reference values?")
    print("-" * 92)
    best = None
    for mname in METHODS:
        sub = res[res.method == mname].set_index("signal")["r"]
        err = sum(abs(sub[k] - v) for k, v in TARGET.items())
        print(f"   {mname:<30} HR {sub['HR']:+.3f} (target +0.71), "
              f"HRV {sub['HRV-pNN35']:+.3f} (target -0.58), "
              f"total error {err:.3f}")
        if best is None or err < best[1]:
            best = (mname, err)
    print(f"\n   closest: {best[0]}, total absolute error {best[1]:.3f}")
    print("   No method reproduces the reference pair." if best[1] > 0.2
          else "   This method reproduces the reference pair.")

    # Non-parametric check and significance for the recommended method.
    print("\n" + "-" * 92)
    print("Recommended reporting: within-trial correlation, averaged over trials,")
    print("with a one-sample test across subject means and a rank check.")
    print("-" * 92)
    print(f"\n{'signal':<14}{'mean r':>9}{'95% CI':>20}{'p':>11}{'Spearman':>11}")
    from scipy.stats import ttest_1samp
    for sig, col in SIGNALS.items():
        pairs = _pairs(df, col)
        per_trial = pd.DataFrame({
            "s": [s for s, _, _ in pairs],
            "r": [pearsonr(a, b)[0] for _, a, b in pairs],
            "rho": [spearmanr(a, b)[0] for _, a, b in pairs]})
        per_subj = per_trial.groupby("s")["r"].mean()
        t, p = ttest_1samp(per_subj, 0)
        lo = per_subj.mean() - 1.96 * per_subj.std(ddof=1) / np.sqrt(len(per_subj))
        hi = per_subj.mean() + 1.96 * per_subj.std(ddof=1) / np.sqrt(len(per_subj))
        print(f"{sig:<14}{per_trial['r'].mean():>+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{p:>11.2e}"
              f"{per_trial['rho'].mean():>+11.3f}")
        rows.append(dict(method="RECOMMENDED within-trial", signal=sig,
                         r=per_trial["r"].mean(), n=len(per_trial),
                         ci_lo=lo, ci_hi=hi, p=p,
                         spearman=per_trial["rho"].mean()))

    pd.DataFrame(rows).to_csv(OUT / "external_validation.csv", index=False)
    print(f"\nwrote {OUT/'external_validation.csv'}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
