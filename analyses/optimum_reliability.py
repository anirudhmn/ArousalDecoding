"""Is there a stable individual difference to personalise on?

Asking which baseline measure predicts who needs a wider band presupposes
something that is rarely tested: that subjects differ stably in their optimal
arousal at all. If the per-subject optimum is not reliable within a subject,
then no baseline marker can predict it, and no median split can rescue it.

Each step is prior to the next:

  A  Is between-subject variation in the optimum real? Split each subject's
     trials in half, estimate their optimum in each half independently, and
     correlate the halves across subjects. This is a reliability ceiling. No
     predictor can beat it.

  B  Does a mixed model need subject-specific curve shape? Likelihood-ratio test
     of a random slope on arousal and arousal squared against a random intercept
     only. The same question asked parametrically, using all trials at once.

  C  Is a subject's best band width reliable? The same split-half logic applied
     to the quantity that the personalised control band varies.

  D  Does a subject's own optimum, fitted on control trials, beat the group
     optimum on their held-out full-BCI trials? The applied version.

Outputs: results/optimum_reliability.csv,
         results/optimum_reliability_randomslope.csv,
         results/optimum_reliability_bandwidth.csv,
         results/optimum_reliability_transfer.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, pearsonr, spearmanr

from common import C, OUT, cfg, hard_control, trial_table

RNG = np.random.default_rng(24)
N_BOOT = 5000
MIN_TRIALS = 8          # per subject, to attempt a split-half


# --------------------------------------------------------------------------- #
# Per-subject summaries of "where this person performs best"
# --------------------------------------------------------------------------- #

def vertex(g):
    """Optimum from that subject's own quadratic. Direct but unstable."""
    if len(g) < 5 or g["arousal"].std() < 1e-6:
        return np.nan
    try:
        b = np.polyfit(g["arousal"], g["performance"], 2)
    except Exception:
        return np.nan
    if abs(b[0]) < 1e-12 or b[0] >= 0:      # no interior maximum
        return np.nan
    return -b[1] / (2 * b[0])


def top_quartile_arousal(g):
    """Mean arousal on this subject's best trials. Robust, assumption-free."""
    if len(g) < 4:
        return np.nan
    thr = g["performance"].quantile(0.75)
    sel = g[g["performance"] >= thr]
    return sel["arousal"].mean() if len(sel) else np.nan


def performance_weighted(g):
    """Performance-weighted mean arousal, a smooth version of the above."""
    if len(g) < 4:
        return np.nan
    w = g["performance"] - g["performance"].min()
    return np.average(g["arousal"], weights=w) if w.sum() > 0 else np.nan


ESTIMATORS = {"quadratic vertex": vertex,
              "top-quartile arousal": top_quartile_arousal,
              "performance-weighted arousal": performance_weighted}


# --------------------------------------------------------------------------- #
# A. split-half reliability of the optimum
# --------------------------------------------------------------------------- #

def split_half(df, estimator, n_rep=200):
    """Correlate independent estimates from random halves, across subjects.

    Repeated over random splits, then Spearman-Brown corrected to the
    full-length reliability that a single-session estimate would rely on.
    """
    subs = [s for s, g in df.groupby("subject") if len(g) >= MIN_TRIALS]
    rs = []
    for _ in range(n_rep):
        a, b = [], []
        for s in subs:
            g = df[df.subject == s].sample(frac=1.0, random_state=RNG.integers(1e9))
            h1, h2 = g.iloc[::2], g.iloc[1::2]
            va, vb = estimator(h1), estimator(h2)
            if np.isfinite(va) and np.isfinite(vb):
                a.append(va)
                b.append(vb)
        if len(a) >= 6:
            rs.append(pearsonr(a, b)[0])
    if not rs:
        return dict(n=0, r=np.nan, sb=np.nan, lo=np.nan, hi=np.nan)
    rs = np.array(rs)
    r = float(np.mean(rs))
    sb = 2 * r / (1 + r) if r > -1 else np.nan       # Spearman-Brown
    return dict(n=len(subs), r=r, sb=sb,
                lo=float(np.percentile(rs, 2.5)),
                hi=float(np.percentile(rs, 97.5)))


def part_a(df):
    print("=" * 88)
    print("(A) is the per-subject optimum reliable at all?")
    print("=" * 88)
    print(f"\nSplit each subject's trials into random halves, estimate the optimum")
    print(f"independently in each, correlate across subjects. 200 random splits.")
    print(f"\n{'estimator':<32}{'subjects':>10}{'split-half r':>15}"
          f"{'95% range':>20}{'full-length':>13}")
    rows = []
    for name, fn in ESTIMATORS.items():
        for label, sub in [("all trials", df),
                           ("hard course", df[df.difficulty == 1])]:
            res = split_half(sub, fn)
            rows.append(dict(estimator=name, subset=label, **res))
            print(f"{name + ' [' + label + ']':<32}{res['n']:>10}"
                  f"{res['r']:>+15.3f}"
                  f"{f'[{res['lo']:+.2f}, {res['hi']:+.2f}]':>20}"
                  f"{res['sb']:>+13.3f}")
    print("\nA reliability near zero means the per-subject optimum is mostly noise,")
    print("and no baseline measure could predict it however well chosen.")
    return pd.DataFrame(rows)


def part_a2(df):
    """Is the 'optimum' anything more than that subject's overall arousal level?

    A subject who simply runs higher will have a higher optimum on any
    performance-weighted estimator, purely arithmetically. Unless the optimum
    is reliable AFTER removing the subject's own mean, there is no
    person-specific optimum to personalise on, only a person-specific level,
    which the per-subject rescaling already handles.
    """
    print("\n" + "-" * 88)
    print("Is that reliability anything more than the subject's overall level?")
    print("-" * 88)

    per = df.groupby("subject").apply(
        lambda g: pd.Series({"mean_arousal": g["arousal"].mean(),
                             **{k: fn(g) for k, fn in ESTIMATORS.items()}}),
        include_groups=False)

    print("\n  correlation with that subject's mean arousal, across subjects:")
    for k in ESTIMATORS:
        v = per[[k, "mean_arousal"]].dropna()
        if len(v) > 4:
            r, p = pearsonr(v["mean_arousal"], v[k])
            print(f"     {k:<32} r = {r:+.3f}  (p = {p:.2g})")

    base = split_half(df, lambda g: g["arousal"].mean() if len(g) >= 4 else np.nan)
    print(f"\n  split-half reliability of subject MEAN arousal, which uses no")
    print(f"  performance information at all: r = {base['r']:+.3f} "
          f"(full-length {base['sb']:+.3f})")

    print("\n  reliability of each optimum after subtracting that subject's mean:")
    rows = []
    for name, fn in ESTIMATORS.items():
        def resid(g, fn=fn):
            v = fn(g)
            return v - g["arousal"].mean() if np.isfinite(v) else np.nan
        r = split_half(df, resid)
        rows.append(dict(estimator=name, kind="residualised", **r))
        print(f"     {name:<32} r = {r['r']:+.3f}  "
              f"95% [{r['lo']:+.2f}, {r['hi']:+.2f}]")

    print("\n  If the residualised reliability includes zero, the stable individual")
    print("  difference is arousal LEVEL, not an individual optimum, and the")
    print("  per-subject rescaling in the decoder already normalises level.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# B. does the model need subject-specific curve shape?
# --------------------------------------------------------------------------- #

def part_b(df):
    print("\n" + "=" * 88)
    print("(B) does a random slope on arousal improve the model?")
    print("=" * 88)

    d = df.copy()
    d["a"] = d["arousal"] / 10.0             # rescale for convergence
    d["a2"] = d["a"] ** 2
    base = "performance ~ C(difficulty)*a + C(difficulty)*a2 + C(condition)"

    fits, rows = {}, []
    for name, re_form in [("random intercept only", "~1"),
                          ("+ random slope on arousal", "~a"),
                          ("+ random slope on arousal and arousal^2", "~a + a2")]:
        try:
            m = smf.mixedlm(base, d, groups=d["subject"],
                            re_formula=re_form).fit(reml=False)
            fits[name] = m
            rows.append(dict(model=name, llf=m.llf, aic=m.aic,
                             converged=bool(m.converged)))
            print(f"  {name:<42} logL {m.llf:10.2f}   AIC {m.aic:9.1f}"
                  f"   {'converged' if m.converged else 'DID NOT CONVERGE'}")
        except Exception as e:
            print(f"  {name:<42} failed: {type(e).__name__}")

    names = list(fits)
    for i in range(len(names) - 1):
        a, b = fits[names[i]], fits[names[i + 1]]
        k = 2 if i == 0 else 3               # extra variance/covariance params
        stat = 2 * (b.llf - a.llf)
        p_naive = chi2.sf(max(stat, 0), k)
        # Variance components sit on a boundary; the null is a 50:50 mixture,
        # so the naive p-value is conservative. Both are reported.
        p_mix = 0.5 * chi2.sf(max(stat, 0), k) + 0.5 * chi2.sf(max(stat, 0), k - 1)
        print(f"\n  LRT {names[i]} -> {names[i+1]}:")
        print(f"     chi2({k}) = {stat:.2f}, p = {p_naive:.4f} "
              f"(boundary-corrected p = {p_mix:.4f})")
        rows.append(dict(model=f"LRT {names[i]} -> {names[i+1]}",
                         llf=stat, aic=np.nan, converged=np.nan,
                         p=p_naive, p_boundary=p_mix))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# C. is a subject's best band width reliable?
# --------------------------------------------------------------------------- #

def best_multiplier(sub_df, traj):
    """Multiplier maximising the in-band/performance association for a subject."""
    best, best_r = np.nan, -np.inf
    for m in C.BAND_MULTIPLIERS:
        met = C.deviation_metrics(sub_df, traj, m)
        if met["pct_in_band"].std() < 1e-9 or len(met) < 4:
            continue
        r = pearsonr(met["pct_in_band"], sub_df["performance"])[0]
        if np.isfinite(r) and r > best_r:
            best, best_r = m, r
    return best


def part_c(df):
    print("\n" + "=" * 88)
    print("(C) is a subject's own best band width reliable?")
    print("=" * 88)

    hc = df[df.difficulty == 1].reset_index(drop=True)
    traj = C.optimal_trajectory(C.performance_surface(hard_control(df)))

    rows = []
    for s, g in hc.groupby("subject"):
        if len(g) < MIN_TRIALS:
            continue
        g = g.sample(frac=1.0, random_state=int(s)).reset_index(drop=True)
        m1 = best_multiplier(g.iloc[::2].reset_index(drop=True), traj)
        m2 = best_multiplier(g.iloc[1::2].reset_index(drop=True), traj)
        rows.append(dict(subject=s, n_trials=len(g), half1=m1, half2=m2))

    r = pd.DataFrame(rows).dropna()
    print(f"\n  {len(r)} subjects with >= {MIN_TRIALS} hard-course trials")
    if len(r) >= 5:
        rho, p = spearmanr(r["half1"], r["half2"])
        pr, pp = pearsonr(r["half1"], r["half2"])
        print(r.to_string(index=False))
        print(f"\n  half-to-half agreement: Spearman rho = {rho:+.3f} (p = {p:.3f}), "
              f"Pearson r = {pr:+.3f} (p = {pp:.3f})")
        print(f"  identical choice in {100*(r['half1']==r['half2']).mean():.0f}% of subjects; "
              f"mean |difference| = {(r['half1']-r['half2']).abs().mean():.2f}")
        print(f"  (the grid spans {min(C.BAND_MULTIPLIERS)} to "
              f"{max(C.BAND_MULTIPLIERS)} in steps of 0.25)")
    else:
        print("  too few subjects to assess")
    return r


# --------------------------------------------------------------------------- #
# D. does a subject's own optimum transfer to their BCI trials?
# --------------------------------------------------------------------------- #

def part_d(df):
    print("\n" + "=" * 88)
    print("(D) own optimum vs group optimum on held-out full-BCI trials")
    print("=" * 88)

    fit = df[(df.difficulty == 1) & (df.condition.isin(cfg.CONTROL_CONDITIONS))]
    test = df[(df.difficulty == 1) & (df.condition == cfg.FEEDBACK_CONDITION)]
    group_opt = top_quartile_arousal(fit)

    rows = []
    for s, g in test.groupby("subject"):
        own = fit[fit.subject == s]
        if len(own) < 4 or len(g) < 2:
            continue
        o = top_quartile_arousal(own)
        if not np.isfinite(o):
            continue
        for r in g.itertuples():
            rows.append(dict(subject=s, performance=r.performance,
                             dev_own=abs(r.arousal - o),
                             dev_group=abs(r.arousal - group_opt)))
    d = pd.DataFrame(rows)
    print(f"\n  group optimum {group_opt:.1f}; {len(d)} held-out full-BCI trials "
          f"from {d.subject.nunique()} subjects")

    if len(d) < 20:
        print("  too few held-out trials")
        return d
    for col, label in [("dev_own", "deviation from OWN optimum"),
                       ("dev_group", "deviation from GROUP optimum")]:
        m = smf.mixedlm(f"performance ~ {col}", d, groups=d["subject"]).fit(reml=False)
        r = pearsonr(d[col], d["performance"])
        print(f"    {label:<34} beta = {m.params[col]:+7.3f}, "
              f"p = {m.pvalues[col]:.3f}   (raw r = {r[0]:+.3f})")
        rows_out = dict(predictor=label, beta=m.params[col], p=m.pvalues[col],
                        r=r[0])
        d.attrs.setdefault("models", []).append(rows_out)
    print("\n  If personalisation works, deviation from the subject's OWN optimum")
    print("  should predict performance better than deviation from the group's.")
    return d


def run():
    df = trial_table()
    a = part_a(df)
    a2 = part_a2(df)
    b = part_b(df)
    c = part_c(df)
    d = part_d(df)

    a = pd.concat([a.assign(kind="raw"), a2], ignore_index=True)
    a.to_csv(OUT / "optimum_reliability.csv", index=False)
    b.to_csv(OUT / "optimum_reliability_randomslope.csv", index=False)
    c.to_csv(OUT / "optimum_reliability_bandwidth.csv", index=False)
    d.to_csv(OUT / "optimum_reliability_transfer.csv", index=False)
    print(f"\nwrote four CSVs to {OUT}")


if __name__ == "__main__":
    run()
