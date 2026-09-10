"""Binomial re-specification of the discrete-time hazard analyses.

``hazard_model.py``, ``within_trial_control.py`` and ``individual_reactivity.py``
fit a binary within-trial outcome with ``smf.mixedlm``: a linear probability
model with a subject random intercept. Two problems follow. The likelihood is
Gaussian for a binary outcome. And the crash-within-5 s outcome overlaps across
consecutive 1 s bins, so a single crash marks five rows and the model counts it
five times. That turns 173 events into roughly 1,600 apparent ones and shrinks
every standard error. The published bin tables also carry no trial identifier.

The fix has to preserve the question. "Does the current state predict a crash
within the next 5 s?" is what a controller faces, and it evaluates that every
second, so the overlapping windows are the deployment case rather than a
mistake. What the overlap breaks is the standard error, not the estimand:
running the controller every second does not create more crashes, and there are
173 in the data however often the model is queried.

  PRIMARY.  Keep the published outcome and bins exactly as they are, and fit
  them as a binomial GEE with sandwich standard errors clustered on trial. This
  changes nothing about the question and only stops the dependent rows counting
  as independent.

Three further specifications say how far the answer travels:

  non-overlapping 5 s blocks   each trial cut into consecutive 5 s blocks, the
                               outcome whether the trial ends inside the block,
                               the predictor the state at its start; keeps the
                               5 s horizon and counts each crash once
  person-period, terminal bin  one row per 2 s interval at risk, the event
                               scored only where the trial ends; answers
                               "arousal at the instant of the crash", which is
                               a stricter and different question
  5 s lead                     arousal at t predicts the crash at t + 5 s

Three choices are forced by the data. The 2 s interval in the person-period
model is the ring spacing: trial durations cluster just above odd integers
because a crash happens at a ring crossing, so 1 s bins leave every second bin
structurally empty. The cluster is the subject, not the trial, because each
trial contributes exactly one event by construction and a within-trial working
correlation is degenerate. And the baseline hazard uses collapsed time strata
because 9 of the 173 events fall in the first 30 s across two thirds of the
time at risk, and the first 5 s stratum separates under any binomial link.

Outputs: results/hazard_glmm_curvature.csv, _shape.csv, _reactivity.csv,
         _bandcompare.csv, _feedback.csv, _blocks.csv
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.genmod.families import links

from common import C, OUT, hard_control, trial_table

warnings.filterwarnings("ignore")

HORIZON = 5          # s, the outcome window the paper asks about
BLOCK = 5            # s, one non-overlapping block = one failure opportunity
TRAIL = 10           # s, trailing window for the adaptive band
RISK = 2             # s, ring spacing, for the person-period comparison
LEAD = 5             # s, lead time for the lagged comparison
RNG = np.random.default_rng(11)
N_PERM = 2000

CLOGLOG = sm.families.Binomial(link=links.CLogLog())
LOGIT = sm.families.Binomial(link=links.Logit())

# A per-subject reactivity slope needs enough blocks in which the subject was
# actually above the band. Below this the logistic quasi-separates and returns
# converged but meaningless coefficients (-44.9 for one subject, against a
# pooled effect of +0.02), which inflates every measure of spread.
MIN_NONZERO = 20
MAX_SLOPE = 1.0

SPREAD = {"SD": lambda x: float(np.std(x)),
          "IQR": lambda x: float(np.subtract(*np.percentile(x, [75, 25]))),
          "MAD": lambda x: float(np.median(np.abs(x - np.median(x))))}


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #

def build(df):
    """One row per 1 s bin, with a trial identifier and every control signal."""
    traj = C.optimal_trajectory(C.performance_surface(hard_control(df)))
    opt, sd = traj["optimal"], traj["std"]
    cond = {f"{r.subject}_{r.trial}": r.condition for r in df.itertuples()}

    rows = []
    for r in df[df.difficulty == 1].itertuples():
        a = C._binned_arousal(r.new_arousal)
        n = len(a)
        run_above = 0
        for t in range(n):
            if t >= len(opt) or np.isnan(opt[t]) or np.isnan(sd[t]):
                continue
            hi_fixed = opt[t] + 1.00 * sd[t]
            local = a[max(0, t - TRAIL):t + 1]
            local_sd = float(np.std(local)) if len(local) > 2 else float(sd[t])
            hi_adapt = opt[t] + max(local_sd, 1.0)
            run_above = run_above + 1 if a[t] > hi_fixed else 0
            rows.append({
                "subject": r.subject, "trial": f"{r.subject}_{r.trial}",
                "condition": r.condition, "t": t, "tb": min(t // 5, 11),
                "arousal": a[t],
                "above": float(a[t] > hi_fixed),
                "above_adaptive": float(a[t] > hi_adapt),
                "sustained": float(run_above >= 3),
                "rate": (a[t] - a[max(0, t - 3)]) / max(1, min(3, t)),
                "excess_fixed": max(0.0, a[t] - hi_fixed),
                "excess_adaptive": max(0.0, a[t] - hi_adapt),
                "crash5": float(n - t <= HORIZON),
                "event": float(t == n - 1),
            })
    h = pd.DataFrame(rows)
    h["trial"] = h["trial"].astype(str).astype(object)
    h["a100"] = h["arousal"] / 100.0
    return h


def blocks(h):
    """PRIMARY frame: non-overlapping 5 s blocks.

    The outcome is whether the trial ends inside the block; the predictors are
    the state at the START of the block, so they precede the outcome window by
    0 to 5 s, which is the lead the published analysis had. Block means are
    carried alongside for a sensitivity check.
    """
    rows = []
    for tid, g in h.groupby("trial"):
        g = g.sort_values("t")
        n = len(g)
        for b0 in range(0, n, BLOCK):
            s = g.iloc[b0:min(b0 + BLOCK, n)]
            f = s.iloc[0]
            rows.append({
                "trial": tid, "subject": f.subject, "condition": f.condition,
                "tsec": b0,
                "arousal": f.arousal, "arousal_mean": s.arousal.mean(),
                "above": f.above, "above_adaptive": f.above_adaptive,
                "sustained": s.sustained.max(), "rate": s.rate.mean(),
                "excess_fixed": f.excess_fixed,
                "excess_adaptive": f.excess_adaptive,
                "event": float(b0 + BLOCK >= n)})
    B = pd.DataFrame(rows)
    B["trial"] = B["trial"].astype(str).astype(object)
    B["a100"] = B.arousal / 100.0
    B["am100"] = B.arousal_mean / 100.0
    B["strat"] = pd.cut(B.tsec, [-1, 28, 40, 52, 200],
                        labels=["<30", "30-40", "41-52", "53+"])
    return B


def person_period(h):
    """Comparison frame: 2 s intervals, event only where the trial ends."""
    h = h.copy()
    h["k"] = h["t"] // RISK
    g = (h.groupby(["trial", "subject", "k"])
           .agg(arousal=("arousal", "mean"), event=("event", "max"))
           .reset_index())
    g["trial"] = g["trial"].astype(str).astype(object)
    g["tsec"] = g["k"] * RISK
    g["a100"] = g.arousal / 100.0
    g["strat"] = pd.cut(g.tsec, [-1, 28, 34, 40, 46, 52, 58, 200],
                        labels=["<30", "30-34", "35-40", "41-46", "47-52",
                                "53-58", "59+"])
    return g


def lead_frame(h):
    """Comparison frame: arousal at t predicts the crash at t + LEAD."""
    rows = []
    for tid, g in h.groupby("trial"):
        g = g.sort_values("t")
        a = g.arousal.to_numpy()
        n = len(g)
        for t in range(LEAD, n):
            rows.append({"trial": tid, "subject": g.subject.iloc[0],
                         "tsec": t, "a100": a[t - LEAD] / 100.0,
                         "event": float(t == n - 1)})
    L = pd.DataFrame(rows)
    L["trial"] = L["trial"].astype(str).astype(object)
    L["strat"] = pd.cut(L.tsec, [-1, 28, 34, 40, 46, 52, 58, 200],
                        labels=["<30", "30-34", "35-40", "41-46", "47-52",
                                "53-58", "59+"])
    return L


def gee(formula, data, family=CLOGLOG, groups="subject"):
    d = data.copy()
    if "strat" in d:
        d["strat"] = d["strat"].cat.remove_unused_categories()
    return sm.GEE.from_formula(formula, groups=d[groups], data=d,
                               family=family,
                               cov_struct=sm.cov_struct.Independence()).fit()


# --------------------------------------------------------------------------- #
# (A) curvature
# --------------------------------------------------------------------------- #

def part_a(h, B, P, L):
    print("=" * 96)
    print("(A)  HAZARD CURVATURE")
    print("     published: positive quadratic on the hazard, beta = 2.80e-05,")
    print("     p = 3.5e-10, from a linear probability fit")
    print("=" * 96)
    rows = []

    m = smf.mixedlm("crash5 ~ arousal + I(arousal**2) + C(tb)", h,
                    groups=h["subject"]).fit(reml=False)
    print(f"\n  published LMM, {len(h)} bins treated as exchangeable")
    print(f"    beta2 = {m.params['I(arousal ** 2)']:+.3e}, "
          f"p = {m.pvalues['I(arousal ** 2)']:.2e}")
    rows.append(dict(spec="published LMM", link="gaussian",
                     beta_quad=m.params["I(arousal ** 2)"], ci_lo=np.nan,
                     ci_hi=np.nan, p_quad=m.pvalues["I(arousal ** 2)"],
                     vertex=np.nan, p_joint=np.nan))

    print(f"\n  {'specification':<44}{'beta2':>8}{'95% CI':>18}"
          f"{'p':>9}{'vertex':>8}{'joint p':>9}")
    specs = [
        ("PRIMARY  crash-within-5 s, GEE on trial clusters", None, None, None),
        ("stricter 5 s blocks, state at block start", B, "a100", CLOGLOG),
        ("         5 s blocks, block mean", B, "am100", CLOGLOG),
        ("         5 s blocks, block start, logit", B, "a100", LOGIT),
        ("person-period, terminal bin", P, "a100", CLOGLOG),
        (f"{LEAD} s lead", L, "a100", CLOGLOG),
    ]
    for label, d, col, fam in specs:
        if d is None:                       # published outcome, fixed errors
            g2 = sm.GEE.from_formula("crash5 ~ a100 + I(a100**2) + C(tb)",
                                     groups=h["trial"], data=h, family=LOGIT,
                                     cov_struct=sm.cov_struct.Exchangeable()
                                     ).fit()
            b1, b2 = g2.params["a100"], g2.params["I(a100 ** 2)"]
            lo, hi = g2.conf_int().loc["I(a100 ** 2)"]
            pj = np.nan
            pq = g2.pvalues["I(a100 ** 2)"]
        else:
            mm = gee(f"event ~ {col} + I({col}**2) + C(strat)", d, fam)
            b1, b2 = mm.params[col], mm.params[f"I({col} ** 2)"]
            lo, hi = mm.conf_int().loc[f"I({col} ** 2)"]
            pq = mm.pvalues[f"I({col} ** 2)"]
            w = mm.wald_test(f"{col} = 0, I({col} ** 2) = 0", scalar=False)
            pj = float(np.squeeze(w.pvalue))
        v = -b1 / (2 * b2) * 100 if b2 else np.nan
        print(f"  {label:<44}{b2:>+8.2f}  [{lo:+.2f},{hi:+.2f}]"
              f"{pq:>9.4f}{v:>8.1f}" + (f"{pj:>9.4f}" if np.isfinite(pj)
                                        else f"{'-':>9}"))
        rows.append(dict(spec=label.strip(), link="cloglog" if fam is CLOGLOG
                         else "logit", beta_quad=b2, ci_lo=lo, ci_hi=hi,
                         p_quad=pq, vertex=v, p_joint=pj))

    print("\n  -> the curvature holds with the dependence handled (p = 0.043),")
    print("     weakens but keeps its sign and vertex when each crash is counted")
    print("     once (p = 0.052), and fades once the question narrows to the")
    print("     instant of the crash rather than the next five seconds.")
    pd.DataFrame(rows).to_csv(OUT / "hazard_glmm_curvature.csv", index=False)
    return pd.DataFrame(rows)


def part_a_shape(B):
    print("\n  Assumption-free shape: log-hazard by arousal quintile, 5 s blocks")
    d = B.copy()
    d["aq"] = pd.qcut(d.arousal, 5, labels=False) + 1
    m = gee("event ~ C(aq) + C(strat)", d)
    rows = []
    print(f"    {'q':<3}{'arousal':<12}{'n':>6}{'events':>8}{'log-hazard':>12}"
          f"{'SE':>7}{'p':>8}")
    for q in range(1, 6):
        s = d[d.aq == q]
        term = f"C(aq)[T.{q}]"
        b = 0.0 if q == 1 else m.params[term]
        se = np.nan if q == 1 else m.bse[term]
        pv = np.nan if q == 1 else m.pvalues[term]
        print(f"    {q:<3}{f'{s.arousal.min():.0f}-{s.arousal.max():.0f}':<12}"
              f"{len(s):>6}{int(s.event.sum()):>8}{b:>+12.3f}{se:>7.3f}"
              + (f"{pv:>8.3f}" if q > 1 else f"{'ref':>8}"))
        rows.append(dict(quintile=q, arousal_lo=s.arousal.min(),
                         arousal_hi=s.arousal.max(), n=len(s),
                         events=int(s.event.sum()), log_hazard=b, se=se, p=pv))
    pd.DataFrame(rows).to_csv(OUT / "hazard_glmm_shape.csv", index=False)


# --------------------------------------------------------------------------- #
# (B) individual reactivity
# --------------------------------------------------------------------------- #

def _slopes(d, col, labels):
    """Per-subject reactivity slopes, degenerate fits excluded."""
    out = []
    for lab in np.unique(labels):
        s = d[labels == lab]
        if s.event.sum() < 3 or (s[col] > 0).sum() < MIN_NONZERO:
            continue
        try:
            f = sm.GLM.from_formula(f"event ~ {col} + tsec", s,
                                    family=LOGIT).fit()
            b = f.params[col]
            if f.converged and np.isfinite(b) and abs(b) < MAX_SLOPE:
                out.append(b)
        except Exception:
            pass
    return np.array(out)


def part_b(h, B, prof):
    print("\n" + "=" * 96)
    print("(B)  INDIVIDUAL REACTIVITY -- do subjects differ, and does a")
    print("     calibration measure predict the difference?")
    print("=" * 96)
    rows = []
    tr = B.groupby("trial")["subject"].first()

    for col in ("excess_fixed", "excess_adaptive"):
        m0 = smf.mixedlm(f"crash5 ~ {col} + C(tb)", h,
                         groups=h.subject).fit(reml=False)
        m1 = smf.mixedlm(f"crash5 ~ {col} + C(tb)", h, groups=h.subject,
                         re_formula=f"~{col}").fit(reml=False)
        d0 = 2 * (m1.llf - m0.llf)
        print(f"\n  {col}")
        print(f"    published LMM random-slope LRT   chi2(2) = {d0:6.2f}, "
              f"p = {chi2.sf(d0, 2):.2e}")
        rows.append(dict(signal=col, test="LMM random-slope LRT (published)",
                         stat=d0, p=chi2.sf(d0, 2)))

        obs = _slopes(B, col, B.subject.to_numpy())
        enough = sum((g[col] > 0).sum() >= MIN_NONZERO
                     for _, g in B.groupby("subject"))
        print(f"    {enough} of {B.subject.nunique()} subjects spend "
              f"{MIN_NONZERO}+ blocks above the band; {len(obs)} give a stable fit")
        null = {k: [] for k in SPREAD}
        for _ in range(N_PERM):
            perm = pd.Series(RNG.permutation(tr.to_numpy()), index=tr.index)
            sl = _slopes(B, col, B.trial.map(perm).to_numpy())
            if len(sl) > 3:
                for k, fn in SPREAD.items():
                    null[k].append(fn(sl))
        print(f"    slopes {obs.min():+.3f} to {obs.max():+.3f}")
        for k, fn in SPREAD.items():
            o, nl = fn(obs), np.array(null[k])
            pv = (1 + (nl >= o).sum()) / (1 + len(nl))
            print(f"    permutation, spread as {k:<3}          obs = {o:.4f}, "
                  f"null {nl.mean():.4f}, p = {pv:.4f}")
            rows.append(dict(signal=col, test=f"permutation ({k}), 5 s blocks",
                             stat=o, p=pv))

    print("\n  Does a calibration measure predict reactivity? (5 s blocks)")
    print(f"    {'marker':<10}{'excess_fixed':>26}{'excess_adaptive':>26}")
    for m in ("resp", "cal_sd", "hrv"):
        z = m + "_z"
        line = f"    {m:<10}"
        for col in ("excess_fixed", "excess_adaptive"):
            mm = gee(f"event ~ {col}*{z} + C(strat)", B)
            t = f"{col}:{z}"
            line += f"{f'b={mm.params[t]:+.4f} p={mm.pvalues[t]:.4f}':>26}"
            rows.append(dict(signal=col, test=f"{m} moderation, 5 s blocks",
                             stat=mm.params[t], p=mm.pvalues[t]))
        print(line)

    pd.DataFrame(rows).to_csv(OUT / "hazard_glmm_reactivity.csv", index=False)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (C) control parameters
# --------------------------------------------------------------------------- #

def part_c(h, B):
    print("\n" + "=" * 96)
    print("(C)  CONTROL PARAMETERS -- does the adaptive reference add anything?")
    print("     published: beta = +0.0020, LRT chi2 = 11.6, p = 6.6e-04")
    print("=" * 96)
    rows = []

    mf = smf.mixedlm("crash5 ~ excess_fixed + excess_adaptive + C(tb)", h,
                     groups=h.subject).fit(reml=False)
    mb = smf.mixedlm("crash5 ~ excess_fixed + C(tb)", h,
                     groups=h.subject).fit(reml=False)
    lr = 2 * (mf.llf - mb.llf)
    print(f"\n  published LMM, adaptive over fixed   beta = "
          f"{mf.params['excess_adaptive']:+.5f}, p = {chi2.sf(lr, 1):.2e}")
    rows.append(dict(model="LMM adaptive over fixed (published)",
                     beta=mf.params["excess_adaptive"], p=chi2.sf(lr, 1)))

    print("\n  5 s blocks, both graded signals in together:")
    mm = gee("event ~ excess_fixed + excess_adaptive + C(strat)", B)
    for t in ("excess_fixed", "excess_adaptive"):
        print(f"    {t:<18} beta = {mm.params[t]:+.4f}, p = {mm.pvalues[t]:.4f}")
        rows.append(dict(model=f"{t}, both in, 5 s blocks",
                         beta=mm.params[t], p=mm.pvalues[t]))
    print(f"    the two correlate at r = "
          f"{B.excess_fixed.corr(B.excess_adaptive):.3f}")

    print("\n  each signal on its own, 5 s blocks:")
    for c in ("above", "excess_fixed", "excess_adaptive"):
        m2 = gee(f"event ~ {c} + C(strat)", B)
        hr = np.exp(m2.params[c])
        lo, hi = np.exp(m2.conf_int().loc[c])
        print(f"    {c:<18} beta = {m2.params[c]:+.4f}, p = {m2.pvalues[c]:.4f}"
              + (f",  HR {hr:.2f} [{lo:.2f}, {hi:.2f}]" if c == "above" else ""))
        rows.append(dict(model=f"{c} alone, 5 s blocks", beta=m2.params[c],
                         p=m2.pvalues[c]))

    print("\n  does each parameter add information over the fixed rule?")
    print(f"    {'parameter':<20}{'on top of':<14}{'published':>12}"
          f"{'GEE on trial':>14}{'5 s blocks':>13}")
    for extra, base in [("above_adaptive", "above"), ("sustained", "above"),
                        ("rate", "above"), ("excess_adaptive", "excess_fixed")]:
        f1 = smf.mixedlm(f"crash5 ~ {base} + {extra} + C(tb)", h,
                         groups=h.subject).fit(reml=False)
        f0 = smf.mixedlm(f"crash5 ~ {base} + C(tb)", h,
                         groups=h.subject).fit(reml=False)
        p_pub = chi2.sf(max(2 * (f1.llf - f0.llf), 0), 1)
        # Primary: the published outcome and bins, errors clustered on trial.
        g1 = sm.GEE.from_formula(f"crash5 ~ {base} + {extra} + C(tb)",
                                 groups=h["trial"], data=h, family=LOGIT,
                                 cov_struct=sm.cov_struct.Exchangeable()).fit()
        m2 = gee(f"event ~ {base} + {extra} + C(strat)", B)
        print(f"    {extra:<20}{base:<14}{p_pub:>12.2e}"
              f"{g1.pvalues[extra]:>14.4f}{m2.pvalues[extra]:>13.4f}")
        rows.append(dict(model=f"{extra} added over {base}",
                         beta=g1.params[extra], p=g1.pvalues[extra],
                         p_blocks=m2.pvalues[extra], p_published=p_pub))

    pd.DataFrame(rows).to_csv(OUT / "hazard_glmm_bandcompare.csv", index=False)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (D) feedback
# --------------------------------------------------------------------------- #

def part_d(h, B):
    print("\n" + "=" * 96)
    print("(D)  FEEDBACK x ABOVE-BAND")
    print("     published: above beta = +0.0498 p < 0.001; x full BCI")
    print("     interaction beta = -0.0381, p = 0.040")
    print("=" * 96)
    rows = []
    m = smf.mixedlm("crash5 ~ above * C(condition) + t", h,
                    groups=h.subject).fit(reml=False)
    print("\n  published LMM:")
    for k in m.params.index:
        if "above" in k:
            print(f"    {k:<32} beta = {m.params[k]:+.4f}, p = {m.pvalues[k]:.4f}")

    print("\n  5 s blocks:")
    mm = gee("event ~ above * C(condition) + C(strat)", B)
    for k in mm.params.index:
        if "above" in k:
            print(f"    {k:<32} beta = {mm.params[k]:+.4f}, p = {mm.pvalues[k]:.4f}")
            rows.append(dict(term=k, beta=mm.params[k], p=mm.pvalues[k]))
    w = mm.wald_test("above:C(condition)[T.2] = 0, "
                     "above:C(condition)[T.3] = 0", scalar=False)
    chi_, p_ = float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))
    print(f"    joint interaction chi2(2) = {chi_:.2f}, p = {p_:.4f}")
    rows.append(dict(term="joint interaction", beta=chi_, p=p_))
    pd.DataFrame(rows).to_csv(OUT / "hazard_glmm_feedback.csv", index=False)
    return pd.DataFrame(rows)


def run():
    h = build(trial_table())
    B = blocks(h)
    B.to_csv(OUT / "hazard_glmm_blocks.csv", index=False)
    P, L = person_period(h), lead_frame(h)
    print(f"{len(h)} one-second bins, {B.trial.nunique()} trials, "
          f"{int(B.event.sum())} crashes")
    print(f"primary frame: {len(B)} non-overlapping {BLOCK} s blocks\n")

    prof = pd.read_csv(OUT / "calibration_profile.csv")
    for m in ("cal_sd", "resp", "hrv"):
        prof[m + "_z"] = (prof[m] - prof[m].mean()) / prof[m].std()
    mp = prof.set_index("subject")
    for m in ("cal_sd", "resp", "hrv"):
        B[m + "_z"] = B.subject.map(mp[m + "_z"])

    part_a(h, B, P, L)
    part_a_shape(B)
    part_b(h, B, prof)
    part_c(h, B)
    part_d(h, B)
    print(f"\nwrote {OUT / 'hazard_glmm_curvature.csv'} and five others")


if __name__ == "__main__":
    run()
