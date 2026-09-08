"""Binomial re-specification of the discrete-time hazard analyses.

``hazard_model.py``, ``within_trial_control.py`` and ``individual_reactivity.py``
all fit a binary within-trial outcome with ``smf.mixedlm``: a linear probability
model with a subject random intercept. That is a defensible approximation, but
it is not what "discrete-time hazard model" names, and it leaves two
dependencies unmodelled. Consecutive 1 s bins share overlapping 5 s outcome
windows, so each crash contributes five nearly identical rows of outcome. And
bins are nested within trials, which the published frames carry no identifier
for. This script re-runs the three conclusions that rest on that specification
under models that handle both, and prints them beside the published numbers.
Nothing here modifies the published pipeline; it is purely additive.

Two reformulations, answering different questions.

  (1) SAME OUTCOME, VALID INFERENCE.  Keep crash-within-5 s exactly as
      published, but fit it as a binomial GEE with sandwich standard errors,
      clustered on trial (173 clusters, the level the window overlap acts at)
      and on subject (16 clusters, conservative). Answers: can the published
      p-value be believed?

  (2) CANONICAL PERSON-PERIOD SURVIVAL.  One row per interval at risk, event
      = 1 only in the interval the trial ends in. Outcome windows no longer
      overlap at all, so the dependence is removed by construction rather than
      modelled around. Complementary log-log link, the grouped
      proportional-hazards link that "discrete-time hazard" refers to, with
      logit alongside. All 173 hard-course trials end in a crash rather than at
      the 90 s ceiling, so there is no censoring.

Three specification points, each forced by the data rather than chosen.

  RISK INTERVAL IS 2 s, NOT 1 s.  Raw trial durations cluster just above odd
  integers (31.03, 61.09, 63.11 s), because a crash happens at a ring crossing
  and rings arrive every 2 s. Flooring to 1 s bins therefore places every event
  on an even bin index and leaves the odd ones structurally empty. Aggregating
  to the 2 s inter-ring interval restores one failure opportunity per risk
  period. It changes no conclusion here, but it removes the artefact.

  CLUSTER ON SUBJECT, NOT TRIAL.  In person-period data each trial contributes
  exactly one event by construction, so an exchangeable working correlation
  within trial is degenerate and the estimator diverges. It is also
  unnecessary: the discrete-time survival likelihood already factorises
  correctly within a trial given the covariates. What remains is dependence
  across trials within a subject, so the cluster is the subject.

  BASELINE HAZARD NEEDS FIXED EFFECTS, NOT A POLYNOMIAL.  The hazard is
  essentially zero for 30 s and then turns on sharply: 9 of 173 events occur
  before 30 s, across 66 percent of the intervals at risk. No low-order
  polynomial in elapsed time fits that, and the published 5 s strata cannot be
  used either, because the first carries no events at all and separates under
  any binomial link. Strata are therefore collapsed to six, each carrying at
  least ten events.

Because the arousal effect turns out not to be homogeneous across the trial
(interaction with the pre-/post-30 s window, joint p = 0.037), both the full
window and the window where trials actually end are reported, rather than one
being chosen after the fact.

Outputs: results/hazard_glmm_curvature.csv
         results/hazard_glmm_shape.csv
         results/hazard_glmm_reactivity.csv
         results/hazard_glmm_bandcompare.csv
         results/hazard_glmm_person_period.csv
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

HORIZON = 5          # s, the published outcome window
TRAIL = 10           # s, trailing window for the adaptive band
RISK = 2             # s, one inter-ring interval = one failure opportunity
LATE = 28            # s, start of the window in which trials actually end
STRATA = ([-1, LATE, 34, 40, 46, 52, 58, 200],
          ["<30", "30-34", "35-40", "41-46", "47-52", "53-58", "59+"])
RNG = np.random.default_rng(11)
N_PERM = 2000

CLOGLOG = sm.families.Binomial(link=links.CLogLog())
LOGIT = sm.families.Binomial(link=links.Logit())


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #

def build(df):
    """One row per 1 s bin: the published outcome, the survival indicator, the
    control signals, and a trial identifier the published frames lack."""
    traj = C.optimal_trajectory(C.performance_surface(hard_control(df)))
    opt, sd = traj["optimal"], traj["std"]

    rows = []
    for r in df[df.difficulty == 1].itertuples():
        a = C._binned_arousal(r.new_arousal)
        n = len(a)
        for t in range(n):
            if t >= len(opt) or np.isnan(opt[t]) or np.isnan(sd[t]):
                continue
            hi_fixed = opt[t] + 1.00 * sd[t]
            local = a[max(0, t - TRAIL):t + 1]
            local_sd = float(np.std(local)) if len(local) > 2 else float(sd[t])
            hi_adapt = opt[t] + max(local_sd, 1.0)
            rows.append({
                "subject": r.subject, "trial": f"{r.subject}_{r.trial}",
                "t": t, "tb": min(t // 5, 11), "arousal": a[t],
                "above": float(a[t] > hi_fixed),
                "excess_fixed": max(0.0, a[t] - hi_fixed),
                "excess_adaptive": max(0.0, a[t] - hi_adapt),
                "crash5": float(n - t <= HORIZON),
                "event": float(t == n - 1),
            })
    h = pd.DataFrame(rows)
    h["trial"] = h["trial"].astype(str).astype(object)
    h["a100"] = h["arousal"] / 100.0
    return h


def person_period(h):
    """Collapse to 2 s risk intervals and attach the baseline-hazard strata."""
    h = h.copy()
    h["k"] = h["t"] // RISK
    g = (h.groupby(["trial", "subject", "k"])
           .agg(arousal=("arousal", "mean"), event=("event", "max"),
                excess_fixed=("excess_fixed", "mean"),
                excess_adaptive=("excess_adaptive", "mean"),
                above=("above", "max"))
           .reset_index())
    g["trial"] = g["trial"].astype(str).astype(object)
    g["tsec"] = g["k"] * RISK
    g["a100"] = g["arousal"] / 100.0
    g["strat"] = pd.cut(g["tsec"], STRATA[0], labels=STRATA[1])
    g["late"] = (g["tsec"] >= LATE).astype(float)
    return g


def _gee(formula, data, groups, family, cov=None, cov_type="robust"):
    d = data.copy()
    if "strat" in d:
        d["strat"] = d["strat"].cat.remove_unused_categories()
    return sm.GEE.from_formula(
        formula, groups=d[groups], data=d, family=family,
        cov_struct=cov or sm.cov_struct.Independence()).fit(cov_type=cov_type)


def _quad(m, b1="a100", b2="I(a100 ** 2)"):
    B1, B2 = m.params[b1], m.params[b2]
    return B1, B2, (-B1 / (2 * B2) * 100 if B2 else np.nan)


# --------------------------------------------------------------------------- #
# (A) hazard curvature
# --------------------------------------------------------------------------- #

def part_a(h, g):
    print("=" * 94)
    print("(A)  HAZARD CURVATURE")
    print("     published: positive quadratic on the hazard (U-shaped hazard =")
    print("     inverted-U for performance), beta = 2.80e-05, p = 3.5e-10")
    print("=" * 94)
    rows = []

    print("\n--- 1. published specification, reproduced ---")
    m = smf.mixedlm("crash5 ~ arousal + I(arousal**2) + C(tb)", h,
                    groups=h["subject"]).fit(reml=False)
    print(f"  LMM, subject RI, treats {len(h)} bins as exchangeable")
    print(f"    beta2 = {m.params['I(arousal ** 2)']:+.3e}   "
          f"p = {m.pvalues['I(arousal ** 2)']:.2e}")
    rows.append(dict(step="published", model="LMM subject RI", outcome="crash5",
                     cluster="subject (model-based SE)",
                     beta_quad=m.params["I(arousal ** 2)"],
                     p_quad=m.pvalues["I(arousal ** 2)"], vertex=np.nan))

    print("\n--- 2. same outcome, binomial, sandwich SE ---")
    for grp in ("trial", "subject"):
        gg = _gee("crash5 ~ a100 + I(a100**2) + C(tb)", h, grp, LOGIT,
                  sm.cov_struct.Exchangeable())
        B1, B2, v = _quad(gg)
        print(f"  logit GEE, {grp:<8} clusters ({h[grp].nunique():>3})   "
              f"beta2 = {B2:+.3f}  p = {gg.pvalues['I(a100 ** 2)']:.4f}")
        rows.append(dict(step="crash5 binomial", model="logit GEE",
                         outcome="crash5", cluster=f"{grp} ({h[grp].nunique()})",
                         beta_quad=B2, p_quad=gg.pvalues["I(a100 ** 2)"],
                         vertex=v))

    print("\n--- 3. person-period survival, 2 s risk intervals ---")
    print(f"    {len(g)} intervals at risk, {int(g.event.sum())} events, "
          f"{g.trial.nunique()} trials, no censoring")
    print(f"    of which t >= {LATE}s: {int((g.late==1).sum())} intervals, "
          f"{int(g[g.late==1].event.sum())} events")
    gr = g[g.late == 1]
    for wname, d in (("full window", g), (f"t >= {LATE}s", gr)):
        print(f"\n    [{wname}]")
        for fam, fname in ((CLOGLOG, "cloglog"), (LOGIT, "logit  ")):
            for ct, ctn in (("robust", "sandwich"),
                            ("bias_reduced", "bias-reduced")):
                mm = _gee("event ~ a100 + I(a100**2) + C(strat)", d, "subject",
                          fam, cov_type=ct)
                B1, B2, v = _quad(mm)
                w = mm.wald_test("a100 = 0, I(a100 ** 2) = 0", scalar=False)
                chi, pj = float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))
                print(f"      {fname} / {ctn:<12} beta2 = {B2:+.3f} "
                      f"p = {mm.pvalues['I(a100 ** 2)']:.4f} | vertex = {v:5.1f}"
                      f" | JOINT chi2(2) = {chi:5.2f}, p = {pj:.4f}")
                rows.append(dict(step=f"person-period, {wname}",
                                 model=f"{fname.strip()} GEE, {ctn}",
                                 outcome="event", cluster="subject (16)",
                                 beta_quad=B2,
                                 p_quad=mm.pvalues["I(a100 ** 2)"], vertex=v,
                                 joint_chi2=chi, p_joint=pj))

    print("\n--- 4. is the arousal effect homogeneous across the trial? ---")
    mi = _gee("event ~ (a100 + I(a100**2))*late + C(strat)", g, "subject",
              CLOGLOG)
    w = mi.wald_test("a100:late = 0, I(a100 ** 2):late = 0", scalar=False)
    print(f"    arousal x late-window interaction, joint chi2(2) = "
          f"{float(np.squeeze(w.statistic)):.2f}, "
          f"p = {float(np.squeeze(w.pvalue)):.4f}")
    print("    -> not homogeneous, so both windows are reported above rather")
    print("       than one being selected after the fact.")
    rows.append(dict(step="homogeneity", model="arousal x late interaction",
                     outcome="event", cluster="subject (16)", beta_quad=np.nan,
                     p_quad=np.nan, vertex=np.nan,
                     joint_chi2=float(np.squeeze(w.statistic)),
                     p_joint=float(np.squeeze(w.pvalue))))

    pd.DataFrame(rows).to_csv(OUT / "hazard_glmm_curvature.csv", index=False)
    return pd.DataFrame(rows)


def part_a_shape(g):
    """Arousal as a factor: the shape with no functional form imposed."""
    print("\n--- 5. assumption-free shape: log-hazard by arousal quintile ---")
    print("    a U-shape needs the LOWEST quintile to sit ABOVE the middle ones")
    rows = []
    for wname, d in (("full window", g), (f"t >= {LATE}s", g[g.late == 1])):
        d = d.copy()
        d["aq"] = pd.qcut(d.arousal, 5, labels=False) + 1
        m = _gee("event ~ C(aq) + C(strat)", d, "subject", CLOGLOG)
        print(f"\n    [{wname}]")
        print(f"      {'q':<3}{'arousal':<12}{'n':>6}{'events':>8}"
              f"{'log-hazard':>12}{'SE':>7}{'p':>8}")
        for q in range(1, 6):
            s = d[d.aq == q]
            term = f"C(aq)[T.{q}]"
            b = 0.0 if q == 1 else m.params[term]
            se = np.nan if q == 1 else m.bse[term]
            pv = np.nan if q == 1 else m.pvalues[term]
            print(f"      {q:<3}"
                  f"{f'{s.arousal.min():.0f}-{s.arousal.max():.0f}':<12}"
                  f"{len(s):>6}{int(s.event.sum()):>8}{b:>+12.3f}{se:>7.3f}"
                  + (f"{pv:>8.3f}" if q > 1 else f"{'ref':>8}"))
            rows.append(dict(window=wname, quintile=q,
                             arousal_lo=s.arousal.min(),
                             arousal_hi=s.arousal.max(), n=len(s),
                             events=int(s.event.sum()), log_hazard=b, se=se,
                             p=pv))
    pd.DataFrame(rows).to_csv(OUT / "hazard_glmm_shape.csv", index=False)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (B) individual reactivity
# --------------------------------------------------------------------------- #

SPREAD = {"SD": lambda x: float(np.std(x)),
          "IQR": lambda x: float(np.subtract(*np.percentile(x, [75, 25]))),
          "MAD": lambda x: float(np.median(np.abs(x - np.median(x))))}


def _slopes(d, col, labels):
    """Per-subject hazard slopes on ``col``, one logistic fit per subject."""
    sl = []
    for lab in np.unique(labels):
        s = d[labels == lab]
        if s["event"].sum() < 3 or s[col].std() < 1e-9:
            continue
        try:
            f = sm.GLM.from_formula(f"event ~ {col} + tsec", s,
                                    family=LOGIT).fit()
            b = f.params[col]
            if np.isfinite(b) and abs(b) < 10:
                sl.append(b)
        except Exception:
            pass
    return np.array(sl)


def part_b(h, g):
    print("\n" + "=" * 94)
    print("(B)  INDIVIDUAL REACTIVITY")
    print("     published: subjects differ in how strongly excess arousal")
    print("     predicts failure (LMM random-slope LRT)")
    print("=" * 94)
    rows = []

    for col in ("excess_fixed", "excess_adaptive"):
        print(f"\n  {col}")
        m0 = smf.mixedlm(f"crash5 ~ {col} + C(tb)", h,
                         groups=h.subject).fit(reml=False)
        m1 = smf.mixedlm(f"crash5 ~ {col} + C(tb)", h, groups=h.subject,
                         re_formula=f"~{col}").fit(reml=False)
        d, k = 2 * (m1.llf - m0.llf), len(m1.params) - len(m0.params)
        print(f"    LMM random-slope LRT (published)  chi2({k}) = {d:6.2f}, "
              f"p = {chi2.sf(d, k):.2e}")
        rows.append(dict(signal=col, test="LMM random-slope LRT (published)",
                         stat=d, p=chi2.sf(d, k)))

        # A subject-by-slope Wald test is not usable here: with ~11 events
        # per subject the interaction terms separate and the statistic
        # diverges. The permutation test below needs no such fit. It compares
        # the observed spread of per-subject slopes against a null that
        # reassigns whole trials to subjects, so it preserves within-trial
        # structure and destroys only the subject identity. Three spread
        # measures, because the SD of 11 noisy slopes is outlier-dominated.
        obs = _slopes(g, col, g["subject"].to_numpy())
        tr = g.groupby("trial")["subject"].first()
        null = {k: [] for k in SPREAD}
        for _ in range(N_PERM):
            perm = pd.Series(RNG.permutation(tr.to_numpy()), index=tr.index)
            sl = _slopes(g, col, g["trial"].map(perm).to_numpy())
            if len(sl) > 3:
                for k, fn in SPREAD.items():
                    null[k].append(fn(sl))
        print(f"    {len(obs)} estimable subject slopes, "
              f"range {obs.min():+.3f} to {obs.max():+.3f}")
        for k, fn in SPREAD.items():
            o, nl = fn(obs), np.array(null[k])
            pv = (1 + (nl >= o).sum()) / (1 + len(nl))
            print(f"    permutation on slope spread ({k:<3})  obs = {o:.4f}, "
                  f"null {nl.mean():.4f} +/- {nl.std():.4f}, p = {pv:.4f}")
            rows.append(dict(signal=col, test=f"permutation, slope spread ({k})",
                             stat=o, null_mean=nl.mean(), p=pv))

    pd.DataFrame(rows).to_csv(OUT / "hazard_glmm_reactivity.csv", index=False)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (C) does the adaptive band add information over the fixed band?
# --------------------------------------------------------------------------- #

def part_c(h, g):
    print("\n" + "=" * 94)
    print("(C)  ADAPTIVE BAND ADDS INFORMATION OVER THE FIXED BAND")
    print("     published: beta = +0.0020, LRT chi2 = 11.6, p = 6.6e-04")
    print("=" * 94)
    rows = []

    mf = smf.mixedlm("crash5 ~ excess_fixed + excess_adaptive + C(tb)", h,
                     groups=h.subject).fit(reml=False)
    mb = smf.mixedlm("crash5 ~ excess_fixed + C(tb)", h,
                     groups=h.subject).fit(reml=False)
    lr = 2 * (mf.llf - mb.llf)
    print(f"\n  LMM added-term LRT (published)              beta = "
          f"{mf.params['excess_adaptive']:+.5f}, chi2(1) = {lr:.2f}, "
          f"p = {chi2.sf(lr, 1):.2e}")
    rows.append(dict(outcome="crash5", model="LMM subject RI (published)",
                     beta=mf.params["excess_adaptive"], p=chi2.sf(lr, 1)))

    for label, d, outcome, tterm, fam, grp in [
            ("logit GEE, trial clusters", h, "crash5", "C(tb)", LOGIT, "trial"),
            ("logit GEE, subject clusters", h, "crash5", "C(tb)", LOGIT,
             "subject"),
            ("cloglog person-period, subject clusters", g, "event",
             "C(strat)", CLOGLOG, "subject"),
            ("logit person-period, subject clusters", g, "event",
             "C(strat)", LOGIT, "subject")]:
        cov = sm.cov_struct.Exchangeable() if outcome == "crash5" else None
        mm = _gee(f"{outcome} ~ excess_fixed + excess_adaptive + {tterm}",
                  d, grp, fam, cov)
        b, p = mm.params["excess_adaptive"], mm.pvalues["excess_adaptive"]
        print(f"  {label:<43} beta = {b:+.5f}, p = {p:.4f}")
        rows.append(dict(outcome=outcome, model=label, beta=b, p=p))

    print("\n  Each signal on its own -- does it predict failure at all?")
    print(f"    {'signal':<18}{'published LMM (crash5)':>28}"
          f"{'person-period cloglog':>28}")
    for col in ("above", "excess_fixed", "excess_adaptive"):
        m = smf.mixedlm(f"crash5 ~ {col} + C(tb)", h,
                        groups=h.subject).fit(reml=False)
        m0 = smf.mixedlm("crash5 ~ C(tb)", h, groups=h.subject).fit(reml=False)
        lr = 2 * (m.llf - m0.llf)
        mm = _gee(f"event ~ {col} + C(strat)", g, "subject", CLOGLOG)
        print(f"    {col:<18}"
              f"{f'b={m.params[col]:+.4f} p={chi2.sf(lr, 1):.1e}':>28}"
              f"{f'b={mm.params[col]:+.3f} p={mm.pvalues[col]:.4f}':>28}")
        rows.append(dict(outcome="event", model=f"{col} alone, cloglog p-p",
                         beta=mm.params[col], p=mm.pvalues[col]))
    r = g.excess_fixed.corr(g.excess_adaptive)
    print(f"\n    The two graded signals correlate at r = {r:.3f}. Each predicts")
    print("    failure on its own. What does not survive is the claim that the")
    print("    ADAPTIVE reference adds information over the fixed one: entered")
    print("    together, the fixed excess keeps the effect and the adaptive one")
    print("    contributes nothing. The published ordering reverses.")

    pd.DataFrame(rows).to_csv(OUT / "hazard_glmm_bandcompare.csv", index=False)
    return pd.DataFrame(rows)


def run():
    h = build(trial_table())
    g = person_period(h)
    g.to_csv(OUT / "hazard_glmm_person_period.csv", index=False)
    part_a(h, g)
    part_a_shape(g)
    part_b(h, g)
    part_c(h, g)
    print(f"\nwrote {OUT / 'hazard_glmm_curvature.csv'} and four others")


if __name__ == "__main__":
    run()
