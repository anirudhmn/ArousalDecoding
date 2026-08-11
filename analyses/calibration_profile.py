"""Personalisation from the calibration block only.

The other personalisation analyses draw their parameters from the same
closed-loop trials they then evaluate on. Even the leave-one-subject-out version
in ``loso_control_bands.py`` refits the sensitivity median and the multipliers on
other subjects' closed-loop data, so the scheme has never been shown to work
from information available before the loop is closed.

This derives every subject-specific quantity from the open-loop calibration
block, condition 0, and evaluates on closed-loop trials. That is what a deployed
system would do: calibrate, then run. It also keeps all 16 subjects in the
comparison.

Two families of calibration parameter:

  physiology   baseline HRV, respiration, EEG band power and so on, as screened
               in ``baseline_marker_screen.py`` and ``vagal_composite.py``.
  decoded      statistics of the decoder's own out-of-fold output during
               calibration: level, variability, volatility. These come from the
               cross-validated probabilities saved by ``label_controls.py``, so
               they are out of sample within calibration as well as independent
               of the closed-loop data.

Sections:

  A  Build the calibration profile.
  B  Are these measures reliable within the calibration block? A parameter that
     is not reliable cannot personalise anything. This is what sank the
     closed-loop band width in ``optimum_reliability.py``.
  C  Do they transfer? Does calibration variability predict closed-loop
     variability?
  D  Out-of-sample personalised band. Width is set from calibration, the
     trajectory is fitted leave-one-subject-out, and the result is evaluated on
     held-out closed-loop trials. Compared against the universal band and
     against the median split.
  E  Holm-corrected screen of every calibration measure against the closed-loop
     outcomes that matter.

Outputs: results/calibration_profile.csv,
         results/calibration_profile_reliability.csv,
         results/calibration_profile_bands.csv,
         results/calibration_profile_screen.csv
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

from common import C, D, OUT, cfg, hard_control, trial_table
from arousal.plotting import holm
from baseline_marker_screen import baseline_panel

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(25)
PROBS = OUT / "label_controls_probs.csv"
N_BOOT = 5000


# --------------------------------------------------------------------------- #
# A - the calibration profile
# --------------------------------------------------------------------------- #

def subject_bounds(probs):
    """The decoder's 5th/95th-percentile rescaling bounds, per subject.

    Computed once over that subject's whole calibration block. Computing them
    inside a subset would renormalise every subset to the same span and make
    spread measures constant by construction.
    """
    return {s: (np.percentile(g["prob"], 5), np.percentile(g["prob"], 95))
            for s, g in probs.groupby("subject")}


def scale_0_100(p, lo, hi):
    return np.clip((p - lo) / (hi - lo + 1e-12), 0, 1) * 100


def decoded_profile(probs, bounds, subset=None):
    """Per-subject statistics of out-of-fold calibration arousal."""
    rows = []
    for s, g in probs.groupby("subject"):
        g = g.sort_values(["trial", "position"])
        if subset is not None:
            g = g[subset(g)]
        if len(g) < 20:
            continue
        lo, hi = bounds[s]
        a = scale_0_100(g["prob"].to_numpy(), lo, hi)
        rows.append({
            "subj_id": s,
            "cal_level": a.mean(),
            "cal_sd": a.std(ddof=1),
            "cal_iqr": np.subtract(*np.percentile(a, [75, 25])),
            # mean absolute step between consecutive epochs: how twitchy is
            # this person's arousal, independent of how high it runs
            "cal_volatility": np.abs(np.diff(a)).mean(),
            # fraction of epochs at the top of the subject's own range
            "cal_frac_high": float((a > 66).mean()),
        })
    return pd.DataFrame(rows)


def build_profile():
    probs = pd.read_csv(PROBS)
    dec = decoded_profile(probs, subject_bounds(probs))

    phys = baseline_panel()                      # calibration-block physiology
    phys = phys.rename(columns={"subject": "s_index"})
    subj_ids = sorted(probs.subject.unique())
    phys["subj_id"] = [subj_ids[i] for i in phys["s_index"]]

    prof = dec.merge(phys.drop(columns=["s_index"]), on="subj_id")
    # map to the trial table's 0-15 subject index
    prof["subject"] = [subj_ids.index(s) for s in prof["subj_id"]]
    return prof.sort_values("subject").reset_index(drop=True)


DECODED = ["cal_level", "cal_sd", "cal_iqr", "cal_volatility", "cal_frac_high"]
PHYSIO = ["hrv", "resp", "hr", "eda_p", "eda_t", "pup",
          "theta", "alpha", "beta", "gamma", "theta_beta", "hrv_sd"]


def part_a(prof):
    print("=" * 90)
    print("(A) calibration profile, built without touching closed-loop data")
    print("=" * 90)
    print(f"\n{len(prof)} subjects\n")
    print(prof[["subject"] + DECODED].round(2).to_string(index=False))
    return prof


# --------------------------------------------------------------------------- #
# B - reliability within the calibration block
# --------------------------------------------------------------------------- #

def part_b(n_rep=200):
    """Split calibration TRIALS in half and re-derive each decoded measure."""
    probs = pd.read_csv(PROBS)
    bounds = subject_bounds(probs)
    print("\n" + "=" * 90)
    print("(B) are the calibration measures reliable?")
    print("=" * 90)
    print("\nSplit each subject's calibration trials into halves, recompute,")
    print("correlate across subjects. 200 random splits.\n")

    rs = {k: [] for k in DECODED}
    for _ in range(n_rep):
        halves = []
        order = {}                      # one random trial order per subject
        for s_, g_ in probs.groupby("subject"):
            tr = g_["trial"].unique()
            order[s_] = RNG.permutation(tr)
        for half in (0, 1):
            def sub(g, half=half):
                pick = order[g["subject"].iloc[0]][half::2]
                return g["trial"].isin(pick)
            halves.append(decoded_profile(probs, bounds, subset=sub)
                          .set_index("subj_id"))
        a, b = halves
        common = a.index.intersection(b.index)
        if len(common) >= 6:
            for k in DECODED:
                x, y = a.loc[common, k], b.loc[common, k]
                if x.std() > 1e-12 and y.std() > 1e-12:
                    rs[k].append(pearsonr(x, y)[0])

    rows = []
    print(f"{'measure':<18}{'split-half r':>15}{'95% range':>20}{'full-length':>14}")
    for k in DECODED:
        v = np.array(rs[k])
        if not len(v):
            print(f"{k:<18}   degenerate (no between-subject variance)")
            continue
        r = v.mean()
        sb = 2 * r / (1 + r) if r > -1 else np.nan
        print(f"{k:<18}{r:>+15.3f}"
              f"{f'[{np.percentile(v,2.5):+.2f}, {np.percentile(v,97.5):+.2f}]':>20}"
              f"{sb:>+14.3f}")
        rows.append(dict(measure=k, r=r, sb=sb,
                         lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# C - does calibration predict closed-loop behaviour?
# --------------------------------------------------------------------------- #

def part_c(prof, df):
    print("\n" + "=" * 90)
    print("(C) does the calibration profile transfer to closed loop?")
    print("=" * 90)

    cl = df.groupby("subject").apply(
        lambda g: pd.Series({
            "cl_level": g["arousal"].mean(),
            "cl_sd": np.mean([np.std(x[256:]) for x in g["new_arousal"]]),
            "cl_between_sd": g["arousal"].std(),
            "cl_perf": g["performance"].mean()}), include_groups=False).reset_index()
    m = prof.merge(cl, on="subject")

    print(f"\n{'calibration measure':<18}{'closed-loop target':<20}{'r':>9}{'p':>10}")
    rows = []
    for a in DECODED:
        for b in ["cl_level", "cl_sd", "cl_perf"]:
            r, p = pearsonr(m[a], m[b])
            flag = " *" if p < 0.05 else ""
            print(f"{a:<18}{b:<20}{r:>+9.3f}{p:>10.3f}{flag}")
            rows.append(dict(cal=a, target=b, r=r, p=p))
    print("\n  cal_level -> cl_level is the key row: if a subject's calibration")
    print("  arousal level predicts their closed-loop level, the profile carries")
    print("  real person-specific information across sessions and conditions.")
    return m, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# D - out-of-sample personalised band
# --------------------------------------------------------------------------- #

def evaluate_bands(df, prof, width_source, label, loso_traj=True):
    """Per-trial in-band metrics with a subject-specific width from calibration.

    The trajectory is refitted leave-one-subject-out so no part of the band -
    centre or width - uses the test subject's own closed-loop trials.
    """
    hc = df[df.difficulty == 1].reset_index(drop=True)
    control = hard_control(df)

    w = prof.set_index("subject")[width_source] if width_source else None
    if w is not None:
        # parameter-free monotone map: proportional to the subject's calibration
        # value, normalised to a median of 1.0, clipped to the standard grid
        mult = (w / w.median()).clip(min(C.BAND_MULTIPLIERS), max(C.BAND_MULTIPLIERS))
    else:
        mult = pd.Series(1.0, index=sorted(hc.subject.unique()))

    out = []
    for s in sorted(hc.subject.unique()):
        if loso_traj:
            train = control[control.subject != s]
        else:
            train = control
        if len(train) < 20:
            continue
        traj = C.optimal_trajectory(C.performance_surface(train))
        idx = np.where(hc.subject.to_numpy() == s)[0]
        met = C.deviation_metrics(hc, traj, float(mult.get(s, 1.0)), indices=idx)
        met["subject"] = s
        met["performance"] = hc.loc[idx, "performance"].to_numpy()
        met["multiplier"] = float(mult.get(s, 1.0))
        out.append(met)

    res = pd.concat(out, ignore_index=True)
    r, p = pearsonr(res["pct_in_band"], res["performance"])
    rho, prho = spearmanr(res["pct_in_band"], res["performance"])
    mm = smf.mixedlm("performance ~ pct_in_band", res,
                     groups=res["subject"]).fit(reml=False)
    d, dp = C.cohens_d_good_bad(res, col="pct_in_band")
    return dict(scheme=label, n=len(res), r=r, p=p, rho=rho, p_rho=prho,
                beta=mm.params["pct_in_band"], p_mixed=mm.pvalues["pct_in_band"],
                d=d, p_d=dp, mult_min=res.multiplier.min(),
                mult_max=res.multiplier.max()), res


def part_d(df, prof):
    print("\n" + "=" * 90)
    print("(D) band width from calibration, evaluated out of sample")
    print("=" * 90)
    print("\nTrajectory refitted leave-one-subject-out; width from calibration only.")
    print("No part of any subject's band uses that subject's closed-loop trials.\n")

    schemes = [(None, "universal (x1.0)")]
    schemes += [(k, f"width from {k}") for k in
                ["cal_sd", "cal_iqr", "cal_volatility"]]

    rows, frames = [], {}
    print(f"{'scheme':<28}{'n':>5}{'r':>8}{'p':>9}{'rho':>8}"
          f"{'Cohen d':>10}{'p(d)':>9}{'mult range':>14}")
    for src, label in schemes:
        try:
            res, frame = evaluate_bands(df, prof, src, label)
        except Exception as e:
            print(f"{label:<28} failed: {type(e).__name__}: {e}")
            continue
        rows.append(res)
        frames[label] = frame
        print(f"{label:<28}{res['n']:>5}{res['r']:>+8.3f}{res['p']:>9.4f}"
              f"{res['rho']:>+8.3f}{res['d']:>10.2f}{res['p_d']:>9.4f}"
              f"{f'{res['mult_min']:.2f}-{res['mult_max']:.2f}':>14}")

    out = pd.DataFrame(rows)

    # Is any personalised scheme reliably better than the universal band?
    print("\n  Bootstrap over subjects, difference in r against the universal band:")
    uni = frames["universal (x1.0)"]
    for label, frame in frames.items():
        if label.startswith("universal"):
            continue
        subs = sorted(uni.subject.unique())
        diffs = []
        for _ in range(N_BOOT):
            pick = RNG.choice(subs, len(subs), replace=True)
            a = pd.concat([uni[uni.subject == s] for s in pick])
            b = pd.concat([frame[frame.subject == s] for s in pick])
            if a.pct_in_band.std() < 1e-9 or b.pct_in_band.std() < 1e-9:
                continue
            diffs.append(pearsonr(b.pct_in_band, b.performance)[0]
                         - pearsonr(a.pct_in_band, a.performance)[0])
        diffs = np.array(diffs)
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        print(f"    {label:<26} delta r = {diffs.mean():+.3f}  "
              f"95% CI [{np.percentile(diffs,2.5):+.3f}, "
              f"{np.percentile(diffs,97.5):+.3f}]  p = {p:.3f}")
        out.loc[out.scheme == label, "delta_r"] = diffs.mean()
        out.loc[out.scheme == label, "delta_lo"] = np.percentile(diffs, 2.5)
        out.loc[out.scheme == label, "delta_hi"] = np.percentile(diffs, 97.5)
        out.loc[out.scheme == label, "delta_p"] = p
    return out


# --------------------------------------------------------------------------- #
# E - screen every calibration measure against closed-loop outcomes
# --------------------------------------------------------------------------- #

def part_e(df, prof):
    print("\n" + "=" * 90)
    print("(E) Holm-corrected screen of calibration measures")
    print("=" * 90)

    base = df.merge(prof[["subject"] + DECODED + PHYSIO], on="subject",
                    suffixes=("", "_cal"))
    feats = DECODED + [f for f in PHYSIO if f in prof.columns]
    for f in feats:
        base[f + "_cz"] = (base[f] - base[f].mean()) / base[f].std()

    rows = []
    for f in feats:
        z = f + "_cz"
        try:
            m = smf.mixedlm(
                "performance ~ C(difficulty)*arousal + C(difficulty)*I(arousal**2)"
                f" + {z} + arousal:{z} + I(arousal**2):{z}",
                base, groups=base["subject"], re_formula="~1").fit(reml=False)
            term = f"I(arousal ** 2):{z}"
            rows.append(dict(measure=f, beta=m.params[term], p=m.pvalues[term]))
        except Exception:
            rows.append(dict(measure=f, beta=np.nan, p=np.nan))

    r = pd.DataFrame(rows).dropna()
    r["p_holm"] = holm(r["p"].to_numpy())
    r = r.sort_values("p")
    print(f"\nModeration of the arousal-performance curve ({len(r)} measures):\n")
    print(f"{'measure':<18}{'beta':>10}{'p':>10}{'Holm':>10}")
    for x in r.itertuples():
        print(f"{x.measure:<18}{x.beta:>+10.3f}{x.p:>10.4f}{x.p_holm:>10.4f}"
              + ("  *" if x.p_holm < 0.05 else ""))

    # With 16 subjects a single influential case can carry a screen result.
    surv = r[r.p_holm < 0.05]["measure"].tolist()
    if surv:
        print("\n  Leave-one-subject-out robustness for the survivors:")
        for f in surv:
            z = f + "_cz"
            ps = []
            for s_out in sorted(base["subject"].unique()):
                d = base[base["subject"] != s_out]
                try:
                    m = smf.mixedlm(
                        "performance ~ C(difficulty)*arousal"
                        " + C(difficulty)*I(arousal**2)"
                        f" + {z} + arousal:{z} + I(arousal**2):{z}",
                        d, groups=d["subject"], re_formula="~1").fit(reml=False)
                    ps.append(m.pvalues[f"I(arousal ** 2):{z}"])
                except Exception:
                    ps.append(np.nan)
            ps = np.array(ps, dtype=float)
            worst = int(np.nanargmax(ps))
            print(f"    {f:<16} p ranges {np.nanmin(ps):.4f}-{np.nanmax(ps):.4f}; "
                  f"still p<0.05 in {int((ps < 0.05).sum())}/16 leave-one-out fits "
                  f"(worst when dropping subject {worst})")
            r.loc[r.measure == f, "loo_p_max"] = np.nanmax(ps)
            r.loc[r.measure == f, "loo_n_sig"] = int((ps < 0.05).sum())
    return r


def run():
    df = trial_table()
    prof = build_profile()
    part_a(prof)
    rel = part_b()
    merged, trans = part_c(prof, df)
    bands = part_d(df, prof)
    screen = part_e(df, prof)

    prof.to_csv(OUT / "calibration_profile.csv", index=False)
    rel.to_csv(OUT / "calibration_profile_reliability.csv", index=False)
    trans.to_csv(OUT / "calibration_profile_transfer.csv", index=False)
    bands.to_csv(OUT / "calibration_profile_bands.csv", index=False)
    screen.to_csv(OUT / "calibration_profile_screen.csv", index=False)
    print(f"\nwrote five CSVs to {OUT}")


if __name__ == "__main__":
    run()
