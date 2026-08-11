"""Cluster-based permutation test on the condition trajectories, fully specified.

Every parameter is fixed here and printed with the results, so that the test can
be reproduced exactly:

  permutation unit           subject. Condition labels are permuted within
                             subject, because trials are nested in subjects and
                             permuting trials would break exchangeability.
  observation                one mean trajectory per subject per condition, so
                             each subject contributes equally regardless of how
                             many trials they completed.
  pointwise statistic        repeated-measures F across the three conditions,
                             and a paired t for the BCI against silence
                             contrast.
  cluster-forming threshold  the statistic's p < 0.05 critical value, stated
                             numerically in the output.
  cluster statistic          sum of the pointwise statistic over contiguous
                             supra-threshold time bins.
  null distribution          maximum cluster statistic per permutation, which
                             controls the family-wise error rate across time.
  permutations               10,000.

The headline result is a null, so the sensitivity of the test matters as much as
its p-value. The last section asks what size of sustained condition difference
this design could have detected.

Outputs: results/cluster_permutation_clusters.csv,
         results/cluster_permutation_sensitivity.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from common import C, OUT, trial_table

N_PERM = 10000
ALPHA = 0.05
WINDOW_S = 30            # seconds of trajectory tested
# The first second of every arousal trace is exactly zero (sliding-window
# inference starts at sample 256) and the IIR filter is still ramping through
# the second. Those bins carry no between-subject variance at all, so they are
# dropped rather than fed to a test that divides by their standard deviation.
WARMUP_BINS = 2
RNG = np.random.default_rng(2026)


# --------------------------------------------------------------------------- #
# Building the subject x condition x time array
# --------------------------------------------------------------------------- #

def trajectory_array(df, difficulty=1, window=WINDOW_S):
    """Mean binned trajectory per subject per condition, (n_subj, 3, window).

    Subjects missing any condition are dropped, so the repeated-measures
    structure is complete and permutation within subject is valid.
    """
    d = df[df.difficulty == difficulty]
    conds = [1, 2, 3]
    per = {}
    for r in d.itertuples():
        b = C._binned_arousal(r.new_arousal)
        if len(b) < WARMUP_BINS + window:
            continue
        per.setdefault((r.subject, r.condition), []).append(
            b[WARMUP_BINS:WARMUP_BINS + window])

    subjects = sorted({s for s, _ in per})
    keep = [s for s in subjects if all((s, c) in per for c in conds)]
    arr = np.stack([[np.mean(per[(s, c)], axis=0) for c in conds] for s in keep])
    return arr, keep


# --------------------------------------------------------------------------- #
# Pointwise statistics
# --------------------------------------------------------------------------- #

def rm_anova_f(x):
    """Repeated-measures one-way F at each time bin. x is (n_subj, k, T)."""
    n, k, _ = x.shape
    grand = x.mean(axis=(0, 1))
    cond_m = x.mean(axis=0)                       # (k, T)
    subj_m = x.mean(axis=1)                       # (n, T)
    ss_cond = n * ((cond_m - grand) ** 2).sum(axis=0)
    resid = x - cond_m[None] - subj_m[:, None] + grand
    ss_err = (resid ** 2).sum(axis=(0, 1))
    df1, df2 = k - 1, (k - 1) * (n - 1)
    f = np.zeros_like(ss_cond)
    ok = ss_err > 1e-9
    f[ok] = (ss_cond[ok] / df1) / (ss_err[ok] / df2)
    return f, df1, df2


def _t_of(d):
    """Paired t per bin, with degenerate (zero-variance) bins forced to 0.

    Without the guard a bin whose between-subject variance is exactly zero
    yields an arbitrarily large t from an epsilon denominator, which then
    dominates every cluster.
    """
    n = d.shape[0]
    sd = d.std(axis=0, ddof=1)
    t = np.zeros(d.shape[1])
    ok = sd > 1e-9
    t[ok] = d.mean(axis=0)[ok] / (sd[ok] / np.sqrt(n))
    return t


def paired_t(x, a, b):
    """Paired t at each time bin between two condition indices."""
    d = x[:, a] - x[:, b]
    return _t_of(d), d.shape[0] - 1


# --------------------------------------------------------------------------- #
# Cluster machinery
# --------------------------------------------------------------------------- #

def clusters(stat, thresh):
    """Contiguous runs above threshold, as (start, stop, summed statistic)."""
    above = stat >= thresh
    out, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            out.append((i, j + 1, float(stat[i:j + 1].sum())))
            i = j + 1
        else:
            i += 1
    return out


def max_cluster(stat, thresh):
    c = clusters(stat, thresh)
    return max((s for _, _, s in c), default=0.0)


def omnibus_test(x, n_perm=N_PERM, rng=RNG):
    stat, df1, df2 = rm_anova_f(x)
    thresh = f_dist.ppf(1 - ALPHA, df1, df2)
    obs = clusters(stat, thresh)

    null = np.empty(n_perm)
    for p in range(n_perm):
        xp = x.copy()
        for s in range(x.shape[0]):              # permute labels within subject
            xp[s] = x[s][rng.permutation(x.shape[1])]
        null[p] = max_cluster(rm_anova_f(xp)[0], thresh)

    res = [(a, b, s, float((null >= s).mean())) for a, b, s in obs]
    return dict(stat=stat, thresh=thresh, df=(df1, df2), clusters=res, null=null)


def pairwise_test(x, a, b, n_perm=N_PERM, rng=RNG):
    stat, dfree = paired_t(x, a, b)
    thresh = t_dist.ppf(1 - ALPHA / 2, dfree)
    obs = clusters(np.abs(stat), thresh)

    null = np.empty(n_perm)
    d = x[:, a] - x[:, b]
    n = d.shape[0]
    for p in range(n_perm):
        sign = rng.choice([-1.0, 1.0], size=n)[:, None]
        null[p] = max_cluster(np.abs(_t_of(d * sign)), thresh)

    res = [(s0, s1, s, float((null >= s).mean())) for s0, s1, s in obs]
    return dict(stat=stat, thresh=thresh, df=dfree, clusters=res, null=null,
                diff=d)


def report(name, r, rows):
    print(f"\n  {name}")
    print(f"    cluster-forming threshold {r['thresh']:.3f}  (df {r['df']})")
    if not r["clusters"]:
        print("    no supra-threshold clusters")
        rows.append(dict(test=name, cluster="none", start=np.nan, stop=np.nan,
                         mass=np.nan, p=np.nan, threshold=r["thresh"]))
    for i, (a, b, s, p) in enumerate(r["clusters"], 1):
        print(f"    cluster {i}: bins {a}-{b} s, mass {s:.2f}, p = {p:.4f}"
              + ("  *" if p < 0.05 else ""))
        rows.append(dict(test=name, cluster=i, start=a, stop=b, mass=s, p=p,
                         threshold=r["thresh"]))
    print(f"    null max-cluster mass: 95th pct {np.percentile(r['null'], 95):.2f}, "
          f"max {r['null'].max():.2f}")


# --------------------------------------------------------------------------- #
# Sensitivity: what would this design have detected?
# --------------------------------------------------------------------------- #

def sensitivity(x, a=2, b=0, n_sim=500, n_perm=1000, rng=RNG):
    """Smallest sustained offset detected in >=80% of simulations.

    A constant shift is added to one condition's trajectories, on top of the
    real between-subject variability, and the same cluster test is run. That
    turns a bare null into a bound: sustained differences of at least D points
    would have been detected, and nothing smaller can be excluded.
    """
    d = x[:, a] - x[:, b]
    n, T = d.shape
    dfree = n - 1
    thresh = t_dist.ppf(1 - ALPHA / 2, dfree)

    # The between-subject spread of the observed difference is what limits
    # detection; a sustained offset is tested against exactly that spread.
    sd_bin = d.std(axis=0, ddof=1).mean()
    sd_sustained = d.mean(axis=1).std(ddof=1)
    print(f"    observed difference: mean {d.mean():+.2f}, "
          f"per-bin between-subject SD {sd_bin:.2f}, "
          f"SD of subject means {sd_sustained:.2f}, SE {sd_sustained/np.sqrt(n):.2f}")

    out = []
    for delta in (0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10):
        hits = 0
        for _ in range(n_sim):
            # Resample subjects from the observed (centred) differences, so the
            # real between-subject variability and temporal structure are kept,
            # then add the sustained offset being probed.
            idx = rng.integers(0, n, size=n)
            sim = (d - d.mean(axis=0))[idx] + delta
            obs = max_cluster(np.abs(_t_of(sim)), thresh)
            null = np.empty(n_perm)
            for p in range(n_perm):
                s = rng.choice([-1.0, 1.0], size=n)[:, None]
                null[p] = max_cluster(np.abs(_t_of(sim * s)), thresh)
            hits += (null >= obs).mean() < 0.05
        power = hits / n_sim
        out.append(dict(delta=delta, power=power))
        print(f"    sustained offset of {delta:>4.1f} arousal points: "
              f"power {power:.2f}")
        if power >= 0.8:
            break
    return pd.DataFrame(out)


def run():
    df = trial_table()
    rows = []

    for difficulty, label in [(1, "hard course"), (0, "easy course")]:
        x, subjects = trajectory_array(df, difficulty)
        print("=" * 84)
        print(f"Cluster-based permutation, {label}")
        print("=" * 84)
        print(f"\n  {len(subjects)} subjects with all three conditions, "
              f"{WINDOW_S} one-second bins, {N_PERM} permutations")
        print(f"  mean arousal by condition: "
              + ", ".join(f"{c} {x[:, i].mean():.1f}"
                          for i, c in enumerate(["silence", "half-sham", "full BCI"])))

        r = omnibus_test(x)
        report(f"{label}: omnibus across three conditions", r, rows)

        for a, b, nm in [(2, 0, "full BCI vs silence"),
                         (2, 1, "full BCI vs half-sham"),
                         (1, 0, "half-sham vs silence")]:
            rp = pairwise_test(x, a, b)
            report(f"{label}: {nm}", rp, rows)
        print()

    res = pd.DataFrame(rows)

    # Every cluster tested here belongs to one family: 4 tests x 2 courses.
    # Reporting the uncorrected p alone would understate that multiplicity.
    fam = res.dropna(subset=["p"]).copy()
    if len(fam):
        from arousal.plotting import holm
        fam["p_holm"] = holm(fam["p"].to_numpy())
        res = res.merge(fam[["test", "cluster", "p_holm"]],
                        on=["test", "cluster"], how="left")
        print("=" * 84)
        print("Family-wise correction across all 8 trajectory tests")
        print("=" * 84)
        print(f"\n{'test':<44}{'p':>10}{'Holm':>10}")
        for r in fam.itertuples():
            print(f"{r.test:<44}{r.p:>10.4f}{r.p_holm:>10.4f}"
                  + ("  *" if r.p_holm < 0.05 else ""))
        print("\nCluster-level p-values already control FWE across time within a")
        print("test; this second correction covers the eight tests themselves.")

    res.to_csv(OUT / "cluster_permutation_clusters.csv", index=False)

    print("\n" + "=" * 84)
    print("Sensitivity of the hard-course BCI-vs-silence test")
    print("=" * 84)
    print("\n  How large a sustained difference would this design have caught?")
    x, _ = trajectory_array(df, 1)
    sens = sensitivity(x)
    sens.to_csv(OUT / "cluster_permutation_sensitivity.csv", index=False)

    det = sens[sens.power >= 0.8]
    if len(det):
        print(f"\n  -> detectable at 80% power: sustained offsets of about "
              f"{det.delta.iloc[0]:.1f} arousal points or more.")
        print("     Smaller sustained differences, and any localised difference,")
        print("     remain outside what this test can exclude.")
    else:
        print("\n  -> no tested offset reached 80% power; the test can exclude")
        print("     only very large sustained differences.")
    print(f"\nwrote {OUT/'cluster_permutation_clusters.csv'} and "
          f"{OUT/'cluster_permutation_sensitivity.csv'}")


if __name__ == "__main__":
    run()
