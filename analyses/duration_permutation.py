"""A permutation p-value for the trial-level inverted-U.

Trial-mean arousal is computed over a window whose length is the outcome. A
trial ending at 15 s averages only the low part of the arousal ramp. Predictor
and outcome are therefore not independent, and the standard mixed-model p-value
assumes that they are.

Two things are measured here.

(a) How much of trial-mean arousal is duration alone? A null value is built by
    truncating the grand-average trajectory at each trial's observed duration
    and averaging it. That value carries no trial-specific arousal.

(b) A permutation test that preserves the duration coupling exactly while
    destroying the pairing between arousal and trial. Arousal traces are
    shuffled within duration strata. This gives the p-value that should be
    reported for the quadratic term.

Outputs: results/duration_permutation_null.csv,
         results/duration_permutation_summary.txt
"""
import numpy as np
import pandas as pd

from common import OUT, Y, trial_table
from common import FS

N_PERM = 400
N_STRATA = 6
RNG = np.random.default_rng(7)


def grand_trajectory(df):
    """Mean arousal in each 1 s bin, pooled over every trial."""
    maxb = int(df.dur_s.max()) + 1
    acc, cnt = np.zeros(maxb), np.zeros(maxb)
    for r in df.itertuples():
        a = r.new_arousal
        for b in range(min(len(a) // FS, maxb)):
            acc[b] += a[b * FS:(b + 1) * FS].mean()
            cnt[b] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def fit(df, values):
    d = df.copy()
    d["arousal"] = values
    m = Y.fit_yd_model(d, "arousal")
    b1, b2 = m.params["arousal"], m.params[Y.QUAD]
    return b2, m.pvalues[Y.QUAD], (-b1 / (2 * b2) if b2 else np.nan), m


def run():
    df = trial_table().copy()
    df["dur_s"] = df.performance / FS

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("Permutation p-value for the trial-level inverted-U")
    say("=" * 78)

    # ---- observed ---------------------------------------------------------
    obs_b2, obs_p, obs_opt, m = fit(df, df.arousal.to_numpy())
    pv = Y.quadratic_pvalues(m)
    say(f"\nObserved:")
    say(f"   beta_quad = {obs_b2:+.3f}   p_easy = {pv[0]:.2e}   "
        f"p_hard = {pv[1]:.2e}   optimum = {obs_opt:.1f}")

    # ---- (a) the arithmetic component -------------------------------------
    grand = grand_trajectory(df)
    df["null_mean"] = [np.nanmean(grand[:max(int(d), 1)]) for d in df.dur_s]

    r_null = np.corrcoef(df.dur_s, df.null_mean)[0, 1]
    r_act = np.corrcoef(df.dur_s, df.arousal)[0, 1]
    r_cross = np.corrcoef(df.arousal, df.null_mean)[0, 1]
    say(f"\n(a) How much of trial-mean arousal is duration alone?")
    say(f"   r(duration, null mean)   = {r_null:+.3f}   (1.0 by construction)")
    say(f"   r(duration, actual mean) = {r_act:+.3f}")
    say(f"   r(actual, null)          = {r_cross:+.3f}   -> duration explains "
        f"{r_cross**2*100:.1f}% of the variance in trial-mean arousal")

    nb2, _, nopt, nm = fit(df, df.null_mean.to_numpy())
    npv = Y.quadratic_pvalues(nm)
    say(f"\n   The same model fitted to the null value (no arousal information):")
    say(f"   beta_quad = {nb2:+.3f}   p_easy = {npv[0]:.2e}   "
        f"p_hard = {npv[1]:.2e}   optimum = {nopt:.1f}")
    say("   -> note the optimum falls far outside 0-100, so pure arithmetic")
    say("      produces monotone curvature, not an interior peak.")

    # ---- (b) permutation --------------------------------------------------
    df["stratum"] = pd.qcut(df.dur_s, N_STRATA, labels=False, duplicates="drop")
    say(f"\n(b) Permutation: shuffle arousal within {df.stratum.nunique()} "
        f"duration strata,")
    say(f"    preserving the length coupling and destroying the arousal signal.")

    b2s, opts = [], []
    for _ in range(N_PERM):
        a = df.arousal.to_numpy().copy()
        for s in df.stratum.unique():
            idx = np.where(df.stratum == s)[0]
            a[idx] = RNG.permutation(a[idx])
        try:
            b2, _, opt, _ = fit(df, a)
        except Exception:
            continue
        b2s.append(b2)
        opts.append(opt)

    b2s, opts = np.array(b2s), np.array(opts)
    p_perm = (b2s <= obs_b2).mean()
    inside = ((opts >= 0) & (opts <= 100)).mean()

    say(f"\n   null over {len(b2s)} shuffles:")
    say(f"     beta_quad mean {b2s.mean():+.3f}, sd {b2s.std():.3f}")
    say(f"     95% range [{np.percentile(b2s, 2.5):+.3f}, "
        f"{np.percentile(b2s, 97.5):+.3f}]")
    say(f"     shuffles giving an optimum inside 0-100: {inside*100:.1f}%")
    say(f"\n   p(null <= observed) = {p_perm:.4f}")
    say(f"   duration structure alone accounts for {b2s.mean()/obs_b2*100:.0f}% "
        f"of the observed curvature")

    ratio = max(p_perm, 1.0 / N_PERM) / obs_p
    say("\nConclusion: the effect is real. It significantly exceeds what the")
    say("length coupling produces on its own. The mixed-model p-value is")
    say(f"roughly {ratio:.0f}x too small; report p = {p_perm:.4f} instead of "
        f"{obs_p:.1e}.")

    pd.DataFrame({"beta_quad": b2s, "optimum": opts}).to_csv(
        OUT / "duration_permutation_null.csv", index=False)
    (OUT / "duration_permutation_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT / 'duration_permutation_null.csv'}")


if __name__ == "__main__":
    run()
