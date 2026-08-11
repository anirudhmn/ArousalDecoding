"""Are both limbs of the inverted-U supported?

The ``pct_in_band`` metric counts time below the band as in band, so it cannot
distinguish under-arousal from good regulation. Separating the three states
tests each limb directly, using a length-normalised metric that is not affected
by the trial-duration coupling in trial-mean arousal.

If both ``pct_above`` and ``pct_below`` predict worse performance, the
inverted-U is measured cleanly. If only ``pct_above`` does, the defensible claim
is a ceiling rather than a symmetric optimum.

Outputs: results/band_limbs.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import C, OUT, hard_control, trial_table
from common import FS


def three_way(df, traj, mult=1.0):
    """Per-trial time above, strictly within, and below the band."""
    opt, sd = traj["optimal"], traj["std"]
    T = len(opt)
    out = []
    for r in df.itertuples():
        b = C._binned_arousal(r.new_arousal)
        above = below = inside = 0
        for t, v in enumerate(b[:T]):
            if np.isnan(opt[t]) or np.isnan(sd[t]):
                continue
            if v > opt[t] + sd[t] * mult:
                above += 1
            elif v < opt[t] - sd[t] * mult:
                below += 1
            else:
                inside += 1
        tot = above + below + inside
        if tot:
            out.append({"subject": r.subject,
                        "performance": r.performance / FS,
                        "pct_above": above / tot * 100,
                        "pct_strict": inside / tot * 100,
                        "pct_below": below / tot * 100})
    return pd.DataFrame(out)


def run():
    df = trial_table()
    hc = hard_control(df)
    traj = C.optimal_trajectory(C.performance_surface(hc))
    m = three_way(hc, traj)
    m.to_csv(OUT / "band_limbs.csv", index=False)

    print("=" * 78)
    print("Both limbs of the inverted-U, measured separately")
    print("=" * 78)
    print(f"\n{len(m)} hard-course control trials")
    print(f"   mean time above band     {m.pct_above.mean():5.1f}%")
    print(f"   mean time strictly within {m.pct_strict.mean():5.1f}%")
    print(f"   mean time below band     {m.pct_below.mean():5.1f}%")
    print(f"   trials with ANY time below band: {(m.pct_below > 0).sum()} "
          f"of {len(m)} ({(m.pct_below > 0).mean()*100:.0f}%)")

    print("\n" + "-" * 78)
    print("Does time on each side predict worse performance?")
    print("(both negative and significant = a genuine inverted-U)")
    print("-" * 78)
    for f in ("performance ~ pct_above",
              "performance ~ pct_below",
              "performance ~ pct_above + pct_below"):
        mm = smf.mixedlm(f, m, groups=m["subject"]).fit(reml=True)
        ci = mm.conf_int()
        print(f"\n  {f}")
        for t in ("pct_above", "pct_below"):
            if t in mm.params.index:
                print(f"      {t:<11} b = {mm.params[t]:+.3f} "
                      f"[{ci.loc[t,0]:+.3f}, {ci.loc[t,1]:+.3f}]  "
                      f"p = {mm.pvalues[t]:.4f}")

    print("\n" + "-" * 78)
    print("Binned means")
    print("-" * 78)
    for col in ("pct_above", "pct_below"):
        d = m.copy()
        d["b"] = pd.cut(d[col], [-0.1, 0, 10, 25, 50, 101])
        g = d.groupby("b", observed=True)["performance"].agg(["mean", "count"])
        print(f"\n  {col}:")
        for b, row in g.iterrows():
            if row["count"] >= 5:
                print(f"    {str(b):<13} n={int(row['count']):>3}  "
                      f"{row['mean']:5.1f}s  " + "#" * int(row["mean"]))

    print("\n-> the falling limb is robust; the rising limb has almost no data")
    print("   behind it, because arousal ramps upward from zero and the task")
    print("   never produces under-arousal during demanding phases.")
    print(f"\nwrote {OUT / 'band_limbs.csv'}")


if __name__ == "__main__":
    run()
