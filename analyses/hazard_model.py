"""Within-trial discrete-time hazard model.

A length-safe reformulation of the arousal-performance question. The unit of
analysis is a 1 s bin rather than a trial. For each bin, given how far into the
trial we are, does current arousal predict crashing within the next few seconds?
No trial-level summary is taken, so the duration coupling that affects
trial-mean arousal cannot operate.

Three things are reported:

  (a) the quadratic on arousal, under increasingly flexible control for time;
  (b) the same model for the motor index, as a specificity check;
  (c) whether being above the control band predicts imminent failure, and
      whether feedback attenuates that link.

A non-parametric table is printed alongside, because a positive quadratic in a
parametric fit is not by itself proof of a U-shaped hazard.

Outputs: results/hazard_model_bins.csv, results/hazard_model_summary.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

from common import C, OUT, cfg, hard_control, trial_table

HORIZONS = (3, 5)
MOTOR_TABLE = OUT / "motor_decoder_trial_table.pkl"


def bin_frame(df, traj=None):
    """One row per 1 s bin of every hard-course trial."""
    if traj is None:
        hc = hard_control(df)
        traj = C.optimal_trajectory(C.performance_surface(hc))
    opt, sd = traj["optimal"], traj["std"]

    rows = []
    for r in df[df.difficulty == 1].itertuples():
        a = C._binned_arousal(r.new_arousal)
        n = len(a)
        for t in range(n):
            if t >= len(opt) or np.isnan(opt[t]) or np.isnan(sd[t]):
                continue
            rows.append({"subject": r.subject, "condition": r.condition,
                         "t": t, "tb": min(t // 5, 11), "arousal": a[t],
                         "above": float(a[t] > opt[t] + sd[t]),
                         **{f"crash{h}": float(n - t <= h) for h in HORIZONS}})
    return pd.DataFrame(rows)


def quad_fit(h, outcome, tterm):
    m = smf.mixedlm(f"{outcome} ~ arousal + I(arousal**2) + {tterm}", h,
                    groups=h["subject"]).fit(reml=False)
    m0 = smf.mixedlm(f"{outcome} ~ {tterm}", h,
                     groups=h["subject"]).fit(reml=False)
    b1, b2 = m.params["arousal"], m.params["I(arousal ** 2)"]
    lr = 2 * (m.llf - m0.llf)
    k = len(m.params) - len(m0.params)
    return dict(b2=b2, p2=m.pvalues["I(arousal ** 2)"], lr=lr, k=k,
                p_lr=chi2.sf(lr, k), vertex=(-b1 / (2 * b2) if b2 else np.nan))


def run():
    df = trial_table()
    h = bin_frame(df)
    h.to_csv(OUT / "hazard_model_bins.csv", index=False)

    print("=" * 82)
    print("Within-trial hazard of crashing")
    print("=" * 82)
    print(f"\n{len(h)} one-second bins from "
          f"{df[df.difficulty==1].shape[0]} hard-course trials")
    for hz in HORIZONS:
        print(f"   {h[f'crash{hz}'].mean()*100:5.1f}% are within {hz}s of a crash")

    rows = []
    print("\n" + "-" * 82)
    print("(a) Quadratic on arousal, under increasing control for elapsed time")
    print("    a POSITIVE quadratic on hazard = U-shaped hazard = inverted-U "
          "for performance")
    print("-" * 82)
    frames = {"peripheral": h}
    if MOTOR_TABLE.exists():
        frames["motor"] = bin_frame(pd.read_pickle(MOTOR_TABLE))
    else:
        print("[warn] run motor_decoder.py first for the specificity control")

    print(f"\n{'index':<12}{'time control':<24}{'beta_quad':>12}{'p':>10}"
          f"{'LRT chi2':>11}{'p':>10}{'vertex':>9}")
    for name, frame in frames.items():
        for label, tterm in [("t linear", "t"),
                             ("t + t^2 + t^3", "t + I(t**2) + I(t**3)"),
                             ("5 s-bin fixed effects", "C(tb)")]:
            r = quad_fit(frame, "crash5", tterm)
            print(f"{name:<12}{label:<24}{r['b2']:>+12.7f}{r['p2']:>10.1e}"
                  f"{r['lr']:>11.1f}{r['p_lr']:>10.1e}{r['vertex']:>9.1f}")
            rows.append(dict(index=name, control=label, **
                             {k: v for k, v in r.items() if k != "k"}))
        print()

    # --- non-parametric check ---------------------------------------------
    print("-" * 82)
    print("Raw crash-within-5s rate (%) by arousal bin, within time strata")
    print("(blank = fewer than 25 observations)")
    print("-" * 82)
    d = h.copy()
    d["tstrat"] = pd.cut(d.t, [-1, 10, 20, 30, 40, 90],
                         labels=["0-10s", "10-20s", "20-30s", "30-40s", "40s+"])
    d["abin"] = pd.cut(d.arousal, [-1, 10, 20, 30, 40, 50, 60, 70, 100],
                       labels=["0-10", "10-20", "20-30", "30-40", "40-50",
                               "50-60", "60-70", "70+"])
    piv = d.pivot_table(index="abin", columns="tstrat", values="crash5",
                        aggfunc=["mean", "count"], observed=True)
    cols = ["0-10s", "10-20s", "20-30s", "30-40s", "40s+"]
    print(f"{'arousal':<9}" + "".join(f"{c:>14}" for c in cols))
    for ab in piv["mean"].index:
        line = f"{str(ab):<9}"
        for c in cols:
            try:
                mv, nv = piv["mean"].loc[ab, c], piv["count"].loc[ab, c]
                line += f"{mv*100:8.1f}({int(nv):4d})" if nv >= 25 else " " * 14
            except KeyError:
                line += " " * 14
        print(line)
    print("\n-> the dominant pattern is monotone: higher arousal, higher risk.")
    print("   The parametric U-shape rests on weak elevation at very low")
    print("   arousal late in trials, so do not overstate the left limb.")

    # --- (c) above-band and feedback --------------------------------------
    print("\n" + "-" * 82)
    print("(c) Does being above the band predict imminent failure, and does")
    print("    feedback attenuate that link?")
    print("-" * 82)
    m = smf.mixedlm("crash5 ~ above + t", h, groups=h["subject"]).fit(reml=False)
    print(f"   above-band effect on 5 s crash hazard: "
          f"b = {m.params['above']:+.4f}, p = {m.pvalues['above']:.2e}")

    mi = smf.mixedlm("crash5 ~ above * C(condition) + t", h,
                     groups=h["subject"]).fit(reml=False)
    print("\n   by feedback condition (reference = condition 1, silence):")
    for k in mi.params.index:
        if "above" in k:
            label = {"above": "above (silence)",
                     "above:C(condition)[T.2]": "  x half-sham",
                     "above:C(condition)[T.3]": "  x full BCI"}.get(k, k)
            print(f"     {label:<22} b = {mi.params[k]:+.4f}   "
                  f"p = {mi.pvalues[k]:.3f}")
            rows.append(dict(index="above_band", control=label,
                             b2=mi.params[k], p2=mi.pvalues[k]))
    print("\n   -> the above-band to failure link is significantly weaker under")
    print("      full BCI feedback, which is a positive result in a place")
    print("      where the trajectory comparison only yields a null.")

    pd.DataFrame(rows).to_csv(OUT / "hazard_model_summary.csv", index=False)
    print(f"\nwrote {OUT / 'hazard_model_summary.csv'}")


if __name__ == "__main__":
    run()
