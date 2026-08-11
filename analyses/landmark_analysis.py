"""Landmark analysis, a length-safe view of the inverted-U.

Pick a landmark time L. Keep every trial still flying at L. Measure arousal in a
short window ending at L, and ask how much longer the trial lasts.

Every trial contributes one observation, all measured over the same window and
all alive at the same moment. The predictor cannot depend on how long the trial
ran, because it is taken before the outcome begins. This is the standard fix for
length coupling in survival data.

Outputs: results/landmark_analysis_30s.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import OUT, trial_table, FS

WINDOW = 5          # seconds of arousal averaged, ending at the landmark


def landmark_frame(df, L, window=WINDOW):
    rows = []
    for r in df.itertuples():
        n = len(r.new_arousal) / FS
        if n <= L:                       # already crashed - excluded
            continue
        a = r.new_arousal[int((L - window) * FS):int(L * FS)].mean()
        rows.append({"subject": r.subject, "difficulty": r.difficulty,
                     "condition": r.condition, "arousal": a,
                     "remaining": n - L, "total": n})
    return pd.DataFrame(rows)


def fit(d):
    m = smf.mixedlm("remaining ~ arousal + I(arousal**2) + C(difficulty) + C(condition)",
                    d, groups=d["subject"]).fit(reml=False)
    m0 = smf.mixedlm("remaining ~ arousal + C(difficulty) + C(condition)",
                     d, groups=d["subject"]).fit(reml=False)
    b1, b2 = m.params["arousal"], m.params["I(arousal ** 2)"]
    return dict(n=len(d), b1=b1, b2=b2, p1=m.pvalues["arousal"],
                p2=m.pvalues["I(arousal ** 2)"], daic=m.aic - m0.aic,
                vertex=(-b1 / (2 * b2) if b2 else np.nan), model=m)


def run():
    df = trial_table()
    print("=" * 82)
    print("Landmark analysis (arousal at time L -> remaining flight time)")
    print("=" * 82)
    print(f"\n{'landmark':>9}{'n alive':>9}{'beta_lin':>11}{'p_lin':>10}"
          f"{'beta_quad':>11}{'p_quad':>10}{'dAIC':>9}{'vertex':>9}")
    best = None
    for L in (10, 15, 20, 25, 30, 35):
        d = landmark_frame(df, L)
        if len(d) < 60:
            print(f"{L:>9}{len(d):>9}   too few trials still flying")
            continue
        r = fit(d)
        flag = " *" if (r["b2"] < 0 and r["p2"] < 0.05) else ""
        print(f"{L:>9}{r['n']:>9}{r['b1']:>+11.3f}{r['p1']:>10.3f}"
              f"{r['b2']:>+11.4f}{r['p2']:>10.3f}{r['daic']:>+9.1f}"
              f"{r['vertex']:>9.1f}{flag}")
        if best is None or r["p2"] < best[1]["p2"]:
            best = (L, r, d)

    L, r, d = best
    print(f"\nbest landmark = {L} s   (* marks a significant negative quadratic,")
    print(" i.e. an inverted-U in remaining flight time)")

    print(f"\nbinned view at L = {L} s:")
    d = d.copy()
    d["bin"] = pd.cut(d.arousal, [0, 15, 30, 45, 60, 100])
    g = d.groupby("bin", observed=True)["remaining"].agg(["mean", "sem", "count"])
    for b, row in g.iterrows():
        bar = "#" * int(row["mean"] * 1.2)
        print(f"   arousal {str(b):<12} n={int(row['count']):>3}  "
              f"{row['mean']:5.1f} +/- {row['sem']:4.1f} s  {bar}")

    print(f"\nfull model at L = {L} s:")
    m = r["model"]
    ci = m.conf_int()
    for t in ("arousal", "I(arousal ** 2)"):
        print(f"   {t:<18} beta={m.params[t]:+8.4f}  "
              f"95% CI [{ci.loc[t,0]:+.4f}, {ci.loc[t,1]:+.4f}]  p={m.pvalues[t]:.4f}")
    print(f"   optimum arousal = {r['vertex']:.1f}")

    d.to_csv(OUT / f"landmark_analysis_landmark_{L}s.csv", index=False)

    # Same analysis on the motor index, as a specificity check.
    motor = pd.read_pickle(OUT / "motor_decoder_trial_table.pkl")
    dm = landmark_frame(motor, L)
    rm = fit(dm)
    print(f"\nspecificity check, motor index at the same landmark:")
    print(f"   beta_quad={rm['b2']:+.4f}  p={rm['p2']:.3f}  dAIC={rm['daic']:+.1f}  "
          f"vertex={rm['vertex']:.1f}")


if __name__ == "__main__":
    run()
