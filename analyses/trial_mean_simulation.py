"""Can the trial-mean estimator manufacture an inverted-U?

A simulation with a deliberately monotone ground truth. Arousal ramps within a
trial, and higher arousal only ever increases the hazard of crashing. There is
no optimum anywhere in the generating process.

Suppose that fitting performance on mean arousal and its square returns a
significant negative quadratic here. The estimator would then be producing the
inverted-U shape from a truth that has none, and the observed shape could not be
taken as evidence for the Yerkes-Dodson relationship on its own.

Outputs: printed summary only
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RNG = np.random.default_rng(4)
N_SUBJ, N_TRIAL = 16, 22
FS_BIN = 1.0          # 1 s bins
MAX_T = 90


def simulate(monotone=True):
    rows = []
    for s in range(N_SUBJ):
        base = RNG.normal(0, 6)                 # subject offset
        for tr in range(N_TRIAL):
            a0 = 5 + RNG.normal(0, 6) + base    # starting arousal
            k = RNG.uniform(0.6, 2.2)           # ramp rate, varies by trial
            arousal, t = [], 0
            while t < MAX_T:
                a = a0 + k * t + RNG.normal(0, 4)
                a = float(np.clip(a, 0, 100))
                arousal.append(a)
                # Hazard: monotone increasing in arousal, nothing else.
                if monotone:
                    h = 0.002 + 0.0016 * max(a - 20, 0)
                else:                            # true inverted-U, for contrast
                    h = 0.004 + 0.00004 * (a - 50) ** 2
                if RNG.random() < h:
                    break
                t += 1
            if len(arousal) >= 5:
                rows.append({"subject": s, "performance": len(arousal),
                             "arousal": float(np.mean(arousal)),
                             "early": float(np.mean(arousal[5:15]))
                             if len(arousal) > 15 else np.nan,
                             "difficulty": 1, "condition": 1})
    return pd.DataFrame(rows)


def fit(df, col):
    d = df.dropna(subset=[col]).copy()
    d["a"] = d[col]
    m = smf.mixedlm("performance ~ a + I(a**2)", d, groups=d["subject"]).fit(reml=False)
    m0 = smf.mixedlm("performance ~ a", d, groups=d["subject"]).fit(reml=False)
    b1, b2 = m.params["a"], m.params["I(a ** 2)"]
    vertex = -b1 / (2 * b2) if b2 else np.nan
    return dict(n=len(d), beta2=b2, p=m.pvalues["I(a ** 2)"],
                daic=m.aic - m0.aic, vertex=vertex)


def run():
    print("=" * 78)
    print("Does the trial-mean estimator invent an inverted-U?")
    print("=" * 78)

    for label, mono in [("MONOTONE truth (no optimum exists)", True),
                        ("TRUE inverted-U (optimum at 50)", False)]:
        df = simulate(mono)
        print(f"\n{label}   [{len(df)} simulated trials]")
        for col, name in [("arousal", "trial-mean arousal"),
                          ("early", "fixed 5-15 s window")]:
            r = fit(df, col)
            verdict = ("INVERTED-U" if r["beta2"] < 0 and r["p"] < 0.05
                       else "no inverted-U")
            print(f"   {name:<22} beta2={r['beta2']:+8.4f}  p={r['p']:.2e}  "
                  f"dAIC={r['daic']:+7.1f}  vertex={r['vertex']:6.1f}   -> {verdict}")

    print("\n" + "-" * 78)
    print("Reading: if the monotone simulation yields a significant negative")
    print("quadratic on trial-mean arousal, the shape is an artefact of the")
    print("estimator, not evidence of an optimum. The fixed-window row shows")
    print("what the same data look like without the length coupling.")


if __name__ == "__main__":
    run()
