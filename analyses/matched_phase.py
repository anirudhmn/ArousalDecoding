"""Matched-phase course discrimination.

Within a trial, ring size and elapsed time are confounded. Rings shrink every
30 s, so "tracks ring size" and "ramps with time" are the same statement. The
easy and hard courses differ in ring size at the same elapsed time, because the
hard course is physically smaller throughout. Comparing courses at matched time
therefore separates task difficulty from trial phase.

A pure time ramp predicts a course coefficient of zero. The motor index from
``motor_decoder.py`` serves as the specificity control.

Outputs: results/matched_phase_crossings_peripheral.csv,
         results/matched_phase_crossings_motor.csv,
         results/matched_phase_summary.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import D, OUT, cfg, trial_table
from common import FS

MOTOR_TABLE = OUT / "motor_decoder_trial_table.pkl"


def calibration_composition():
    """What the decoder is trained on, versus what it is applied to."""
    ring = D.load_ring_epochs()
    cal = ring[ring.condition == cfg.CALIBRATION_CONDITION]
    cl = ring[ring.condition != cfg.CALIBRATION_CONDITION]

    print("=" * 78)
    print("Matched-phase course discrimination")
    print("=" * 78)
    print("\nCalibration block (condition 0) - the decoder's training set")
    print(pd.crosstab(cal.trial_difficulty, cal.ring_size, margins=True).to_string())
    print(f"  trials: {cal.groupby(['subj_idx','trial_idx']).ngroups}")
    print("\nClosed-loop block (conditions 1-3) - where it is applied")
    print(pd.crosstab(cl.trial_difficulty, cl.ring_size, margins=True).to_string())
    print("\nRing diameters differ between courses: easy large/medium/small = "
          "108/72/36, hard = 60/30/14.4.")
    return cal, cl


def crossings(df):
    """Decoded arousal in the 1 s before every ring crossing."""
    rows = []
    for r in df.itertuples():
        if not r.events:
            continue
        a = r.new_arousal
        for size, samp in r.events:
            if samp >= len(a) or samp < FS:
                continue
            rows.append({"subject": r.subject, "difficulty": r.difficulty,
                         "condition": r.condition, "size": size,
                         "arousal": a[max(0, samp - FS):samp].mean(),
                         "t_s": samp / FS})
    return pd.DataFrame(rows)


def run():
    calibration_composition()

    tables = {"peripheral": trial_table()}
    if MOTOR_TABLE.exists():
        tables["motor"] = pd.read_pickle(MOTOR_TABLE)
    else:
        print("\n[warn] run motor_decoder.py first for the specificity control")

    frames = {k: crossings(v) for k, v in tables.items()}
    for k, e in frames.items():
        e.to_csv(OUT / f"matched_phase_crossings_{k}.csv", index=False)

    # --- ring size is inseparable from time within a trial -----------------
    print("\n" + "-" * 78)
    print("Within the hard course: does arousal track ring size, or just time?")
    print("-" * 78)
    h = frames["peripheral"]
    h = h[h.difficulty == 1].copy()
    h["sz"] = h["size"].map({"large": 0, "medium": 1, "small": 2})
    for f in ("arousal ~ sz", "arousal ~ sz + t_s"):
        m = smf.mixedlm(f, h, groups=h["subject"]).fit(reml=False)
        print(f"   {f:<24} beta_size={m.params['sz']:+7.2f}  "
              f"p={m.pvalues['sz']:.2e}")
    print("   -> ring size adds nothing once elapsed time is controlled;")
    print("      do not cite ring-size tracking as evidence.")

    # --- the test that does separate them ----------------------------------
    print("\n" + "-" * 78)
    print("Across courses at matched elapsed time (hard rings are smaller at")
    print("every time point). A pure time-ramp predicts beta_hard = 0.")
    print("-" * 78)
    print(f"\n{'index':<12}{'model':<34}{'beta_hard':>11}{'p':>13}")
    rows = []
    for name, e in frames.items():
        lg = e[e["size"] == "large"]
        for label, frame, formula in [
                ("all crossings, + time", e, "arousal ~ difficulty + t_s"),
                ("all crossings, + time + time^2", e,
                 "arousal ~ difficulty + t_s + I(t_s**2)"),
                ("large rings only, + time", lg, "arousal ~ difficulty + t_s")]:
            m = smf.mixedlm(formula, frame, groups=frame["subject"]).fit(reml=False)
            b, p = m.params["difficulty"], m.pvalues["difficulty"]
            print(f"{name:<12}{label:<34}{b:>+11.2f}{p:>13.2e}")
            rows.append(dict(index=name, model=label, beta=b, p=p, n=len(frame)))
        print()

    # --- the same thing, shown as a table ----------------------------------
    print("-" * 78)
    print("Mean decoded arousal by elapsed-time bin (hard minus easy)")
    print("-" * 78)
    for name, e in frames.items():
        d = e.copy()
        d["bin"] = pd.cut(d.t_s, [0, 10, 20, 30, 40, 50, 90])
        p = d.pivot_table(index="bin", columns="difficulty", values="arousal",
                          observed=True)
        p["hard - easy"] = p[1] - p[0]
        print(f"\n  {name}:")
        print(p.round(1).to_string())

    pd.DataFrame(rows).to_csv(OUT / "matched_phase_summary.csv", index=False)
    print(f"\nwrote {OUT / 'matched_phase_summary.csv'}")


if __name__ == "__main__":
    run()
