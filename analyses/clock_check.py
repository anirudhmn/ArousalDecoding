"""Is the decoded index a clock, and is the motor index an arousal index?

(a) IS THE DECODED INDEX JUST ELAPSED TIME?
    It ramps within a trial at a mean r of about +0.59, which is what a clock
    would do. But a clock makes a strong prediction: every trial ends at a high
    value, because every trial ends late in its own ramp. If a substantial share
    of trials crash at low arousal, the index carries trial-specific information
    that a clock cannot.

(b) DOES THE MOTOR INDEX TRACK AUTONOMIC PHYSIOLOGY THE WAY THE PERIPHERAL
    INDEX DOES?
    The motor decoder, built from joystick and head motion, reaches 86 percent
    AUC and reproduces the U-shaped crash hazard. If it also correlates with HR,
    HRV, EDA and pupil, the two indices are not cleanly separable and recovery
    of the inverted-U cannot be treated as diagnostic of an arousal signal. If
    it does not, the motor decoder is a decoder of task difficulty that is not
    an arousal signal.

Outputs: results/clock_check_endpoints.csv,
         results/clock_check_motor_external.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp

from common import OUT, trial_table
from common import FS

MOTOR_TABLE = OUT / "motor_decoder_trial_table.pkl"
SIGNALS = {"HR": "hr_signal", "HRV-pNN35": "hrv_signal",
           "EDA-phasic": "eda_phasic", "EDA-tonic": "eda_tonic",
           "Pupil": "pupil_signal"}
END_S = 3          # seconds before the end used as the "final" value


def endpoints(df):
    """Arousal in the last few seconds of each trial, plus trial duration."""
    rows = []
    for r in df.itertuples():
        a = np.asarray(r.new_arousal, float)
        if len(a) < (END_S + 5) * FS:
            continue
        rows.append({"subject": r.subject, "difficulty": r.difficulty,
                     "condition": r.condition,
                     "duration_s": len(a) / FS,
                     "final": a[-END_S * FS:].mean(),
                     "peak": a[256:].max(),
                     "mean": a[256:].mean()})
    return pd.DataFrame(rows)


def part_a(df):
    e = endpoints(df)
    e.to_csv(OUT / "clock_check_endpoints.csv", index=False)

    print("=" * 84)
    print("(a) does every trial end high, as a clock would require?")
    print("=" * 84)
    print(f"\n{len(e)} trials. Arousal in the final {END_S} s before the trial ends.")

    for name, sub in [("all trials", e), ("hard course", e[e.difficulty == 1])]:
        f = sub["final"]
        print(f"\n  {name}  (n = {len(sub)})")
        print(f"    mean {f.mean():.1f}, median {f.median():.1f}, "
              f"SD {f.std():.1f}, range {f.min():.1f}-{f.max():.1f}")
        for thr in (25, 50, 75):
            print(f"    ending below {thr:>3}: {100*(f < thr).mean():5.1f}% of trials")
        r, p = pearsonr(sub["duration_s"], f)
        print(f"    correlation with trial duration: r = {r:+.3f} (p = {p:.3g})")

    print("\n  A clock predicts every trial ends near the top of its ramp and that")
    print("  final value rises steeply with duration. Compare the spread and the")
    print("  duration correlation above against that prediction.")

    # Quartile view: does a long trial guarantee a high ending?
    print("\n  Final arousal by duration quartile (hard course):")
    h = e[e.difficulty == 1].copy()
    h["q"] = pd.qcut(h["duration_s"], 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"])
    for q, g in h.groupby("q", observed=True):
        print(f"    {q:<9} n={len(g):>3}  duration {g.duration_s.mean():5.1f}s  "
              f"final arousal {g['final'].mean():5.1f} +/- {g['final'].std():4.1f}"
              f"   below 50: {100*(g['final']<50).mean():4.1f}%")
    return e


def part_b(df):
    """Run the same external validation on the motor index."""
    if not MOTOR_TABLE.exists():
        print("\n[skip] run motor_decoder.py first")
        return None
    motor = pd.read_pickle(MOTOR_TABLE)

    print("\n" + "=" * 84)
    print("(b) does the motor index track autonomic physiology?")
    print("=" * 84)
    print("\nWithin-trial correlation with each raw signal, averaged over trials.")
    print("The peripheral column repeats that validation on the same trials.")

    rows = []
    print(f"\n{'signal':<14}{'peripheral':>13}{'motor':>10}{'motor p':>12}")
    for sig, col in SIGNALS.items():
        per_r, mot_r = [], []
        for tbl, acc in ((df, per_r), (motor, mot_r)):
            for r in tbl.itertuples():
                a = np.asarray(r.new_arousal, float)
                b = np.asarray(getattr(r, col), float)
                n = min(len(a), len(b))
                a, b = a[256:n], b[256:n]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() > 512 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9:
                    acc.append(pearsonr(a[m], b[m])[0])
        per_r, mot_r = np.array(per_r), np.array(mot_r)
        t, p = ttest_1samp(mot_r, 0)
        print(f"{sig:<14}{per_r.mean():>+13.3f}{mot_r.mean():>+10.3f}{p:>12.2e}")
        rows.append(dict(signal=sig, peripheral_r=per_r.mean(),
                         motor_r=mot_r.mean(), motor_p=p,
                         n_peripheral=len(per_r), n_motor=len(mot_r)))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "clock_check_motor_external.csv", index=False)

    print("\n  Mean |r| across the five signals: "
          f"peripheral {res.peripheral_r.abs().mean():.3f}, "
          f"motor {res.motor_r.abs().mean():.3f}")
    print(f"  Ratio: the peripheral index tracks autonomic physiology "
          f"{res.peripheral_r.abs().mean()/max(res.motor_r.abs().mean(),1e-9):.1f}x "
          f"more strongly.")

    # Are the two indices even related?
    both = df[["subject", "trial_idx"]].copy()
    both["per"] = [np.mean(x[256:]) for x in df["new_arousal"]]
    mm = motor[["subject", "trial_idx"]].copy()
    mm["mot"] = [np.mean(x[256:]) for x in motor["new_arousal"]]
    j = both.merge(mm, on=["subject", "trial_idx"])
    r, p = pearsonr(j["per"], j["mot"])
    rho, prho = spearmanr(j["per"], j["mot"])
    print(f"\n  Trial-mean correlation between the two indices: "
          f"r = {r:+.3f} (p = {p:.3g}), rho = {rho:+.3f} (p = {prho:.3g}), n = {len(j)}")
    return res


def run():
    df = trial_table()
    part_a(df)
    res = part_b(df)

    print("\n" + "=" * 84)
    print("Reading")
    print("=" * 84)
    print("""
  (a) decides whether the decoded index is a clock. A clock requires every
  trial to end near the top of its own ramp. The endpoint distribution above
  shows how many trials instead crash at low arousal, which elapsed time alone
  cannot produce.

  (b) decides how the motor index should be described. If it does not track
  autonomic physiology, it is a decoder of task difficulty that is not an
  arousal signal, and recovery of the inverted-U cannot be treated as
  diagnostic of arousal. If it does track autonomic physiology, then it is an
  arousal-correlated behavioural signal, and recovering the inverted-U is what
  a noisy arousal proxy is expected to do rather than a counterexample. The
  ratio and the trial-level correlation printed above are what separate those
  two readings.""")


if __name__ == "__main__":
    run()
