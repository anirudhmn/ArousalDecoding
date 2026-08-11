"""Does a motor decoder also recover the inverted-U?

If joystick and head motion decode task demand as accurately as peripheral
physiology, then classification accuracy alone cannot establish that the
peripheral signal is an arousal index. Two outcomes are possible:

  * the motor index also recovers the inverted-U at trial level, in which case
    accuracy is not diagnostic and the validity argument has to rest on
    something else;
  * the motor index does not recover it, in which case accuracy and
    physiological validity come apart.

The motor decoder is trained and applied through the identical pipeline as the
peripheral one: calibration-block training, sliding-window inference,
per-subject percentile normalisation, asymmetric IIR smoothing.

Outputs: results/motor_decoder_arousal.pkl,
         results/motor_decoder_trial_table.pkl
"""
import pickle
import time

import numpy as np
import pandas as pd
import torch

from common import C, D, OUT, Y, cfg
from arousal import training as T
from arousal.config import CHANNEL_GROUPS, CH
from arousal.signals import normalize_train_val

MOTOR_CHANNELS = [CH[c] for c in (CHANNEL_GROUPS["Joystick Controls"]
                                  + CHANNEL_GROUPS["Head Motion"])]
MODALITIES_MOTOR = {"joystick": list(range(12)), "head": list(range(12, 18))}
N_REPS = 5
CACHE = OUT / "motor_decoder_arousal.pkl"


def decode():
    device = T.get_device()
    online = D.load_ring_epochs_online()
    fixed = D.load_ring_epochs()
    print(f"device: {device} | {N_REPS} repetitions")

    results, t0 = [], time.time()
    for s in cfg.SUBJECTS:
        calib = fixed[(fixed.subj_idx == s)
                      & (fixed.condition == cfg.CALIBRATION_CONDITION)
                      & (fixed.label >= 0)]
        X_train = np.stack(calib["data"].to_numpy())[:, MOTOR_CHANNELS, :]
        y_train = calib["label"].to_numpy()

        subj = online[(online.subj_idx == s) & (online.label >= 0)]
        trials, _, _ = D.concatenate_trials(
            subj[subj.condition != cfg.CALIBRATION_CONDITION],
            channels=MOTOR_CHANNELS)

        Xn, trials_n = normalize_train_val(X_train, trials)
        results.append(T.decode_subject_trials(
            Xn, y_train, trials_n, MODALITIES_MOTOR, device,
            n_repetitions=N_REPS, seed_base=7000 + 100 * s))
        print(f"  S{s:02d}: {len(trials)} trials  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    with open(CACHE, "wb") as f:
        pickle.dump(results, f)
    return results


def evaluate(results):
    df = D.build_trial_table(results)
    ref = pd.read_pickle(
        __import__("common").ROOT / "data" / "trial_table.pkl")

    print("\n" + "=" * 74)
    print("Yerkes-Dodson fit for the motor decoder")
    print("=" * 74)
    print(f"motor trials: {len(df)}   reference trials: {len(ref)}")

    out = {}
    for label, frame, col in [("motor", df, "arousal"),
                              ("peripheral", ref, "arousal"),
                              ("EEG feedback", ref, "arousal_old_mean")]:
        m = Y.fit_yd_model(frame, col)
        lin = Y.fit_yd_model(frame, col, Y.FORMULA_LINEAR)
        pv = Y.quadratic_pvalues(m)
        o0, _ = Y.optimum_with_ci(m, 0)
        o1, _ = Y.optimum_with_ci(m, 1)
        out[label] = dict(beta=m.params[Y.QUAD], p_easy=pv[0], p_hard=pv[1],
                          aic=m.aic, aic_lin=lin.aic, opt_e=o0, opt_h=o1)

    print(f"\n{'decoder':<14}{'beta quad':>11}{'p easy':>10}{'p hard':>10}"
          f"{'opt E/H':>15}{'dAIC quad-lin':>15}")
    for k, v in out.items():
        print(f"{k:<14}{v['beta']:>+11.3f}{v['p_easy']:>10.1e}{v['p_hard']:>10.1e}"
              f"   {v['opt_e']:6.1f}/{v['opt_h']:6.1f}"
              f"{v['aic']-v['aic_lin']:>+15.1f}")

    print("\n(a negative beta and a negative dAIC together indicate a genuine "
          "inverted-U)")

    # How similar are the two arousal indices trial by trial?
    merged = df[["subject", "trial_idx", "arousal"]].merge(
        ref[["subject", "trial_idx", "arousal"]], on=["subject", "trial_idx"],
        suffixes=("_motor", "_periph"))
    r = np.corrcoef(merged.arousal_motor, merged.arousal_periph)[0, 1]
    print(f"\ntrial-mean correlation, motor vs peripheral index: r = {r:+.3f} "
          f"(n = {len(merged)})")

    df.to_pickle(OUT / "motor_decoder_trial_table.pkl")
    return df, out


if __name__ == "__main__":
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            results = pickle.load(f)
        print("loaded cached motor decoding")
    else:
        results = decode()
    evaluate(results)
