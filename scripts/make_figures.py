#!/usr/bin/env python3
"""Build every figure in the paper.

Figures 1, 2, 6 and 7 need the cached trial table and the stored decoder
outputs. Figures 3, 4 and 5 read the tables written by ``analyses/``, so run
``analyses/run_all.sh`` first.

    python scripts/make_figures.py            # all seven
    python scripts/make_figures.py 3 5        # only those
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analyses"))

from arousal import config as cfg
from arousal import control as C
from arousal import yerkes as Y
from arousal.plotting import (FS_AN, FS_AX, FS_PAN, FS_TK, fmt_p, holm, panel,
                              stars, tidy)

RES = ROOT / "analyses" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Okabe-Ito, which stays distinguishable under the common colour-vision
# deficiencies and in greyscale.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GREY = "#7F7F7F"
BLACK = "#000000"

mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.dpi": 110,
    "savefig.dpi": 300,
})

SEC = cfg.FS


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def trial_table():
    return pd.read_pickle(cfg.DATA / "trial_table.pkl")


def hard_control(df):
    return df[(df.difficulty == 1)
              & (df.condition.isin(cfg.CONTROL_CONDITIONS))].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figure 1: decoder accuracy, generalisation and attribution
# --------------------------------------------------------------------------- #

FBCSP_AUC, FBCSP_SEM = 79.8, 7.2


def figure_1():
    results = {}
    for name in ("physio", "eeg", "all"):
        with open(cfg.RESULTS / f"results_offline_simple_{name}.pkl", "rb") as f:
            results[name] = pickle.load(f)
    per_subject = {n: r["final_results"][:, :, -1, 1].mean(axis=1) * 100
                   for n, r in results.items()}

    loso = pd.read_csv(RES / "loso_decoder.csv")
    ctrl = pd.read_csv(RES / "negative_controls.csv")

    fig = plt.figure(figsize=(15, 7.6))
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.62, 1], hspace=0.26,
                           wspace=0.36, width_ratios=[1.30, 1, 1.05], figure=fig)

    ax_a = fig.add_subplot(gs[0, :])
    ax_a.imshow(Image.open(cfg.ASSETS / "nn_architecture.png"))
    ax_a.axis("off")
    ax_a.set_position([0.175, 0.585, 0.66, 0.39])
    panel(ax_a, "A", x=-0.01, y=1.00)

    # -- B: within-subject AUC, decoders and controls ------------------------
    ax_b = fig.add_subplot(gs[1, 0])
    labels = ["FBCSP", "EEG", "Autonomic", "Auto.\n+ EEG", "Motor", "Shuffled"]
    vals = [FBCSP_AUC, per_subject["eeg"].mean(), per_subject["physio"].mean(),
            per_subject["all"].mean(), ctrl.motor.mean(), ctrl.shuffled.mean()]
    errs = [FBCSP_SEM,
            per_subject["eeg"].std(ddof=1) / 4, per_subject["physio"].std(ddof=1) / 4,
            per_subject["all"].std(ddof=1) / 4,
            ctrl.motor.std(ddof=1) / 4, ctrl.shuffled.std(ddof=1) / 4]
    cols = [GREY, SKY, BLUE, BLUE, ORANGE, "#BBBBBB"]

    x = np.arange(len(labels))
    bars = ax_b.bar(x, vals, color=cols, edgecolor="black", linewidth=0.8,
                    width=0.66, zorder=2)
    ax_b.errorbar(x, vals, yerr=errs, fmt="none", ecolor="black", capsize=3.5,
                  linewidth=1.0, zorder=3)
    for b, v, e in zip(bars, vals, errs):
        ax_b.text(b.get_x() + b.get_width() / 2, v + e + 1.2, f"{v:.1f}",
                  ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax_b.axhline(50, color="black", lw=0.9, ls=":", alpha=0.7)
    ax_b.text(-0.55, 51.2, "chance", fontsize=8.5, ha="left", alpha=0.8)
    ax_b.set_ylabel("Within-subject AUC (%)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=9.5)
    ax_b.set_ylim(40, 107)
    ax_b.axvline(3.5, color="#BBBBBB", lw=1.0, zorder=1)
    ax_b.text(1.5, 101.5, "decoders", ha="center", fontsize=9.5, style="italic")
    ax_b.text(4.5, 101.5, "controls", ha="center", fontsize=9.5, style="italic")
    tidy(ax_b)
    panel(ax_b, "B", x=-0.20)

    # -- C: leave-one-subject-out, paired by subject -------------------------
    ax_c = fig.add_subplot(gs[1, 1])
    piv = loso.pivot(index="subject", columns="feature_set", values="auc")
    for _, row in piv.iterrows():
        ax_c.plot([0, 1], [row["eeg"], row["physio"]], color=GREY, lw=0.9,
                  alpha=0.65, marker="o", ms=3.5, zorder=2)
    for i, (fs, col) in enumerate([("eeg", SKY), ("physio", BLUE)]):
        v = piv[fs].to_numpy()
        ax_c.scatter([i] * len(v), v, s=42, color=col, edgecolor="black",
                     linewidth=0.6, zorder=3)
        ax_c.plot([i - 0.16, i + 0.16], [v.mean()] * 2, color="black", lw=2.4,
                  zorder=4)
        ax_c.text(i, 100.5, f"{v.mean():.1f}", ha="center", fontsize=10,
                  fontweight="bold")
    ax_c.axhline(50, color="black", lw=0.9, ls=":", alpha=0.7)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["EEG", "Autonomic"])
    ax_c.set_xlim(-0.45, 1.45)
    ax_c.set_ylim(40, 106)
    ax_c.set_ylabel("Leave-one-subject-out AUC (%)")
    ax_c.text(0.5, 43.2, "difference +5.9,\n95% CI [$-$2.9, +14.8]", ha="center",
              fontsize=9, style="italic")
    tidy(ax_c)
    panel(ax_c, "C", x=-0.22)

    # -- D: integrated gradients --------------------------------------------
    ax_d = fig.add_subplot(gs[1, 2])
    imps = results["all"]["imps"]
    styles = {"EEG": dict(color=GREY, lw=1.5, ls="-"),
              "HR": dict(color=RED, lw=2.0, ls="-"),
              "RESP": dict(color=GREEN, lw=1.6, ls="--"),
              "EDA": dict(color=BLUE, lw=1.8, ls="-"),
              "Pupil": dict(color=PURPLE, lw=1.5, ls=":")}
    t = np.linspace(-2, 0, imps.shape[-1])
    for name, chans in cfg.MODALITIES_ALL.items():
        ax_d.plot(t[:-1], imps[chans].mean(axis=0)[:-1], label=name, **styles[name])
    ax_d.set_xlabel("Time relative to ring crossing (s)")
    ax_d.set_ylabel("Mean integrated gradient")
    ax_d.set_xlim(-2, 0)
    tidy(ax_d)
    ax_d.legend(fontsize=9, frameon=False, loc="upper left")
    panel(ax_d, "D", x=-0.22)

    save(fig, "figure_1.png")


# --------------------------------------------------------------------------- #
# Figure 2: Yerkes-Dodson, at matched scaling
# --------------------------------------------------------------------------- #

def figure_2():
    from scaling_parity import build_variants

    df = build_variants(trial_table())
    # The scaling variants are stored as traces; the models take trial means.
    df["eeg_subject"] = [float(np.mean(x)) for x in df["eeg_subj_cl"]]
    df = df[np.isfinite(df["eeg_subject"])].reset_index(drop=True)

    perm = pd.read_csv(RES / "duration_permutation_null.csv")
    obs = -2.5616

    models = {
        "new_quad": Y.fit_yd_model(df, "arousal"),
        "new_lin": Y.fit_yd_model(df, "arousal", Y.FORMULA_LINEAR),
        "old_quad": Y.fit_yd_model(df, "eeg_subject"),
        "old_lin": Y.fit_yd_model(df, "eeg_subject", Y.FORMULA_LINEAR),
    }
    aic = {k: m.aic for k, m in models.items()}
    best = min(aic.values())
    p_perm = float((perm.beta_quad <= obs).mean())

    fig = plt.figure(figsize=(15, 4.8))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.34,
                           width_ratios=[1, 1, 0.9, 0.9])
    grid = np.linspace(0, 100, 300)

    def plot_yd(ax, model, col, letter, xlabel, show_y=True, mark_ns=False):
        sig = Y.quadratic_pvalues(model)
        preds = Y.predict_curve(model, grid)
        colours = {0: SKY, 1: RED}
        names = {0: "Easy", 1: "Hard"}
        if mark_ns:
            names = {d: f"{names[d]} ({stars(sig[d])})" for d in (0, 1)}
        for d in (0, 1):
            sub = df[df.difficulty == d]
            ax.scatter(sub[col], sub.performance / SEC, color=colours[d],
                       alpha=0.28, s=13, linewidths=0, zorder=2)
        for d in (0, 1):
            p = preds[d]
            ax.plot(grid, p["mean"] / SEC, color=colours[d], lw=2.4,
                    label=names[d], zorder=3)
            ax.fill_between(grid, p["ci_lo"] / SEC, p["ci_hi"] / SEC,
                            color=colours[d], alpha=0.16, zorder=1)
            opt, _ = Y.optimum_with_ci(model, d)
            if 0 <= opt <= 100:
                ax.axvline(opt, color=colours[d], ls="--", lw=1.1, alpha=0.7)
        ax.set_xlabel(xlabel)
        if show_y:
            ax.set_ylabel("Flight time (s)")
        ax.set_xlim(-3, 103)
        tidy(ax)
        panel(ax, letter, x=-0.20)
        ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=2)

    ax_a = fig.add_subplot(gs[0])
    plot_yd(ax_a, models["new_quad"], "arousal", "A", "Decoded arousal (autonomic)")
    ax_a.text(0.03, 0.02, f"permutation $p$ = {p_perm:.4f}", transform=ax_a.transAxes,
              fontsize=9.5, style="italic")

    ax_b = fig.add_subplot(gs[1])
    plot_yd(ax_b, models["old_quad"], "eeg_subject", "B",
            "Decoded arousal (EEG)", show_y=False, mark_ns=True)

    # -- C: permutation null --------------------------------------------------
    ax_c = fig.add_subplot(gs[2])
    ax_c.hist(perm.beta_quad, bins=32, color="#CCCCCC", edgecolor="white",
              linewidth=0.5, zorder=2)
    ax_c.axvline(obs, color=RED, lw=2.4, zorder=3)
    ax_c.axvline(perm.beta_quad.mean(), color=BLUE, lw=1.8, ls="--", zorder=3)
    ax_c.text(obs, ax_c.get_ylim()[1] * 0.96, " observed", color=RED,
              fontsize=9, va="top")
    ax_c.text(perm.beta_quad.mean(), ax_c.get_ylim()[1] * 0.72,
              " duration-only\n mean", color=BLUE, fontsize=9, va="top")
    ax_c.set_xlabel(r"$\beta$ (arousal$^2$)")
    ax_c.set_ylabel("Permutations")
    tidy(ax_c)
    panel(ax_c, "C", x=-0.24)

    # -- D: AIC ---------------------------------------------------------------
    ax_d = fig.add_subplot(gs[3])
    order = ["new_quad", "new_lin", "old_quad", "old_lin"]
    names = ["Auto.\nquad.", "Auto.\nlinear", "EEG\nquad.", "EEG\nlinear"]
    deltas = [aic[k] - best for k in order]
    bars = ax_d.bar(range(4), deltas, color=[BLUE, BLUE, SKY, SKY],
                    edgecolor="black", linewidth=0.7, width=0.6, zorder=2)
    for b, h in zip(bars, ["", "///", "", "///"]):
        b.set_hatch(h)
    for b, v in zip(bars, deltas):
        ax_d.text(b.get_x() + b.get_width() / 2, v + max(deltas) * 0.025,
                  f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax_d.set_xticks(range(4))
    ax_d.set_xticklabels(names, fontsize=8.5)
    ax_d.set_ylabel(r"$\Delta$AIC (vs best)")
    ax_d.set_ylim(0, max(deltas) * 1.28)
    tidy(ax_d)
    panel(ax_d, "D", x=-0.24)

    save(fig, "figure_2.png")
    print(f"    permutation p = {p_perm:.4f}; dAIC = "
          + ", ".join(f"{k} {aic[k]-best:.1f}" for k in order))


# --------------------------------------------------------------------------- #
# Figure 3: validity of the decoded index
# --------------------------------------------------------------------------- #

def figure_3():
    ends = pd.read_csv(RES / "clock_check_endpoints.csv")
    mp = pd.read_csv(RES / "scaling_parity_matched_phase.csv")
    mp_motor = pd.read_csv(RES / "matched_phase_summary.csv")
    ext = pd.read_csv(RES / "external_validation.csv")
    land = pd.read_csv(RES / "landmark_analysis_30s.csv")

    fig = plt.figure(figsize=(15, 8.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

    # -- A: where trials end --------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    hard = ends[ends.difficulty == 1]
    ax.hist(hard["final"], bins=np.arange(0, 105, 5), color=BLUE,
            edgecolor="white", linewidth=0.6, zorder=2)
    ax.axvline(50, color=BLACK, ls="--", lw=1.3, zorder=3)
    below25 = 100 * (hard["final"] < 25).mean()
    below50 = 100 * (hard["final"] < 50).mean()
    ax.text(0.03, 0.96, f"{below25:.1f}% of hard trials end below 25\n"
                        f"{below50:.1f}% end below 50\n"
                        f"SD = {hard['final'].std():.1f} over the full range",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBBBBB"))
    ax.set_xlabel("Decoded arousal in the final 3 s")
    ax.set_ylabel("Hard-course trials")
    ax.set_xlim(0, 100)
    tidy(ax)
    panel(ax, "A", x=-0.13)

    # -- B: matched-phase course discrimination -------------------------------
    ax = fig.add_subplot(gs[0, 1])
    rows = [
        ("Autonomic", "peripheral (per-subject)", BLUE),
        ("EEG, per-trial", "EEG (per-trial)", SKY),
        ("EEG, per-subject", "EEG, per-subject percentile", "#9FCFE8"),
        ("Motor", None, ORANGE),
    ]
    specs = ["all crossings, + time", "all crossings, + time + t^2",
             "large rings only, + time"]
    spec_labels = ["+ time", "+ time + time$^2$", "large rings only"]
    width = 0.24
    for j, (label, key, colour) in enumerate(rows):
        for i, spec in enumerate(specs):
            if key is None:
                m = mp_motor[(mp_motor["index"] == "motor")
                             & (mp_motor.model.str.startswith(spec[:18]))]
            else:
                m = mp[(mp["index"] == key) & (mp.model == spec)]
            if not len(m):
                continue
            b, p = float(m.beta.iloc[0]), float(m.p.iloc[0])
            xpos = j + (i - 1) * width
            ax.bar(xpos, b, width=width * 0.9, color=colour,
                   alpha=[1.0, 0.72, 0.45][i], edgecolor="black",
                   linewidth=0.6, zorder=2)
            ax.text(xpos, b + (0.22 if b >= 0 else -0.55), stars(p),
                    ha="center", fontsize=8, rotation=90 if b > 3 else 0)
    ax.axhline(0, color=BLACK, lw=1.0)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylabel(r"Hard vs easy at matched time ($\beta$)")
    ax.legend(handles=[mpl.patches.Patch(facecolor=GREY, alpha=a, edgecolor="black",
                                         label=l)
                       for a, l in zip([1.0, 0.72, 0.45], spec_labels)],
              fontsize=8.5, frameon=False, loc="upper right")
    ax.set_ylim(-1.6, 8.6)
    tidy(ax)
    panel(ax, "B", x=-0.13)

    # -- C: external validation ----------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    sub = ext[ext.method == "RECOMMENDED within-trial"].copy()
    order = ["HR", "HRV-pNN35", "EDA-phasic", "EDA-tonic", "Pupil"]
    sub = sub.set_index("signal").loc[order].reset_index()
    ypos = np.arange(len(order))[::-1]
    for y, (_, r) in zip(ypos, sub.iterrows()):
        sig = not (pd.notna(r.p) and r.p > 0.05)
        colour = BLUE if r.r > 0 else RED
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=colour, lw=2.4,
                alpha=1.0 if sig else 0.4, zorder=2)
        ax.scatter([r.r], [y], s=70, color=colour, edgecolor="black",
                   linewidth=0.7, zorder=3, alpha=1.0 if sig else 0.4)
        if not sig:
            ax.text(r.ci_hi + 0.03, y, "n.s.", va="center", fontsize=9,
                    style="italic")
    ax.axvline(0, color=BLACK, lw=1.0, ls="--", alpha=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order)
    ax.set_xlabel("Within-trial correlation with decoded arousal")
    ax.set_xlim(-0.42, 0.68)
    tidy(ax, grid_axis="x")
    panel(ax, "C", x=-0.20)

    # -- D: landmark regression ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(land.arousal, land.remaining, s=26, color=BLUE, alpha=0.5,
               linewidths=0, zorder=2)
    x = land.arousal.to_numpy()
    y = land.remaining.to_numpy()
    c = np.polyfit(x, y, 2)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, np.polyval(c, xs), color=BLACK, lw=2.4, zorder=3)
    ax.axvline(23.4, color=RED, ls="--", lw=1.3, zorder=3)
    ax.text(24.5, ax.get_ylim()[1] * 0.94, "optimum 23.4", color=RED, fontsize=9.5)
    ax.text(0.03, 0.04, r"$\beta$(arousal$^2$) = $-$0.0021, $p$ = 0.033",
            transform=ax.transAxes, fontsize=9.5, style="italic")
    ax.set_xlabel("Decoded arousal in the 5 s ending at 30 s")
    ax.set_ylabel("Remaining flight time (s)")
    tidy(ax)
    panel(ax, "D", x=-0.13)

    save(fig, "figure_3.png")


# --------------------------------------------------------------------------- #
# Figure 4: individual differences, from calibration
# --------------------------------------------------------------------------- #

CAL_LABELS = {"cal_sd": "Calibration arousal SD", "resp": "Respiration",
              "hrv": "HRV-pNN35"}

SCREEN_LABELS = {
    "cal_sd": "calibration\narousal SD", "resp": "respiration", "hrv": "HRV-pNN35",
    "theta_beta": "theta / beta", "cal_level": "calibration\narousal level",
    "gamma": "gamma power", "cal_volatility": "calibration\nvolatility",
    "cal_iqr": "calibration\narousal IQR", "hrv_sd": "HRV variability",
    "alpha": "alpha power", "hr": "heart rate", "theta": "theta power",
    "cal_frac_high": "calibration\ntime high", "beta": "beta power",
    "eda_t": "EDA tonic", "pup": "pupil", "eda_p": "EDA phasic",
    "alpha_theta": "alpha / theta", "hr_sd": "HR variability",
    "pup_sd": "pupil variability",
}


def _interaction_model(df, feature):
    import statsmodels.formula.api as smf
    f = (f"performance ~ C(difficulty)*arousal + C(difficulty)*I(arousal**2)"
         f" + {feature}_z + arousal:{feature}_z + I(arousal**2):{feature}_z")
    return smf.mixedlm(f, data=df, groups=df["subject"], re_formula="~1").fit(reml=False)


def figure_4():
    from calibration_profile import build_profile

    df = trial_table()
    prof = build_profile()
    keep = ["subject"] + [c for c in CAL_LABELS if c in prof.columns]
    merged = df.merge(prof[keep], on="subject", how="left", suffixes=("", "_cal"))
    for c in CAL_LABELS:
        col = c if c in merged.columns else c + "_cal"
        merged[c + "_z"] = (merged[col] - merged[col].mean()) / merged[col].std()

    models = {c: _interaction_model(merged, c) for c in CAL_LABELS}
    screen = pd.read_csv(RES / "calibration_profile_screen.csv")

    fig = plt.figure(figsize=(15, 8.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30,
                           height_ratios=[1, 0.92])

    # Restrict the plotted range to the observed trial-mean arousal, so the
    # curves are not read off an extrapolated region.
    lo, hi = np.percentile(merged.arousal, [2, 98])
    grid = np.linspace(lo, hi, 600)
    slices = np.linspace(-1.5, 1.5, 120)
    norm = mpl.colors.Normalize(vmin=-1.5, vmax=1.5)
    cmap = plt.get_cmap("coolwarm")

    axes = []
    for k, (feature, label) in enumerate(CAL_LABELS.items()):
        ax = fig.add_subplot(gs[0, k])
        axes.append(ax)
        m = models[feature]
        for z in slices:
            pred = m.predict(pd.DataFrame({"arousal": grid, "difficulty": 1,
                                           "condition": 1, f"{feature}_z": z}))
            ax.plot(grid, pred / SEC, color=cmap(norm(z)), lw=1.1, alpha=0.55)
        for z, style in [(-1.5, "-"), (1.5, "-")]:
            pred = m.predict(pd.DataFrame({"arousal": grid, "difficulty": 1,
                                           "condition": 1, f"{feature}_z": z}))
            ax.plot(grid, pred / SEC, color=cmap(norm(z)), lw=2.6, ls=style,
                    label=f"{'low' if z < 0 else 'high'} ($z$ = {z:+.1f})")
        term = f"I(arousal ** 2):{feature}_z"
        ax.set_xlabel("Decoded arousal")
        ax.set_title(label, pad=6)
        ax.set_xlim(lo, hi)
        ax.text(0.5, 0.03, f"arousal$^2$ interaction: {fmt_p(m.pvalues[term])}",
                transform=ax.transAxes, ha="center", fontsize=9.5)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        tidy(ax)
        panel(ax, "ABC"[k], x=-0.17)
    axes[0].set_ylabel("Predicted flight time (s)")
    lims = [a.get_ylim() for a in axes]
    for a in axes:
        a.set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))
    for a in axes[1:]:
        a.tick_params(labelleft=False)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes,
                        orientation="vertical", pad=0.015, fraction=0.02,
                        aspect=26)
    cbar.set_label("Baseline measure (z)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # -- D: the screen --------------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    s = screen.sort_values("p").reset_index(drop=True)
    xs = np.arange(len(s))
    colours = [GREEN if r.p_holm < 0.05 else (ORANGE if r.p < 0.05 else "#BBBBBB")
               for r in s.itertuples()]
    ax.bar(xs, -np.log10(s.p), color=colours, edgecolor="black", linewidth=0.6,
           width=0.66, zorder=2)
    for i, r in enumerate(s.itertuples()):
        if r.p_holm < 0.05:
            ax.annotate(f"Holm {r.p_holm:.3f}, {int(r.loo_n_sig)}/16 folds",
                        xy=(i, -np.log10(r.p)),
                        xytext=(i + 0.9, -np.log10(r.p) + 0.35 - 0.28 * i),
                        fontsize=8.5, ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="#666666"))
    ax.axhline(-np.log10(0.05), color=BLACK, ls="--", lw=1.1)
    ax.text(len(s) - 0.4, -np.log10(0.05) + 0.03, "p = 0.05", ha="right",
            fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([SCREEN_LABELS.get(m, m) for m in s.measure],
                       rotation=38, ha="right", fontsize=8.5)
    ax.set_ylabel(r"$-\log_{10} p$ (uncorrected)")
    ax.set_ylim(0, 3.5)
    ax.legend(handles=[
        mpl.patches.Patch(facecolor=GREEN, edgecolor="black",
                          label="survives Holm across the screen"),
        mpl.patches.Patch(facecolor=ORANGE, edgecolor="black",
                          label="nominally significant only"),
        mpl.patches.Patch(facecolor="#BBBBBB", edgecolor="black",
                          label="not significant")],
        fontsize=9, frameon=False, loc="upper right")
    tidy(ax)
    panel(ax, "D", x=-0.055)

    save(fig, "figure_4.png")
    for f, m in models.items():
        t = f"I(arousal ** 2):{f}_z"
        print(f"    {f:<8} beta={m.params[t]:+.3f} p={m.pvalues[t]:.4f}")


# --------------------------------------------------------------------------- #
# Figure 5: what feedback does
# --------------------------------------------------------------------------- #

def figure_5():
    fl = pd.read_csv(RES / "failure_locked_arousal.csv")
    bins = pd.read_csv(RES / "hazard_model_bins.csv")
    sens = pd.read_csv(RES / "cluster_permutation_sensitivity.csv")
    df = trial_table()

    fig = plt.figure(figsize=(15, 8.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28)

    cond_col = {1: GREY, 2: SKY, 3: BLUE}
    cond_lab = {1: "Silence", 2: "Half-sham", 3: "Full BCI"}

    # -- A: arousal locked to the crash --------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    span = 20
    hard = df[df.difficulty == 1]
    for c in (1, 2, 3):
        rows = []
        for r in hard[hard.condition == c].itertuples():
            tr = np.asarray(r.new_arousal, float)
            if len(tr) < span * SEC:
                continue
            rows.append([tr[len(tr) - (k + 1) * SEC:len(tr) - k * SEC].mean()
                         for k in range(span)][::-1])
        a = np.array(rows)
        t = np.arange(-span, 0) + 0.5
        m = a.mean(axis=0)
        se = a.std(axis=0, ddof=1) / np.sqrt(len(a))
        ax.fill_between(t, m - se, m + se, color=cond_col[c], alpha=0.18)
        ax.plot(t, m, color=cond_col[c], lw=2.4,
                label=f"{cond_lab[c]} (n = {len(a)})")
    ax.axvline(0, color=BLACK, lw=1.2, ls="--")
    ax.text(-0.4, ax.get_ylim()[0] + 2, "crash", rotation=90, ha="right",
            fontsize=9)
    ax.set_xlabel("Time before trial end (s)")
    ax.set_ylabel("Decoded arousal")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    tidy(ax)
    panel(ax, "A", x=-0.14)

    # -- B: the rise, by condition -------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for i, c in enumerate((1, 2, 3)):
        g = fl[fl.condition == c]
        se = g.rise.std(ddof=1) / np.sqrt(len(g))
        ax.bar(i, g.rise.mean(), yerr=se, color=cond_col[c], edgecolor="black",
               linewidth=0.7, width=0.58, capsize=4, zorder=2)
        ax.text(i, g.rise.mean() + se + 0.6, f"+{g.rise.mean():.1f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color=BLACK, lw=1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([cond_lab[c] for c in (1, 2, 3)])
    ax.set_ylabel("Arousal rise into the crash")
    ax.text(0.5, 0.94, "overall +14.1, $p$ = 1.7e-11\ncondition effect $p$ = 0.16",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB"))
    ax.set_ylim(0, 26)
    tidy(ax)
    panel(ax, "B", x=-0.18)

    # -- C: crash hazard against arousal, within elapsed-time strata ---------
    ax = fig.add_subplot(gs[1, 0])
    b = bins[bins.arousal > 0].copy()
    strata = [(0, 15, GREY, "0 to 15 s"), (15, 30, SKY, "15 to 30 s"),
              (30, 999, BLUE, "30 s and later")]
    for lo, hi, colour, lab in strata:
        s = b[(b.t >= lo) & (b.t < hi)].copy()
        if len(s) < 200:
            continue
        s["q"] = pd.qcut(s.arousal, 6, labels=False, duplicates="drop")
        g = s.groupby("q").agg(ar=("arousal", "mean"), hz=("crash5", "mean"),
                               n=("crash5", "size"))
        err = np.sqrt(g.hz * (1 - g.hz) / g.n) * 100
        ax.errorbar(g.ar, g.hz * 100, yerr=err, marker="o", ms=6, lw=2.0,
                    color=colour, capsize=3, label=lab, zorder=3)
    ax.set_xlabel("Decoded arousal (sextile mean within stratum)")
    ax.set_ylabel("Crash within 5 s (%)")
    ax.text(0.03, 0.96, "hazard model with elapsed time controlled:\n"
                        r"$\beta$(arousal$^2$) > 0, $p$ = 3.5e-10;"
                        "\nminimum hazard near arousal 23",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(fontsize=9, frameon=False, loc="lower right", title="elapsed time",
              title_fontsize=9)
    tidy(ax)
    panel(ax, "C", x=-0.14)

    # -- D: what the trajectory test could have detected ----------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(sens.delta, sens.power, marker="o", ms=6, lw=2.2, color=BLUE,
            zorder=3)
    ax.axhline(0.8, color=RED, ls="--", lw=1.3)
    ax.text(sens.delta.max(), 0.82, "80% power", ha="right", color=RED,
            fontsize=9.5)
    ax.axvline(2.76, color=GREY, ls=":", lw=1.4)
    ax.text(2.9, 0.05, "observed\nBCI vs silence\n(+2.8)", fontsize=9,
            color="#555555")
    ax.set_xlabel("Sustained condition difference (arousal points)")
    ax.set_ylabel("Power of the cluster test")
    ax.set_ylim(0, 1.0)
    tidy(ax)
    panel(ax, "D", x=-0.18)

    save(fig, "figure_5.png")


# --------------------------------------------------------------------------- #
# Figure 6: the optimal trajectory and the deviation metrics
# --------------------------------------------------------------------------- #

MAX_TIME = 60


def _mean_trace(trials, idx, n=MAX_TIME):
    rows = []
    for i in idx:
        b = C._binned_arousal(trials.loc[i, "new_arousal"])[:n]
        if len(b) < 5:
            continue
        rows.append(np.pad(b, (0, n - len(b)), constant_values=np.nan))
    a = np.array(rows)
    k = np.sum(~np.isnan(a), axis=0)
    mean = np.full(n, np.nan)
    se = np.full(n, np.nan)
    ok = k >= 2
    mean[ok] = np.nanmean(a[:, ok], axis=0)
    se[ok] = np.nanstd(a[:, ok], axis=0, ddof=1) / np.sqrt(k[ok])
    return mean, se


def figure_6():
    from scipy.stats import pearsonr, sem, ttest_ind

    trials = hard_control(trial_table())
    surface = C.performance_surface(trials)
    traj = C.optimal_trajectory(surface)
    fitted = traj["confidence"] >= C.MIN_CONFIDENCE_FOR_FIT
    metrics = C.deviation_metrics(trials, traj, 1.0)

    fig = plt.figure(figsize=(16, 9.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.28,
                           width_ratios=[2.1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # -- A: performance surface ----------------------------------------------
    extent = [surface["time_bins"][0], surface["time_bins"][-1],
              surface["arousal_bins"][0], surface["arousal_bins"][-1]]
    masked = np.ma.masked_invalid(surface["perf_smooth"])
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#D9D9D9")
    im = ax_a.imshow(masked, aspect="auto", origin="lower", extent=extent,
                     cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax_a, pad=0.015, fraction=0.030)
    cbar.set_label("Mean flight time (s)", fontsize=10.5)
    ax_a.plot(traj["time_centers"][fitted], traj["optimal"][fitted],
              color="white", ls="--", lw=3.0, zorder=3)
    ax_a.plot(traj["time_centers"][fitted], traj["optimal"][fitted],
              color=RED, ls="--", lw=1.6, zorder=4, label="Optimal trajectory")
    ax_a.set_xlabel("Trial time (s)")
    ax_a.set_ylabel("Decoded arousal")
    ax_a.set_xlim(0, MAX_TIME)
    ax_a.set_ylim(extent[2], extent[3])
    ax_a.legend(handles=[Line2D([0], [0], color=RED, ls="--", lw=2,
                                label="Optimal trajectory"),
                         mpl.patches.Patch(facecolor="#D9D9D9", edgecolor="black",
                                           label="no trials in bin")],
                fontsize=9.5, frameon=True, loc="upper right", framealpha=0.9)
    tidy(ax_a, grid_axis="none")
    panel(ax_a, "A", x=-0.075)

    # -- B: in-band against flight time --------------------------------------
    r, p = pearsonr(metrics.pct_in_band, metrics.performance)
    ceiling = metrics.pct_in_band >= 99.9
    ax_b.scatter(metrics.pct_in_band[~ceiling], metrics.performance[~ceiling],
                 alpha=0.6, s=30, color=BLUE, linewidths=0, zorder=2)
    ax_b.scatter(metrics.pct_in_band[ceiling], metrics.performance[ceiling],
                 alpha=0.85, s=34, facecolor="none", edgecolor=ORANGE,
                 linewidth=1.2, zorder=3,
                 label=f"at 100% ceiling ({100*ceiling.mean():.1f}%)")
    c = np.polyfit(metrics.pct_in_band, metrics.performance, 1)
    xs = np.linspace(metrics.pct_in_band.min(), metrics.pct_in_band.max(), 100)
    ax_b.plot(xs, np.polyval(c, xs), color=BLACK, ls="--", lw=1.8, zorder=4)
    ax_b.text(0.04, 0.96, f"r = {r:.3f}, p < 0.001\nmixed model "
                          r"$\beta$ = +0.180 [+0.088, +0.271]",
              transform=ax_b.transAxes, va="top", fontsize=9.5)
    ax_b.set_xlabel("% time within optimal band")
    ax_b.set_ylabel("Flight time (s)")
    ax_b.legend(fontsize=9, frameon=False, loc="lower right")
    tidy(ax_b)
    panel(ax_b, "B", x=-0.20)

    # -- C: trajectories by performance group --------------------------------
    perf = trials.performance / SEC
    q25, q75 = np.percentile(perf, [25, 75])
    groups = {"top": trials.index[perf >= q75],
              "mid": trials.index[(perf > q25) & (perf < q75)],
              "bottom": trials.index[perf <= q25]}
    t_c = traj["time_centers"][:MAX_TIME]
    opt = traj["optimal"][:MAX_TIME]
    std = traj["std"][:MAX_TIME]
    ax_c.fill_between(t_c, opt - std, opt + std, alpha=0.22, color="#CCCCCC",
                      label="$\\pm$1 SD band")
    ax_c.plot(t_c, opt, color=BLACK, lw=2.0, ls="--", label="Optimal trajectory")
    for key, colour, lab in [("top", GREEN, f"Good ($\\geq${q75:.0f} s)"),
                             ("mid", GREY, "Middle"),
                             ("bottom", RED, f"Bad ($\\leq${q25:.0f} s)")]:
        m, se_ = _mean_trace(trials, groups[key])
        ok = ~np.isnan(m)
        ax_c.fill_between(t_c[ok], (m - se_)[ok], (m + se_)[ok], alpha=0.22,
                          color=colour)
        ax_c.plot(t_c[ok], m[ok], color=colour, lw=2.4, label=lab)
    fit_end = traj["time_centers"][fitted].max()
    ax_c.axvline(fit_end, color=BLACK, lw=1.1, ls=":", alpha=0.7)
    ax_c.text(fit_end + 0.7, 4, "trajectory fit ends", fontsize=9, alpha=0.8)
    ax_c.set_xlabel("Trial time (s)")
    ax_c.set_ylabel("Decoded arousal")
    ax_c.set_xlim(0, MAX_TIME)
    ax_c.set_ylim(0, 100)
    ax_c.legend(fontsize=9, frameon=False, loc="upper left", ncol=2)
    tidy(ax_c)
    panel(ax_c, "C", x=-0.075)

    # -- D: in-band by performance group -------------------------------------
    bad = metrics.loc[metrics.performance <= q25, "pct_in_band"]
    mid = metrics.loc[(metrics.performance > q25)
                      & (metrics.performance < q75), "pct_in_band"]
    good = metrics.loc[metrics.performance >= q75, "pct_in_band"]
    data = [bad, mid, good]
    pairs = [(0, 1), (0, 2), (1, 2)]
    padj = holm([ttest_ind(data[a], data[b]).pvalue for a, b in pairs])
    ax_d.bar(range(3), [g.mean() for g in data], yerr=[sem(g) for g in data],
             capsize=4, color=[RED, GREY, GREEN], edgecolor="black",
             linewidth=0.7, width=0.6, zorder=2)
    top = max(g.mean() for g in data) + 6
    for lev, ((a, b), pv) in enumerate(zip(pairs, padj)):
        if pv < 0.05:
            yy = top + lev * 4.5
            ax_d.plot([a, a, b, b], [yy, yy + 1, yy + 1, yy], lw=1.0, color=BLACK)
            ax_d.text((a + b) / 2, yy + 1.3, "**" if pv < 0.01 else "*",
                      ha="center", fontsize=11)
    ax_d.set_xticks(range(3))
    ax_d.set_xticklabels([f"Bad\n($\\leq${q25:.0f} s)", "Middle",
                          f"Good\n($\\geq${q75:.0f} s)"])
    ax_d.set_ylabel("% time within optimal band")
    tidy(ax_d)
    panel(ax_d, "D", x=-0.20)

    save(fig, "figure_6.png")
    print(f"    bad {bad.mean():.1f}%  middle {mid.mean():.1f}%  good {good.mean():.1f}%"
          f"   r = {r:.3f}   ceiling {100*ceiling.mean():.1f}%")


# --------------------------------------------------------------------------- #
# Figure 7: what personalisation can and cannot do
# --------------------------------------------------------------------------- #

def figure_7():
    from scipy.stats import linregress, pearsonr, spearmanr

    folds = pd.read_csv(RES / "loso_control_bands_folds.csv")
    rel = pd.read_csv(RES / "optimum_reliability.csv")
    bw = pd.read_csv(RES / "optimum_reliability_bandwidth.csv")
    params = pd.read_csv(RES / "within_trial_control_params.csv")
    uni = pd.read_csv(RES / "loso_control_bands_universal.csv")
    per = pd.read_csv(RES / "loso_control_bands_personalised.csv")

    fig = plt.figure(figsize=(15, 8.6))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.34)

    # -- A: held-out gain -----------------------------------------------------
    # Every scheme is scored on the same 113 held-out trials by
    # band_scheme_comparison.py, so the bars are directly comparable. The
    # retuned universal band is the benchmark a personalised scheme has to
    # beat, and it carries no per-subject information at all.
    ax = fig.add_subplot(gs[0, 0])
    schemes = pd.read_csv(RES / "band_scheme_comparison.csv")
    short = {"universal x1.0": "Universal $\\times$1.0",
             "universal x1.75 (retuned)": "Universal $\\times$1.75\n(retuned, no\npersonalisation)",
             "group widths (composite score)": "Group widths\n(composite score)",
             "group widths (calibration marker)": "Group widths\n(calibration marker)",
             "centre from fitted curve": "Centre from\nfitted curve",
             "width from fitted curve": "Width from\nfitted curve"}
    order = list(short)
    schemes = schemes.set_index("scheme").loc[order].reset_index()
    cols = [GREY, BLACK, SKY, SKY, BLUE, BLUE]
    ys = np.arange(len(schemes))[::-1]
    ax.barh(ys, schemes.cohens_d, height=0.62, color=cols, edgecolor="black",
            linewidth=0.7, zorder=2)
    for y, row in zip(ys, schemes.itertuples()):
        ax.text(row.cohens_d + 0.03, y, f"d = {row.cohens_d:.2f},  "
                f"r = {row.r:+.3f}", va="center", fontsize=8)
    ax.axvline(schemes.cohens_d.iloc[1], color=BLACK, ls="--", lw=1.0,
               zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([short[s] for s in schemes.scheme], fontsize=8)
    ax.set_xlabel("Cohen's d, good vs bad trials")
    ax.set_xlim(0, 1.55)
    tidy(ax, grid_axis="x")
    panel(ax, "A", x=-0.44)

    # -- B: multiplier stability across folds --------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(folds.subject, folds.m_sens_train, "o-", color=RED, ms=6, lw=1.6,
            label="Sensitive group")
    ax.plot(folds.subject, folds.m_tol_train, "s-", color=BLUE, ms=6, lw=1.6,
            label="Tolerant group")
    ax.set_title("Group widths, composite score", fontsize=9.5, pad=6)
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Selected band multiplier")
    ax.set_ylim(0.2, 3.2)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.text(0.5, 0.02, "sensitive 0.5 to 2.25, tolerant 1.75 to 3.0",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    tidy(ax)
    panel(ax, "B", x=-0.24)

    # -- C: reliability -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    # The residualised rows carry no subset label, so they are selected on
    # ``kind`` alone. Filtering both series on subset silently dropped the
    # residualised bars, which are the point of the panel.
    raw = rel[(rel.kind == "raw") & (rel.subset == "all trials")]
    resid = rel[rel.kind == "residualised"]
    names = ["quadratic vertex", "top-quartile arousal",
             "performance-weighted arousal"]
    short = ["quadratic\nvertex", "top-quartile\narousal", "perf.-weighted\narousal"]
    xs = np.arange(len(names))

    def pick(tab, name, col="r"):
        row = tab[tab.estimator == name]
        return float(row[col].iloc[0]) if len(row) else np.nan

    rr = [pick(raw, n) for n in names]
    rd = [pick(resid, n) for n in names]
    err_r = np.array([[rr[i] - pick(raw, n, "lo"), pick(raw, n, "hi") - rr[i]]
                      for i, n in enumerate(names)]).T
    err_d = np.array([[rd[i] - pick(resid, n, "lo"), pick(resid, n, "hi") - rd[i]]
                      for i, n in enumerate(names)]).T
    bar_kw = dict(width=0.36, edgecolor="black", linewidth=0.7, zorder=2)
    err_kw = dict(fmt="none", ecolor=BLACK, elinewidth=0.9, capsize=2.5, zorder=3)
    ax.bar(xs - 0.19, rr, color=SKY, label="raw", **bar_kw)
    ax.errorbar(xs - 0.19, rr, yerr=np.abs(err_r), **err_kw)
    ax.bar(xs + 0.19, rd, color=ORANGE,
           label="after removing subject mean", **bar_kw)
    ax.errorbar(xs + 0.19, rd, yerr=np.abs(err_d), **err_kw)

    r_bw = pearsonr(bw.half1, bw.half2)[0]
    ax.bar([len(names)], [r_bw], color=RED, **bar_kw)
    for xi, v in zip(list(xs - 0.19) + list(xs + 0.19) + [len(names)],
                     rr + rd + [r_bw]):
        if np.isfinite(v):
            ax.text(xi, v + 0.035 if v >= 0 else v - 0.075, f"{v:+.2f}",
                    ha="center", fontsize=8, color="#333333")
    ax.axhline(0, color=BLACK, lw=1.0)
    ax.set_xticks(list(xs) + [len(names)])
    ax.set_xticklabels(short + ["best band\nwidth"], fontsize=8.5, rotation=22,
                       ha="right")
    ax.set_ylabel("Split-half reliability (r)")
    ax.set_ylim(-0.55, 1.40)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2,
              columnspacing=1.0, handlelength=1.3, borderpad=0.1)
    tidy(ax)
    panel(ax, "C", x=-0.24)

    # -- D: within-trial control parameters ----------------------------------
    ax = fig.add_subplot(gs[1, :2])
    keep = ["above universal band (reference)",
            "above per-subject band (calibration SD)",
            "above ADAPTIVE band (trailing 10 s SD)",
            "sustained above band (>=3 s)",
            "excess over universal band (graded)",
            "excess over ADAPTIVE band (graded)",
            "rate of change of arousal"]
    nice = ["above universal band\n(reference rule)", "above per-subject band",
            "above adaptive band", "sustained above band\n($\\geq$3 s)",
            "graded excess over\nuniversal band", "graded excess over\nadaptive band",
            "rate of change\nof arousal"]
    p = params.set_index("parameter")
    ys = np.arange(len(keep))[::-1]
    for y, k in zip(ys, keep):
        if k not in p.index:
            continue
        row = p.loc[k]
        col = GREEN if "adaptive band (graded)" in k.lower() else (
            GREY if "rate" in k.lower() else BLUE)
        ax.barh(y, row.auc - 50, left=50, height=0.6, color=col,
                edgecolor="black", linewidth=0.7, zorder=2)
        ax.text(row.auc + 0.25, y, f"{row.auc:.1f}", va="center", fontsize=9.5)
    ax.axvline(50, color=BLACK, lw=1.1, ls="--")
    ax.text(50.05, len(keep) - 0.35, "chance", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels(nice, fontsize=9)
    ax.set_xlabel("AUC for crashing within 5 s, stratified by time bin (%)")
    ax.set_xlim(49, 61)
    tidy(ax, grid_axis="x")
    panel(ax, "D", x=-0.20)

    # -- E: does it add over the fixed rule? ---------------------------------
    ax = fig.add_subplot(gs[1, 2])
    added = [("adaptive band", 0.77), ("sustained ($\\geq$3 s)", 0.13),
             ("rate of change", 0.026), ("graded excess over\nadaptive band", 6.6e-4)]
    ys = np.arange(len(added))[::-1]
    for y, (lab, pv) in zip(ys, added):
        col = GREEN if pv < 0.01 else (ORANGE if pv < 0.05 else "#BBBBBB")
        ax.barh(y, -np.log10(pv), height=0.58, color=col, edgecolor="black",
                linewidth=0.7, zorder=2)
        ax.text(-np.log10(pv) + 0.05, y, f"p = {pv:.3g}", va="center", fontsize=9)
    ax.axvline(-np.log10(0.05), color=BLACK, ls="--", lw=1.1)
    ax.set_yticks(ys)
    ax.set_yticklabels([a[0] for a in added], fontsize=9)
    ax.set_xlabel(r"$-\log_{10} p$, added over the fixed band")
    ax.set_xlim(0, 4.4)
    tidy(ax, grid_axis="x")
    panel(ax, "E", x=-0.40)

    save(fig, "figure_7.png")
    rho_bw = spearmanr(bw.half1, bw.half2)[0]
    for row in schemes.itertuples():
        print(f"    {row.scheme:<38} d={row.cohens_d:.2f} r={row.r:+.3f}")
    print(f"    band-width split-half: Pearson r={r_bw:+.3f}, "
          f"Spearman rho={rho_bw:+.3f}")
    print(f"    reliability raw={['%+.3f' % v for v in rr]} "
          f"residualised={['%+.3f' % v for v in rd]}")


BUILDERS = {1: figure_1, 2: figure_2, 3: figure_3, 4: figure_4,
            5: figure_5, 6: figure_6, 7: figure_7}


def main():
    wanted = [int(a) for a in sys.argv[1:]] or sorted(BUILDERS)
    for n in wanted:
        print(f"figure {n}")
        BUILDERS[n]()


if __name__ == "__main__":
    main()
