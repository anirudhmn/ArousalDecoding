"""Yerkes-Dodson validation: quadratic mixed-effects models of performance."""

from __future__ import annotations

import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm

QUAD = "I(arousal ** 2)"
QUAD_HARD = "C(difficulty)[T.1]:I(arousal ** 2)"
LIN_HARD = "C(difficulty)[T.1]:arousal"

FORMULA_QUADRATIC = ("performance ~ C(difficulty) * arousal "
                     "+ C(difficulty) * I(arousal**2) + C(condition)")
FORMULA_LINEAR = "performance ~ C(difficulty) * arousal + C(condition)"


def fit_yd_model(df, arousal_col="arousal", formula=FORMULA_QUADRATIC):
    """Fit performance ~ arousal + arousal^2, by difficulty, with subject RE.

    Fitted by maximum likelihood (``reml=False``) so that AIC is comparable
    across models with different fixed-effects structures.
    """
    d = df.copy()
    d["arousal"] = d[arousal_col]
    return smf.mixedlm(formula, data=d, groups=d["subject"]).fit(reml=False)


def fit_interaction_model(df, feature):
    """Add a baseline-physiology interaction with the linear and quadratic terms."""
    formula = (
        "performance ~ C(difficulty)*arousal + C(difficulty)*I(arousal**2)"
        f" + {feature}_z + arousal:{feature}_z + I(arousal**2):{feature}_z"
    )
    return smf.mixedlm(formula, data=df, groups=df["subject"],
                       re_formula="~1").fit(reml=False)


def quadratic_pvalues(model):
    """p-value of the quadratic term for each difficulty level.

    For the hard course the term is the sum of the base and interaction
    coefficients, so its variance includes the covariance between them.
    """
    p, cov = model.params, model.cov_params()
    b = p[QUAD] + p[QUAD_HARD]
    var = (cov.loc[QUAD, QUAD] + cov.loc[QUAD_HARD, QUAD_HARD]
           + 2 * cov.loc[QUAD, QUAD_HARD])
    z = b / np.sqrt(var)
    return {0: model.pvalues[QUAD], 1: 2 * (1 - norm.cdf(abs(z)))}


def optimum_with_ci(model, difficulty=0):
    """Arousal at peak performance, -b1 / 2*b2, with a delta-method 95% CI."""
    p, cov = model.params, model.cov_params()
    if difficulty == 0:
        b1, b2 = p["arousal"], p[QUAD]
        c11 = cov.loc["arousal", "arousal"]
        c22 = cov.loc[QUAD, QUAD]
        c12 = cov.loc["arousal", QUAD]
    else:
        b1 = p["arousal"] + p[LIN_HARD]
        b2 = p[QUAD] + p[QUAD_HARD]
        c11 = (cov.loc["arousal", "arousal"] + cov.loc[LIN_HARD, LIN_HARD]
               + 2 * cov.loc["arousal", LIN_HARD])
        c22 = (cov.loc[QUAD, QUAD] + cov.loc[QUAD_HARD, QUAD_HARD]
               + 2 * cov.loc[QUAD, QUAD_HARD])
        c12 = (cov.loc["arousal", QUAD] + cov.loc[LIN_HARD, QUAD]
               + cov.loc["arousal", QUAD_HARD] + cov.loc[LIN_HARD, QUAD_HARD])

    opt = -b1 / (2 * b2)
    d1, d2 = -1 / (2 * b2), b1 / (2 * b2 ** 2)
    se = np.sqrt(d1 ** 2 * c11 + d2 ** 2 * c22 + 2 * d1 * d2 * c12)
    return opt, (opt - 1.96 * se, opt + 1.96 * se)


def predict_curve(model, arousal_grid):
    """Fitted curve and 95% CI per difficulty, at the reference condition."""
    p, cov = model.params, model.cov_params()
    preds = {}
    for d in (0, 1):
        ar, ar2 = arousal_grid, arousal_grid ** 2
        mean = p["Intercept"] + p["arousal"] * ar + p[QUAD] * ar2
        cols = ["Intercept", "arousal", QUAD]
        X = np.column_stack([np.ones(len(ar)), ar, ar2])
        if d == 1:
            mean = mean + (p["C(difficulty)[T.1]"] + p[LIN_HARD] * ar
                           + p[QUAD_HARD] * ar2)
            cols += ["C(difficulty)[T.1]", LIN_HARD, QUAD_HARD]
            X = np.column_stack([X, np.ones(len(ar)), ar, ar2])

        se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov.loc[cols, cols].values, X))
        preds[d] = {"mean": mean, "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se}
    return preds


def coefficient_table(model, terms=("arousal", QUAD, LIN_HARD, QUAD_HARD)):
    """Coefficients, SEs, 95% CIs and p-values for the terms of interest."""
    import pandas as pd
    ci = model.conf_int()
    rows = []
    for t in terms:
        if t not in model.params.index:
            continue
        rows.append({
            "term": t,
            "coef": model.params[t],
            "se": model.bse[t],
            "ci_lo": ci.loc[t, 0],
            "ci_hi": ci.loc[t, 1],
            "p": model.pvalues[t],
        })
    return pd.DataFrame(rows)
