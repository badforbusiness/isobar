"""
Basel IRB Risk Weight Calculation
=================================

Implements the Internal Ratings-Based (IRB) Advanced approach for computing
regulatory risk weights, as specified in:

    Basel II: International Convergence of Capital Measurement and Capital
    Standards, Annex 4 (2006), with Basel III revisions.

The core model is the Asymptotic Single Risk Factor (ASRF) / Vasicek model:

    K = [LGD × Φ((Φ⁻¹(PD) + √R × Φ⁻¹(0.999)) / √(1-R)) - PD × LGD]
        × (1 + (M - 2.5) × b) / (1 - 1.5 × b)

    RiskWeight = K × 12.5

Where:
    - PD:  Probability of Default
    - LGD: Loss Given Default
    - R:   Asset correlation (function of PD, varies by asset class)
    - M:   Effective maturity in years
    - b:   Maturity adjustment factor = (0.11852 - 0.05478 × ln(PD))²
    - Φ:   Standard normal CDF
    - Φ⁻¹: Standard normal inverse CDF (quantile function)

Design Decision (documented in implementation_plan.md):
    The `precalc_rw` column in the asset register uses these formulas.
    Both the classical and quantum solvers consume this pre-calculated value,
    keeping the optimisation objective simple and the IRB physics isolated here.
"""

import numpy as np
from scipy.stats import norm


def asset_correlation(pd: float, asset_class: str) -> float:
    """
    Compute the Basel IRB asset correlation R as a function of PD.

    The correlation formula differs by asset class to reflect the empirical
    observation that higher-quality (lower-PD) borrowers are more exposed
    to systematic risk (macro factors) than lower-quality borrowers.

    Parameters
    ----------
    pd : float
        Probability of Default, in (0, 1).
    asset_class : str
        One of 'Corporate', 'SME', 'Retail-Mortgage'.

    Returns
    -------
    float
        Asset correlation R, in (0, 1).

    Formulae
    --------
    Corporate (Basel II §272):
        R = 0.12 × (1 - e^(-50×PD)) / (1 - e^(-50))
          + 0.24 × (1 - (1 - e^(-50×PD)) / (1 - e^(-50)))

    SME (Basel II §273):
        Same as Corporate but with a firm-size adjustment that reduces
        correlation. We apply the standard 0.04 reduction as a simplification
        (assumes mid-range SME turnover).

    Retail-Mortgage (Basel II §328):
        Fixed R = 0.15 (Basel specifies a flat correlation for residential
        mortgages, reflecting the lower systematic risk sensitivity of
        household credit).
    """
    if asset_class == "Retail-Mortgage":
        return 0.15

    # Corporate correlation formula
    exp_term = (1.0 - np.exp(-50.0 * pd)) / (1.0 - np.exp(-50.0))
    r = 0.12 * exp_term + 0.24 * (1.0 - exp_term)

    if asset_class == "SME":
        # SME firm-size adjustment: reduce correlation by up to 0.04
        # Full formula: R_sme = R_corp - 0.04 × (1 - (S-5)/45) where S = turnover in €M
        # We use the midpoint simplification (S ≈ 25M → adjustment ≈ 0.04 × 0.556)
        r -= 0.04 * (1.0 - (25.0 - 5.0) / 45.0)

    return r


def maturity_adjustment(pd: float, maturity: float) -> float:
    """
    Compute the Basel maturity adjustment factor.

    The maturity adjustment accounts for the fact that longer-maturity
    exposures have more time for credit quality to deteriorate, increasing
    unexpected loss relative to the 1-year horizon of the base formula.

    Parameters
    ----------
    pd : float
        Probability of Default.
    maturity : float
        Effective maturity in years. Basel floors this at 1.0 and caps at 5.0.

    Returns
    -------
    float
        Maturity adjustment multiplier.

    Formula
    -------
        b = (0.11852 - 0.05478 × ln(PD))²
        MA = (1 + (M - 2.5) × b) / (1 - 1.5 × b)
    """
    # Clamp maturity to Basel limits
    m = np.clip(maturity, 1.0, 5.0)

    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    return (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b)


def irb_capital_requirement(pd: float, lgd: float, maturity: float,
                            asset_class: str) -> float:
    """
    Compute the IRB capital requirement K for a single exposure.

    Parameters
    ----------
    pd : float
        Probability of Default, in (0, 1). Must be > 0.
    lgd : float
        Loss Given Default, in (0, 1).
    maturity : float
        Effective maturity in years.
    asset_class : str
        One of 'Corporate', 'SME', 'Retail-Mortgage'.

    Returns
    -------
    float
        Capital requirement K (as a fraction of EAD).
    """
    # Clamp PD to avoid log(0) and ensure meaningful calculation
    pd = np.clip(pd, 1e-6, 0.999)

    r = asset_correlation(pd, asset_class)

    # Vasicek conditional default probability at 99.9% confidence
    g_pd = norm.ppf(pd)
    g_999 = norm.ppf(0.999)
    conditional_pd = norm.cdf(
        (g_pd + np.sqrt(r) * g_999) / np.sqrt(1.0 - r)
    )

    # Base capital charge (unexpected loss only — expected loss PD×LGD is subtracted)
    k_base = lgd * conditional_pd - pd * lgd

    # Apply maturity adjustment (not applied to Retail-Mortgage per Basel II §328)
    if asset_class == "Retail-Mortgage":
        return max(k_base, 0.0)
    else:
        ma = maturity_adjustment(pd, maturity)
        return max(k_base * ma, 0.0)


def irb_risk_weight(pd: float, lgd: float, maturity: float,
                    asset_class: str) -> float:
    """
    Compute the full IRB Risk Weight for a single exposure.

    This is the value stored in the `precalc_rw` column of the asset register
    and used directly in the optimisation objective:

        RWA_i = EAD_i × RiskWeight_i

    Parameters
    ----------
    pd : float
        Probability of Default.
    lgd : float
        Loss Given Default.
    maturity : float
        Effective maturity in years.
    asset_class : str
        One of 'Corporate', 'SME', 'Retail-Mortgage'.

    Returns
    -------
    float
        Risk Weight (dimensionless multiplier). Multiply by EAD to get RWA.

    Formula
    -------
        RiskWeight = K × 12.5

    The 12.5 multiplier is the reciprocal of the 8% minimum capital ratio
    (1/0.08 = 12.5), converting the capital requirement fraction into a
    risk-weight percentage compatible with the RWA framework.
    """
    k = irb_capital_requirement(pd, lgd, maturity, asset_class)
    return k * 12.5
