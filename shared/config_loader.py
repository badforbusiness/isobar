"""
Configuration Loader
====================

Loads ``config.yaml`` and exposes it as typed dataclasses so every service
gets validated, predictable access to the same parameters.

Usage::

    from shared.config_loader import load_config
    cfg = load_config()           # loads /app/config.yaml by default
    print(cfg.portfolio.n_assets) # 50
    print(cfg.optimization.target_rwa)  # 500_000_000
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Dataclass hierarchy mirrors config.yaml structure
# ---------------------------------------------------------------------------

@dataclass
class AssetClassConfig:
    name: str
    weight: float
    pd_range: tuple[float, float]
    lgd_range: tuple[float, float]
    eligible_grades: list[str]
    ead_baseline_range: tuple[float, float]


@dataclass
class PortfolioConfig:
    n_assets: int
    seed: int
    rating_pd_map: dict[str, float]
    asset_classes: list[AssetClassConfig]
    maturity_range: tuple[float, float]
    ead_min_factor: float
    ead_max_factor: float


@dataclass
class ClassicalConfig:
    method: str
    maxiter: int
    ftol: float


@dataclass
class QuantumConfig:
    num_bits: int
    num_reads: int
    num_sweeps: int
    penalty_weight: float


@dataclass
class ConcentrationLimit:
    asset_class: str
    max_share: float


@dataclass
class ConcentrationConfig:
    enabled: bool
    limits: list[ConcentrationLimit]


@dataclass
class OptimizationConfig:
    target_rwa: float
    lambda_reg: float
    classical: ClassicalConfig
    quantum: QuantumConfig
    concentration_constraints: ConcentrationConfig


@dataclass
class OutputConfig:
    format: str
    results_dir: str
    data_dir: str


@dataclass
class IsobarConfig:
    portfolio: PortfolioConfig
    optimization: OptimizationConfig
    output: OutputConfig


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_asset_class(raw: dict[str, Any]) -> AssetClassConfig:
    return AssetClassConfig(
        name=raw["name"],
        weight=raw["weight"],
        pd_range=tuple(raw["pd_range"]),
        lgd_range=tuple(raw["lgd_range"]),
        eligible_grades=raw.get("eligible_grades", []),
        ead_baseline_range=tuple(raw["ead_baseline_range"]),
    )


def _parse_portfolio(raw: dict[str, Any]) -> PortfolioConfig:
    return PortfolioConfig(
        n_assets=int(raw["n_assets"]),
        seed=int(raw["seed"]),
        rating_pd_map=raw["rating_pd_map"],
        asset_classes=[_parse_asset_class(ac) for ac in raw["asset_classes"]],
        maturity_range=tuple(raw["maturity_range"]),
        ead_min_factor=float(raw["ead_min_factor"]),
        ead_max_factor=float(raw["ead_max_factor"]),
    )


def _parse_concentration(raw: dict[str, Any]) -> ConcentrationConfig:
    limits = []
    for lim in raw.get("limits", []) or []:
        limits.append(ConcentrationLimit(
            asset_class=lim["asset_class"],
            max_share=float(lim["max_share"]),
        ))
    return ConcentrationConfig(
        enabled=bool(raw.get("enabled", False)),
        limits=limits,
    )


def _parse_optimization(raw: dict[str, Any]) -> OptimizationConfig:
    return OptimizationConfig(
        target_rwa=float(raw["target_rwa"]),
        lambda_reg=float(raw["lambda_reg"]),
        classical=ClassicalConfig(
            method=raw["classical"]["method"],
            maxiter=int(raw["classical"]["maxiter"]),
            ftol=float(raw["classical"]["ftol"]),
        ),
        quantum=QuantumConfig(
            num_bits=int(raw["quantum"]["num_bits"]),
            num_reads=int(raw["quantum"]["num_reads"]),
            num_sweeps=int(raw["quantum"]["num_sweeps"]),
            penalty_weight=float(raw["quantum"]["penalty_weight"]),
        ),
        concentration_constraints=_parse_concentration(
            raw.get("concentration_constraints", {"enabled": False, "limits": []})
        ),
    )


def _parse_output(raw: dict[str, Any]) -> OutputConfig:
    return OutputConfig(
        format=raw.get("format", "csv"),
        results_dir=raw.get("results_dir", "/data/results"),
        data_dir=raw.get("data_dir", "/data/assets"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> IsobarConfig:
    """
    Load and parse the ISOBAR configuration file.

    Parameters
    ----------
    path : str or Path, optional
        Path to ``config.yaml``. Defaults to ``/app/config.yaml`` (the
        Docker mount point) or ``ISOBAR_CONFIG`` environment variable.

    Returns
    -------
    IsobarConfig
        Fully parsed and typed configuration.
    """
    if path is None:
        path = os.environ.get("ISOBAR_CONFIG", "/app/config.yaml")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return IsobarConfig(
        portfolio=_parse_portfolio(raw["portfolio"]),
        optimization=_parse_optimization(raw["optimization"]),
        output=_parse_output(raw["output"]),
    )
