"""
Data Schemas
============

Pydantic models for validating the synthetic asset register and solver results.
These ensure data integrity at service boundaries (datagen → solver → notebook).

Schema matches RFC § 3 exactly.
"""

from pydantic import BaseModel, Field, field_validator


class AssetRecord(BaseModel):
    """Single row of the synthetic asset register (RFC § 3)."""

    asset_id: str = Field(description="UUID — unique identifier for the bucket")
    asset_class: str = Field(description="e.g. 'Corporate', 'SME', 'Retail-Mortgage'")
    rating_grade: str = Field(description="e.g. 'AAA' through 'CCC'")
    pd: float = Field(gt=0, lt=1, description="Probability of Default")
    lgd: float = Field(gt=0, lt=1, description="Loss Given Default")
    ead_baseline: float = Field(gt=0, description="Current outstanding exposure")
    ead_min_limit: float = Field(gt=0, description="Policy floor for origination")
    ead_max_limit: float = Field(gt=0, description="Policy ceiling for origination")
    maturity_years: float = Field(gt=0, le=5, description="Effective maturity")
    precalc_rw: float = Field(ge=0, description="Pre-calculated IRB Risk Weight")

    @field_validator("asset_class")
    @classmethod
    def validate_asset_class(cls, v: str) -> str:
        allowed = {"Corporate", "SME", "Retail-Mortgage"}
        if v not in allowed:
            raise ValueError(f"asset_class must be one of {allowed}, got '{v}'")
        return v

    @field_validator("rating_grade")
    @classmethod
    def validate_rating_grade(cls, v: str) -> str:
        allowed = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC"}
        if v not in allowed:
            raise ValueError(f"rating_grade must be one of {allowed}, got '{v}'")
        return v

    @field_validator("ead_max_limit")
    @classmethod
    def validate_ead_limits(cls, v: float, info) -> float:
        if "ead_min_limit" in info.data and v < info.data["ead_min_limit"]:
            raise ValueError("ead_max_limit must be >= ead_min_limit")
        return v


class SolverResult(BaseModel):
    """Output schema for both classical and quantum solvers."""

    solver_type: str = Field(description="'classical' or 'quantum'")
    target_rwa: float = Field(description="Requested target RWA")
    achieved_rwa: float = Field(description="Actual RWA of the solution")
    rwa_delta_pct: float = Field(description="Percentage deviation from target")
    lambda_reg: float = Field(description="Regularisation weight used")
    disruption_score: float = Field(
        description="Sum of squared normalised EAD deviations from baseline"
    )
    wall_clock_seconds: float = Field(description="Solver wall-clock time")
    converged: bool = Field(description="Whether the solver converged")
    ead_vector: list[float] = Field(description="Solved EAD values per asset")

    # Quantum-specific fields (None for classical)
    num_bits: int | None = Field(default=None, description="QUBO discretisation bits")
    best_energy: float | None = Field(default=None, description="Best QUBO energy")
    num_reads: int | None = Field(default=None, description="SA num_reads used")
