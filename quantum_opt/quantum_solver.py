import sys
sys.path.insert(0, '/app')

import os
import json
import time
import pandas as pd
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from collections import defaultdict
from shared.config_loader import load_config
from shared.schemas import SolverResult

def build_and_solve_qubo():
    """
    Constructs and solves the QUBO model for Project ISOBAR.
    
    Discretisation Strategy:
    ------------------------
    Continuous Exposure at Default (EAD) bounds [EAD_min, EAD_max] are mapped onto a set of 
    binary variables using Binary Expansion.
    EAD_i = EAD_min_i + (EAD_max_i - EAD_min_i) / (2^B - 1) * sum(2^k * q_{i,k} for k in 0..B-1)
    
    Why Boundary Constraints are Free:
    ----------------------------------
    By construction, the minimum possible value occurs when all q_{i,k} = 0 (resulting in EAD_min),
    and the maximum possible value occurs when all q_{i,k} = 1 (resulting in EAD_max).
    Therefore, the encoding cannot produce values outside [EAD_min, EAD_max].
    
    QUBO Construction:
    ------------------
    The QUBO matrix Q is constructed from two objective terms:
    1. RWA Targeting (Term A): Penalty for deviating from the target RWA.
       penalty_weight * (sum_i RW_i * EAD_i(q) - RWA_target)^2
    2. Regularisation (Term B): Penalty for deviating from the baseline EAD, scaled by lambda.
       lambda * sum_i ((EAD_i(q) - EAD_baseline_i) / EAD_baseline_i)^2
       
    Variable Naming Convention:
    ---------------------------
    Binary variables are named as `q_{i}_{k}`, where `i` is the asset index and `k` is the bit index.
    
    Solver:
    -------
    This implementation uses `dwave-samplers` (SimulatedAnnealingSampler) because the older 
    `dwave-neal` package is deprecated.
    """
    config = load_config()
    
    # Setup directories
    data_dir = config.output.data_dir
    results_dir = config.output.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(f"{data_dir}/asset_register.csv")
    
    # Extract config parameters
    B = config.optimization.quantum.num_bits
    penalty_weight = config.optimization.quantum.penalty_weight
    lambda_reg = config.optimization.lambda_reg
    target_rwa = config.optimization.target_rwa
    
    start_time = time.perf_counter()
    
    Q = defaultdict(float)
    
    # STEP 1 & 2 - Discretisation & QUBO Construction
    
    # Pre-calculate step sizes and offsets
    step_sizes = {}
    offsets = {}
    for i, row in df.iterrows():
        step_sizes[i] = (row['ead_max_limit'] - row['ead_min_limit']) / ((1 << B) - 1)
        offsets[i] = row['ead_min_limit']
        
    # --- Term A: RWA Targeting ---
    # a_i = RW_i * step_i
    a = {i: row['precalc_rw'] * step_sizes[i] for i, row in df.iterrows()}
    # offset_A = sum_i RW_i * EAD_min_i - RWA_target
    offset_A = sum(row['precalc_rw'] * row['ead_min_limit'] for _, row in df.iterrows()) - target_rwa
    
    # Expand (sum_{i,k} a_i * 2^k * q_{i,k} + offset_A)^2
    # = sum_{i,k} (a_i * 2^k)^2 * q_{i,k} + 2 * offset_A * sum_{i,k} a_i * 2^k * q_{i,k}
    #   + 2 * sum_{(i,k) < (j,l)} (a_i * 2^k) * (a_j * 2^l) * q_{i,k} * q_{j,l}
    
    # Linear and diagonal quadratic terms for Term A
    for i in df.index:
        for k in range(B):
            var_ik = f"q_{i}_{k}"
            coeff_ik = a[i] * (1 << k)
            # Linear part from 2 * offset_A * term + diagonal quadratic since q^2 = q
            Q[(var_ik, var_ik)] += penalty_weight * (coeff_ik**2 + 2 * offset_A * coeff_ik)
            
            # Cross terms within same asset
            for l in range(k + 1, B):
                var_il = f"q_{i}_{l}"
                coeff_il = a[i] * (1 << l)
                Q[(var_ik, var_il)] += penalty_weight * 2 * coeff_ik * coeff_il
                
    # Cross terms between different assets
    indices = list(df.index)
    for idx, i in enumerate(indices):
        for j in indices[idx+1:]:
            for k in range(B):
                var_ik = f"q_{i}_{k}"
                coeff_ik = a[i] * (1 << k)
                for l in range(B):
                    var_jl = f"q_{j}_{l}"
                    coeff_jl = a[j] * (1 << l)
                    # For Q matrix, maintaining order to avoid duplicates (could also just use both (u,v) and (v,u))
                    # dimod handles symmetric/upper triangular, we use standard upper triangular
                    u, v = min(var_ik, var_jl), max(var_ik, var_jl)
                    Q[(u, v)] += penalty_weight * 2 * coeff_ik * coeff_jl

    # --- Term B: Regularisation ---
    for i, row in df.iterrows():
        baseline = row['ead_baseline']
        if baseline == 0:
            continue
            
        c_i = step_sizes[i] / baseline
        d_i = (row['ead_min_limit'] - baseline) / baseline
        
        # lambda * (sum_k c_i * 2^k * q_{i,k} + d_i)^2
        for k in range(B):
            var_ik = f"q_{i}_{k}"
            coeff_ik = c_i * (1 << k)
            
            Q[(var_ik, var_ik)] += lambda_reg * (coeff_ik**2 + 2 * d_i * coeff_ik)
            
            for l in range(k + 1, B):
                var_il = f"q_{i}_{l}"
                coeff_il = c_i * (1 << l)
                Q[(var_ik, var_il)] += lambda_reg * 2 * coeff_ik * coeff_il

    # STEP 3 - Solve
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(
        bqm, 
        num_reads=config.optimization.quantum.num_reads,
        num_sweeps=config.optimization.quantum.num_sweeps,
        seed=config.portfolio.seed
    )
    
    # STEP 4 - Decode
    best_sample = sampleset.first.sample
    
    optimized_ead = []
    achieved_rwa = 0.0
    disruption_score = 0.0
    
    for i, row in df.iterrows():
        binary_val = sum(int(best_sample.get(f"q_{i}_{k}", 0)) * (1 << k) for k in range(B))
        ead_val = row['ead_min_limit'] + step_sizes[i] * binary_val
        optimized_ead.append(ead_val)
        
        achieved_rwa += ead_val * row['precalc_rw']
        baseline = row['ead_baseline']
        if baseline > 0:
            disruption_score += ((ead_val - baseline) / baseline) ** 2
            
    df['optimized_ead'] = optimized_ead
    df.to_csv(f"{results_dir}/quantum_optimized_assets.csv", index=False)
    
    end_time = time.perf_counter()
    solve_time = end_time - start_time
    
    rwa_delta_pct = abs(achieved_rwa - target_rwa) / target_rwa * 100
    
    # Save result
    result = SolverResult(
        solver_type="quantum",
        target_rwa=target_rwa,
        achieved_rwa=achieved_rwa,
        rwa_delta_pct=rwa_delta_pct,
        lambda_reg=lambda_reg,
        disruption_score=disruption_score,
        wall_clock_seconds=solve_time,
        converged=True,  # SA always produces a result
        ead_vector=optimized_ead,
        num_bits=B,
        best_energy=float(sampleset.first.energy),
        num_reads=config.optimization.quantum.num_reads,
    )
    
    with open(f"{results_dir}/quantum_result.json", "w") as f:
        f.write(result.model_dump_json(indent=2))
        
    print("Quantum Solver Complete!")
    print(f"Time: {solve_time:.2f}s")
    print(f"Target RWA          : £{target_rwa:,.2f}")
    print(f"Achieved RWA        : £{achieved_rwa:,.2f}")
    print(f"RWA Delta %         : {rwa_delta_pct:.4f}%")
    print(f"Disruption Score    : {disruption_score:.6f}")
    print(f"Best QUBO Energy    : {sampleset.first.energy:.4f}")
    print(f"Binary Variables    : {len(best_sample)}")
    print(f"Results Saved To    : {results_dir}/quantum_result.json")

if __name__ == "__main__":
    build_and_solve_qubo()
