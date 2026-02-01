from __future__ import annotations

"""
Deterministic QMPC (Quadratic MPC) for the greenhouse using a fitted *linear incremental* state-space model.

Model (per step k):
  Δx_k = M x_k + N u_k + O d_k + m
  x_{k+1} = x_k + Δx_k = (I+M) x_k + N u_k + O d_k + m

Decision variables (k=0..K-1):
  x_k (states), u_k (controls), slack variables for comfort bounds, and L_delta for DLI deficit.

Solver:
  Gurobi (QP with linear constraints).

Inputs:
  - model: used only for vector names/indexing and cost parameter constants (from YAML).
  - fitted_mats: loaded from data/for_fitting_qmpc/fitting_results/fitted_matrices.npz (see run script).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

import gurobipy as gp
from gurobipy import GRB


@dataclass(frozen=True)
class QmpcSolveResult:
    U_opt: np.ndarray  # (K, n_u)
    success: bool
    objective: float
    status: int
    message: str

def modeling_solving_qmpc_deterministic_problem(
     *,
     model: Any,
     x_ini: np.ndarray,
     d_pred_h: np.ndarray,
     L_DLI_star_h: np.ndarray,   # (K,)
     step_index: int,
     mpc_global_parameters: Dict[str, Any],
     fitted_mats: Dict[str, np.ndarray],
     solver_options: Optional[Dict[str, Any]] = None,
     K_eff: int,
) -> QmpcSolveResult:
    """
    Builds and solves the deterministic QMPC QP with Gurobi.

    Notes:
      - This function treats x_0 as fixed to x_ini.
      - All controls are continuous in [0,1] (no binaries).
      - Warm-start is optional; if provided, it is used as variable Start values.
    """

    defs = model.defs
    climate_parameters = model.params

    # -------------------------
    # Dimensions / indices
    # -------------------------
    K = int(K_eff)
    n_x = len(defs.state_names)
    n_u = len(defs.control_names)
    n_d = len(defs.disturbance_names)

    x_ini = np.asarray(x_ini, dtype=float).reshape(n_x)
    d_pred_h = np.asarray(d_pred_h, dtype=float).reshape(K, n_d)

    T_in_idx = defs.state_idx["T_in"]
    H_in_idx = defs.state_idx["H_in"]
    C_in_idx = defs.state_idx["C_in"]
    L_DLI_idx = defs.state_idx["L_DLI"]

    U_fan_idx = defs.control_idx["U_fan"]
    U_pad_idx = defs.control_idx["U_pad"]
    U_heat_idx = defs.control_idx["U_heat"]
    U_LED_idx = defs.control_idx["U_LED"]
    U_hum_idx = defs.control_idx["U_hum"]
    U_dehum_idx = defs.control_idx["U_dehum"]
    U_dos_idx = defs.control_idx["U_dos"]

    # -------------------------
    # MPC parameters (same as NMPC)
    # -------------------------
    Delta_t = float(mpc_global_parameters["Delta_t"])
    steps_per_day = int(mpc_global_parameters["steps_per_day"])
    gamma = float(mpc_global_parameters["gamma"])

    alpha_heat = float(mpc_global_parameters["alpha_heat"])
    alpha_elec = float(mpc_global_parameters["alpha_elec"])
    alpha_water = float(mpc_global_parameters["alpha_water"])
    alpha_hum = float(mpc_global_parameters["alpha_hum"])
    alpha_dehum = float(mpc_global_parameters["alpha_dehum"])
    alpha_dos = float(mpc_global_parameters["alpha_dos"])

    beta_C = float(mpc_global_parameters["beta_C"])
    xi = float(mpc_global_parameters["xi"])
    S_fan = float(mpc_global_parameters["S_fan"])

    lambda_T = float(mpc_global_parameters["lambda_T"])
    lambda_Z = float(mpc_global_parameters["lambda_Z"])
    lambda_L = float(mpc_global_parameters["lambda_L"])
    lambda_B_day = float(mpc_global_parameters["lambda_B"]["day"])
    lambda_B_night = float(mpc_global_parameters["lambda_B"]["night"])

    lambda_T_plus = float(mpc_global_parameters["lambda_T_plus"])
    lambda_T_minus = float(mpc_global_parameters["lambda_T_minus"])
    lambda_Z_plus = float(mpc_global_parameters["lambda_Z_plus"])
    lambda_Z_minus = float(mpc_global_parameters["lambda_Z_minus"])
    lambda_B_plus = float(mpc_global_parameters["lambda_B_plus"])
    lambda_B_minus = float(mpc_global_parameters["lambda_B_minus"])

    T_in_bar_day = float(mpc_global_parameters["T_in_bar"]["day"])
    T_in_under_day = float(mpc_global_parameters["T_in_under"]["day"])
    T_in_bar_night = float(mpc_global_parameters["T_in_bar"]["night"])
    T_in_under_night = float(mpc_global_parameters["T_in_under"]["night"])

    Z_in_bar = float(mpc_global_parameters["Z_in_bar"])
    Z_in_under = float(mpc_global_parameters["Z_in_under"])

    B_in_bar_day = float(mpc_global_parameters["B_in_bar"]["day"])
    B_in_under_day = float(mpc_global_parameters["B_in_under"]["day"])
    B_in_bar_night = float(mpc_global_parameters["B_in_bar"]["night"])
    B_in_under_night = float(mpc_global_parameters["B_in_under"]["night"])

    L_DLI_target = float(mpc_global_parameters["L_DLI_target"])
    ramp_start_step = int(mpc_global_parameters["L_DLI_profile"]["ramp_start_step"])
    ramp_end_step = int(mpc_global_parameters["L_DLI_profile"]["ramp_end_step"])

    H_tilde_sat = float(mpc_global_parameters["H_tilde_sat"])

    # -------------------------
    # Linear model matrices
    # -------------------------
    M = np.asarray(fitted_mats["M"], dtype=float)
    N = np.asarray(fitted_mats["N"], dtype=float)
    O = np.asarray(fitted_mats["O"], dtype=float)
    m = np.asarray(fitted_mats["m"], dtype=float).reshape(-1)

    A = np.eye(n_x) + M  # x_{k+1} = A x_k + N u_k + O d_k + m

    # -------------------------
    # Time-of-day helpers
    # -------------------------
    def kappa_at(k: int) -> int:
        return int((step_index % steps_per_day + k) % steps_per_day)

    def is_day(kappa: int) -> bool:
        return ramp_start_step <= kappa < ramp_end_step

    def bounds_at(k: int) -> Tuple[float, float, float, float]:
        kap = kappa_at(k)
        if is_day(kap):
            return T_in_under_day, T_in_bar_day, B_in_under_day, B_in_bar_day
        return T_in_under_night, T_in_bar_night, B_in_under_night, B_in_bar_night

    def stars_at(k: int) -> Tuple[float, float, float, float]:
        T_lo, T_hi, B_lo, B_hi = bounds_at(k)
        T_star = 0.5 * (T_lo + T_hi)
        Z_star = 0.5 * (Z_in_under + Z_in_bar)
        B_star = 0.5 * (B_lo + B_hi)
        L_star = _L_DLI_star(kappa_at(k), steps_per_day, ramp_start_step, ramp_end_step, L_DLI_target)
        return T_star, Z_star, B_star, L_star

    # -------------------------
    # Gurobi model + variables
    # -------------------------
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    mqp = gp.Model("qmpc_deterministic", env=env)

    if solver_options:
        for k, v in solver_options.items():
            try:
                mqp.setParam(k, v)
            except Exception:
                pass

    X = mqp.addVars(K + 1, n_x, lb=-GRB.INFINITY, name="x")  # x_0..x_K
    U = mqp.addVars(K, n_u, lb=0.0, ub=1.0, name="u")  # u_0..u_{K-1}

    S_T_minus = mqp.addVars(K, lb=0.0, name="S_T_minus")
    S_T_plus = mqp.addVars(K, lb=0.0, name="S_T_plus")
    S_Z_minus = mqp.addVars(K, lb=0.0, name="S_Z_minus")
    S_Z_plus = mqp.addVars(K, lb=0.0, name="S_Z_plus")
    S_B_minus = mqp.addVars(K, lb=0.0, name="S_B_minus")
    S_B_plus = mqp.addVars(K, lb=0.0, name="S_B_plus")

    L_delta = mqp.addVars(K, lb=0.0, name="L_delta")

    # x0 fixed
    for i in range(n_x):
        X[0, i].lb = float(x_ini[i])
        X[0, i].ub = float(x_ini[i])

    # -------------------------
    # Dynamics constraints
    # -------------------------
    for k in range(K):
        dk = d_pred_h[k]
        for i in range(n_x):
            lhs = X[k + 1, i]
            rhs = gp.LinExpr()
            rhs += m[i]
            for j in range(n_x):
                rhs += A[i, j] * X[k, j]
            for j in range(n_u):
                rhs += N[i, j] * U[k, j]
            for j in range(n_d):
                rhs += O[i, j] * float(dk[j])
            mqp.addConstr(lhs == rhs, name=f"dyn[{k},{i}]")

    # -------------------------
    # Stage constraints + objective
    # -------------------------
    obj = gp.QuadExpr()

    for k in range(K):
        disc = float(gamma) ** k

        T_lo, T_hi, B_lo, B_hi = bounds_at(k)
        T_star, Z_star, B_star, L_star = stars_at(k)
        lambda_B = lambda_B_day if is_day(kappa_at(k)) else lambda_B_night

        # convenience views: x_k = X[k,*]
        T = X[k, T_in_idx]
        H = X[k, H_in_idx]
        C = X[k, C_in_idx]
        Ldli = X[k, L_DLI_idx]

        Z = (H_tilde_sat - H) * xi
        B = beta_C * C

        # comfort bounds with slack
        mqp.addConstr(T >= T_lo + S_T_minus[k], name=f"T_lo[{k}]")
        mqp.addConstr(T <= T_hi + S_T_plus[k], name=f"T_hi[{k}]")

        mqp.addConstr(Z >= Z_in_under + S_Z_minus[k], name=f"Z_lo[{k}]")
        mqp.addConstr(Z <= Z_in_bar + S_Z_plus[k], name=f"Z_hi[{k}]")

        mqp.addConstr(B >= B_lo + S_B_minus[k], name=f"B_lo[{k}]")
        mqp.addConstr(B <= B_hi + S_B_plus[k], name=f"B_hi[{k}]")

        # fan >= pad
        mqp.addConstr(U[k, U_fan_idx] >= U[k, U_pad_idx], name=f"fan_ge_pad[{k}]")

        # DLI deficit: L_delta >= L_star - L_DLI
        mqp.addConstr(L_delta[k] >= float(L_star) - Ldli, name=f"L_delta_def[{k}]")

        # resource energy terms (linear in U)
        U_heat = U[k, U_heat_idx]
        U_fan = U[k, U_fan_idx]
        U_pad = U[k, U_pad_idx]
        U_LED = U[k, U_LED_idx]
        U_hum = U[k, U_hum_idx]
        U_dehum = U[k, U_dehum_idx]
        U_dos = U[k, U_dos_idx]

        E_heat = alpha_heat * (climate_parameters.Q_bar_heat * U_heat) * Delta_t / 3.6e6
        E_fan = alpha_elec * (S_fan * climate_parameters.V_bar_fan * U_fan) * Delta_t / 3.6e6
        E_LED = alpha_elec * (climate_parameters.P_bar_LED * U_LED) * Delta_t / 3.6e6

        # Note: pad energy is bilinear in (U_fan * U_pad). Keep deterministic QMPC as QP:
        #   Introduce bilinear? Not allowed in QP. We follow NMPC formulation exactly, so we keep this as quadratic term:
        #   F_pad = const * U_fan * U_pad  => E_pad linear in F_pad => quadratic in (U_fan, U_pad).
        pad_const = (climate_parameters.c_vh * climate_parameters.V_bar_fan * climate_parameters.DeltaT_pad) / climate_parameters.q_evap
        E_pad = alpha_water * (pad_const * U_fan * U_pad) * Delta_t

        E_hum = alpha_hum * (climate_parameters.F_bar_hum * U_hum) * Delta_t
        E_dehum = alpha_dehum * (climate_parameters.F_bar_dehum * U_dehum) * Delta_t
        E_dos = alpha_dos * (climate_parameters.D_bar_dos * U_dos) * Delta_t

        # stage quadratic tracking + slack linear + L_delta quadratic
        stage = gp.QuadExpr()
        stage += E_heat + E_fan + E_LED + E_pad + E_hum + E_dehum + E_dos
        stage += lambda_T * (T - float(T_star)) * (T - float(T_star))
        stage += lambda_Z * (Z - float(Z_star)) * (Z - float(Z_star))
        stage += lambda_B * (B - float(B_star)) * (B - float(B_star))
        stage += lambda_L * L_delta[k] * L_delta[k]

        stage += lambda_T_plus * S_T_plus[k] + lambda_T_minus * S_T_minus[k]
        stage += lambda_Z_plus * S_Z_plus[k] + lambda_Z_minus * S_Z_minus[k]
        stage += lambda_B_plus * S_B_plus[k] + lambda_B_minus * S_B_minus[k]

        obj += disc * stage

    mqp.setObjective(obj, GRB.MINIMIZE)
    mqp.optimize()

    status = int(mqp.Status)
    success = status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)

    if success:
        U_opt = np.array([[float(U[k, j].X) for j in range(n_u)] for k in range(K)], dtype=float)
        obj_val = float(mqp.ObjVal) if mqp.SolCount > 0 else float("nan")
        msg = "optimal" if status == GRB.OPTIMAL else "suboptimal"
    else:
        U_opt = np.zeros((K, n_u), dtype=float)
        obj_val = float("nan")
        msg = f"gurobi_status_{status}"

    return QmpcSolveResult(
        U_opt=U_opt,
        success=success,
        objective=obj_val,
        status=status,
        message=msg,
    )
