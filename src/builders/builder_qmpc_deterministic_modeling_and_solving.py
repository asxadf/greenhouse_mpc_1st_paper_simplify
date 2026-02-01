# src/builders/builder_qmpc_deterministic_modeling_and_solving.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB


@dataclass(frozen=True)
class AffineDynamicsMatrices:
    M: np.ndarray
    N: np.ndarray
    O: np.ndarray
    m: np.ndarray


@dataclass(frozen=True)
class QmpcSolveResult:
    U_opt: np.ndarray
    u0: np.ndarray
    success: bool
    objective: float
    status: int
    message: str
    solve_time_s: Optional[float]


def modeling_solving_qmpc_deterministic_problem(
    *,
    model: Any,
    x_ini: np.ndarray,
    d_pred_h: np.ndarray,
    step_index: int,
    K: int,
    mpc_global_parameters: Dict[str, Any],
    affine: AffineDynamicsMatrices,
    solver_options: Optional[Dict[str, Any]] = None,
) -> QmpcSolveResult:
    """This module only builds and solves the QP; rollout is handled by the runner."""
    defs = model.defs
    params = model.params

    num_x = len(defs.state_names)
    num_u = len(defs.control_names)

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

    K = int(K)
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

    d_pred_h = np.asarray(d_pred_h, dtype=float)
    if d_pred_h.shape != (K, len(defs.disturbance_names)):
        raise ValueError("d_pred_h must be shaped (K, n_d)")

    A = np.asarray(affine.M, dtype=float)
    N = np.asarray(affine.N, dtype=float)
    O = np.asarray(affine.O, dtype=float)
    m = np.asarray(affine.m, dtype=float).reshape(num_x)

    def kappa_at(k: int) -> int:
        return int((step_index % steps_per_day + k) % steps_per_day)

    def is_day(kappa: int) -> bool:
        return ramp_start_step <= kappa < ramp_end_step

    def bounds_at(k: int) -> Tuple[float, float, float, float, float]:
        kap = kappa_at(k)
        if is_day(kap):
            return T_in_under_day, T_in_bar_day, B_in_under_day, B_in_bar_day, lambda_B_day
        return T_in_under_night, T_in_bar_night, B_in_under_night, B_in_bar_night, lambda_B_night

    solver_options = solver_options or {}

    mpc = gp.Model("qmpc_deterministic")
    mpc.Params.OutputFlag = int(solver_options.get("OutputFlag", 0))
    if "TimeLimit" in solver_options:
        mpc.Params.TimeLimit = float(solver_options["TimeLimit"])

    x = mpc.addVars(K + 1, num_x, lb=-GRB.INFINITY, name="x")
    u = mpc.addVars(K, num_u, lb=0.0, ub=1.0, name="u")

    S_T_minus = mpc.addVars(K, lb=0.0, name="S_T_minus")
    S_T_plus = mpc.addVars(K, lb=0.0, name="S_T_plus")
    S_Z_minus = mpc.addVars(K, lb=0.0, name="S_Z_minus")
    S_Z_plus = mpc.addVars(K, lb=0.0, name="S_Z_plus")
    S_B_minus = mpc.addVars(K, lb=0.0, name="S_B_minus")
    S_B_plus = mpc.addVars(K, lb=0.0, name="S_B_plus")
    L_delta = mpc.addVars(K, lb=0.0, name="L_delta")

    Z_in = mpc.addVars(K, lb=-GRB.INFINITY, name="Z_in")
    B_in = mpc.addVars(K, lb=-GRB.INFINITY, name="B_in")

    for i in range(num_x):
        mpc.addConstr(x[0, i] == float(x_ini[i]), name=f"x0_{i}")

    obj = gp.QuadExpr()

    for k in range(K):
        for i in range(num_x):
            expr = gp.quicksum(A[i, j] * x[k, j] for j in range(num_x))
            expr += gp.quicksum(N[i, j] * u[k, j] for j in range(num_u))
            expr += gp.quicksum(O[i, j] * float(d_pred_h[k, j]) for j in range(len(defs.disturbance_names)))
            expr += float(m[i])
            mpc.addConstr(x[k + 1, i] == expr, name=f"dyn_{k}_{i}")

        mpc.addConstr(Z_in[k] == (H_tilde_sat - x[k, H_in_idx]) * xi, name=f"Z_in_{k}")
        mpc.addConstr(B_in[k] == beta_C * x[k, C_in_idx], name=f"B_in_{k}")

        T_in_under, T_in_bar, B_in_under, B_in_bar, lambda_B = bounds_at(k)
        T_star = 0.5 * (T_in_under + T_in_bar)
        Z_star = 0.5 * (Z_in_under + Z_in_bar)
        B_star = 0.5 * (B_in_under + B_in_bar)
        if kappa_at(k) < ramp_start_step:
            L_star = 0.0
        elif kappa_at(k) < ramp_end_step:
            L_star = float(L_DLI_target) * float(kappa_at(k) - ramp_start_step) / float(
                ramp_end_step - ramp_start_step
            )
        else:
            L_star = float(L_DLI_target)

        mpc.addConstr(x[k, T_in_idx] >= T_in_under - S_T_minus[k], name=f"T_under_{k}")
        mpc.addConstr(x[k, T_in_idx] <= T_in_bar + S_T_plus[k], name=f"T_bar_{k}")

        mpc.addConstr(Z_in[k] >= Z_in_under - S_Z_minus[k], name=f"Z_under_{k}")
        mpc.addConstr(Z_in[k] <= Z_in_bar + S_Z_plus[k], name=f"Z_bar_{k}")

        mpc.addConstr(B_in[k] >= B_in_under - S_B_minus[k], name=f"B_under_{k}")
        mpc.addConstr(B_in[k] <= B_in_bar + S_B_plus[k], name=f"B_bar_{k}")

        mpc.addConstr(L_delta[k] >= L_star - x[k, L_DLI_idx], name=f"L_delta_{k}")

        mpc.addConstr(u[k, U_fan_idx] >= u[k, U_pad_idx], name=f"fan_pad_{k}")

        E_heat = alpha_heat * params.Q_bar_heat * u[k, U_heat_idx] * Delta_t / 3.6e6
        E_fan = alpha_elec * S_fan * params.V_bar_fan * u[k, U_fan_idx] * Delta_t / 3.6e6
        E_LED = alpha_elec * params.P_bar_LED * u[k, U_LED_idx] * Delta_t / 3.6e6
        E_pad = alpha_water * u[k, U_pad_idx] * Delta_t
        E_hum = alpha_hum * params.F_bar_hum * u[k, U_hum_idx] * Delta_t
        E_dehum = alpha_dehum * params.F_bar_dehum * u[k, U_dehum_idx] * Delta_t
        E_dos = alpha_dos * params.D_bar_dos * u[k, U_dos_idx] * Delta_t

        stage_cost = E_heat + E_fan + E_LED + E_pad + E_hum + E_dehum + E_dos
        stage_cost += lambda_T * (x[k, T_in_idx] - T_star) * (x[k, T_in_idx] - T_star)
        stage_cost += lambda_Z * (Z_in[k] - Z_star) * (Z_in[k] - Z_star)
        stage_cost += lambda_B * (B_in[k] - B_star) * (B_in[k] - B_star)
        stage_cost += lambda_L * L_delta[k] * L_delta[k]
        stage_cost += lambda_T_plus * S_T_plus[k] + lambda_T_minus * S_T_minus[k]
        stage_cost += lambda_Z_plus * S_Z_plus[k] + lambda_Z_minus * S_Z_minus[k]
        stage_cost += lambda_B_plus * S_B_plus[k] + lambda_B_minus * S_B_minus[k]

        obj += (gamma ** k) * stage_cost

    mpc.setObjective(obj, GRB.MINIMIZE)

    # mpc.write("/Users/xianbangchen/Desktop/xxx/qmpc.lp")

    mpc.optimize()

    status = int(mpc.Status)
    success = status == GRB.OPTIMAL
    solve_time_s = float(mpc.Runtime) if hasattr(mpc, "Runtime") else None

    if success:
        U_opt = np.array([[u[k, j].X for j in range(num_u)] for k in range(K)], dtype=float)
        u0 = U_opt[0]
        objective = float(mpc.ObjVal)
        message = "Solve_Succeeded"
    else:
        U_opt = np.zeros((K, num_u), dtype=float)
        u0 = U_opt[0]
        objective = float("nan")
        message = "Solve_Failed"

    return QmpcSolveResult(
        U_opt=U_opt,
        u0=u0,
        success=success,
        objective=objective,
        status=status,
        message=message,
        solve_time_s=solve_time_s,
    )
