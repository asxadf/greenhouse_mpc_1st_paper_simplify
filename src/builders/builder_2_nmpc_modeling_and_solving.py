# src/builders/builder_2_nmpc_modeling_and_solving.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from cyipopt import minimize_ipopt


@dataclass(frozen=True)
class NmpcSolveResult:
    U_opt: np.ndarray  # (K, n_u)
    success: bool
    objective: float
    status: int
    message: str
    iters: Optional[int]


def modeling_solving_nmpc_problem(
    *,
    model: Any,
    x_ini: np.ndarray,
    d_pred_h: np.ndarray,                    # (K, n_d)
    step_index: int,
    mpc_global_parameters: Dict[str, Any],   # variable_and_parameter_keeper.yaml::mpc_global_parameters
    solver_options: Dict[str, Any],          # setting_nmpc.yaml::solver::options
    K_eff: int,                              # <-- NEW
    fd_eps: float,  # <-- NEW (from YAML)
    u_warm_start: Optional[np.ndarray] = None,  # (K, num_u)
) -> NmpcSolveResult:
    # =========================
    # [Block 0] Housekeeping
    # =========================

    defs = model.defs # name/index maps (state/control/disturbance)
    climate_parameters = model.params # model physical constants (from YAML)

    # =========================
    # [Block 1] Dimensions + indices (from model.defs)
    # =========================
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

    # =========================
    # [Block 2] Load MPC parameters
    # =========================
    K = int(K_eff)
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

    # =========================
    # [Block 3] Time-of-day helpers (kappa, bounds, references)
    # =========================
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
        T_in_under, T_in_bar, B_in_under, B_in_bar = bounds_at(k)
        T_in_star = 0.5 * (T_in_under + T_in_bar)
        Z_in_star = 0.5 * (Z_in_under + Z_in_bar)
        B_in_star = 0.5 * (B_in_under + B_in_bar)
        L_DLI_star = _L_DLI_star(kappa_at(k), steps_per_day, ramp_start_step, ramp_end_step, L_DLI_target)
        return T_in_star, Z_in_star, B_in_star, L_DLI_star

    # =========================
    # [Block 4] Decision variables layout + bounds
    #   z = [X1..XK, U0..U(K-1), S_*(6K), L_delta(K)]
    # =========================
    X_LO, X_HI = -1e20, 1e20
    Slack_LO, Slack_HI = 0.0, 1e20
    Ld_LO, Ld_HI = 0.0, 1e20

    bounds: List[Tuple[float, float]] = [(X_LO, X_HI)] * (K * num_x)
    bounds += [(0.0, 1.0)] * (K * num_u)
    bounds += [(Slack_LO, Slack_HI)] * (6 * K)
    bounds += [(Ld_LO, Ld_HI)] * K

    # =========================
    # [Block 5] Initial guess (warm-start + consistent X0)
    # =========================
    U0 = np.zeros((K, num_u), dtype=float) if u_warm_start is None else np.clip(u_warm_start, 0.0, 1.0)

    X0 = np.zeros((K, num_x), dtype=float)
    x = np.asarray(x_ini, dtype=float).reshape(num_x)
    for k in range(K):
        x = np.asarray(model.step(x, U0[k], d_pred_h[k]), dtype=float).reshape(num_x)
        X0[k] = x

    z0 = np.concatenate([X0.reshape(-1), U0.reshape(-1), np.zeros(6 * K), np.zeros(K)], axis=0)

    def unpack(z: np.ndarray):
        z = np.asarray(z, dtype=float).reshape(-1)
        off = 0
        X = z[off : off + K * num_x].reshape(K, num_x)  # x_1..x_K
        off += K * num_x
        U = z[off : off + K * num_u].reshape(K, num_u)  # u_0..u_{K-1}
        off += K * num_u
        S_T_minus = z[off : off + K]; off += K
        S_T_plus = z[off : off + K]; off += K
        S_Z_minus = z[off : off + K]; off += K
        S_Z_plus = z[off : off + K]; off += K
        S_B_minus = z[off : off + K]; off += K
        S_B_plus = z[off : off + K]; off += K
        L_delta = z[off : off + K]
        return X, U, S_T_minus, S_T_plus, S_Z_minus, S_Z_plus, S_B_minus, S_B_plus, L_delta

    def x_at(k: int, X: np.ndarray) -> np.ndarray:
        return x_ini if k == 0 else X[k - 1]

    # =========================
    # [Block 6] Derived quantities (Z_in, B_in) + resource costs
    # =========================
    def Z_in(H_in: float) -> float:
        return (H_tilde_sat - float(H_in)) * xi

    def B_in(C_in: float) -> float:
        return beta_C * float(C_in)

    def resource_costs(Uk: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
        U_heat = float(Uk[U_heat_idx])
        U_fan = float(Uk[U_fan_idx])
        U_pad = float(Uk[U_pad_idx])
        U_LED = float(Uk[U_LED_idx])
        U_hum = float(Uk[U_hum_idx])
        U_dehum = float(Uk[U_dehum_idx])
        U_dos = float(Uk[U_dos_idx])

        E_heat = alpha_heat * (climate_parameters.Q_bar_heat * U_heat) * Delta_t / 3.6e6
        E_fan = alpha_elec * (S_fan * climate_parameters.V_bar_fan * U_fan) * Delta_t / 3.6e6
        E_LED = alpha_elec * (climate_parameters.P_bar_LED * U_LED) * Delta_t / 3.6e6

        F_pad = (climate_parameters.c_vh * climate_parameters.V_bar_fan * U_fan * climate_parameters.DeltaT_pad * U_pad) / climate_parameters.q_evap
        E_pad = alpha_water * F_pad * Delta_t

        E_hum = alpha_hum * (climate_parameters.F_bar_hum * U_hum) * Delta_t
        E_dehum = alpha_dehum * (climate_parameters.F_bar_dehum * U_dehum) * Delta_t
        E_dos = alpha_dos * (climate_parameters.D_bar_dos * U_dos) * Delta_t
        return E_heat, E_fan, E_LED, E_pad, E_hum, E_dehum, E_dos

    # =========================
    # [Block 7] Objective J(z)
    # =========================
    def objective(z: np.ndarray) -> float:
        X, U, S_T_minus, S_T_plus, S_Z_minus, S_Z_plus, S_B_minus, S_B_plus, L_delta = unpack(z)
        J = 0.0
        for k in range(K):
            disc = gamma ** k
            xk = x_at(k, X)

            T_in_val = float(xk[T_in_idx])
            H_in_val = float(xk[H_in_idx])
            C_in_val = float(xk[C_in_idx])

            Z_val = Z_in(H_in_val)
            B_val = B_in(C_in_val)

            T_star, Z_star, B_star, _ = stars_at(k)
            lambda_B = lambda_B_day if is_day(kappa_at(k)) else lambda_B_night

            E_heat, E_fan, E_LED, E_pad, E_hum, E_dehum, E_dos = resource_costs(U[k])

            stage = (
                E_heat + E_fan + E_LED + E_pad + E_hum + E_dehum + E_dos
                + lambda_T * (T_in_val - T_star) ** 2
                + lambda_Z * (Z_val - Z_star) ** 2
                + lambda_B * (B_val - B_star) ** 2
                + lambda_L * (float(L_delta[k]) ** 2)
                + lambda_T_plus * float(S_T_plus[k]) + lambda_T_minus * float(S_T_minus[k])
                + lambda_Z_plus * float(S_Z_plus[k]) + lambda_Z_minus * float(S_Z_minus[k])
                + lambda_B_plus * float(S_B_plus[k]) + lambda_B_minus * float(S_B_minus[k])
            )
            J += disc * stage
        return float(J)

    # =========================
    # [Block 8] Constraints (eq: dynamics, ineq: comfort/DLI/fan>=pad)
    # =========================
    def g_eq(z: np.ndarray) -> np.ndarray:
        X, U, *_ = unpack(z)
        resids = []
        for k in range(K):
            xk = x_ini if k == 0 else X[k - 1]
            xkp1_var = X[k]
            xkp1_model = np.asarray(model.step(xk, U[k], d_pred_h[k]), dtype=float).reshape(num_x)
            resids.append((xkp1_var - xkp1_model).reshape(-1))
        return np.concatenate(resids, axis=0)

    def g_ineq(z: np.ndarray) -> np.ndarray:
        X, U, S_T_minus, S_T_plus, S_Z_minus, S_Z_plus, S_B_minus, S_B_plus, L_delta = unpack(z)
        g: List[float] = []
        for k in range(K):
            T_under, T_bar, B_under, B_bar = bounds_at(k)
            L_DLI_star = _L_DLI_star(kappa_at(k), steps_per_day, ramp_start_step, ramp_end_step, L_DLI_target)

            xk = x_at(k, X)
            T_in_val = float(xk[T_in_idx])
            H_in_val = float(xk[H_in_idx])
            C_in_val = float(xk[C_in_idx])
            L_DLI_val = float(xk[L_DLI_idx])

            Z_val = Z_in(H_in_val)
            B_val = B_in(C_in_val)

            g.append(float(U[k][U_fan_idx] - U[k][U_pad_idx]))

            g.append(T_in_val - (T_under + float(S_T_minus[k])))
            g.append((T_bar + float(S_T_plus[k])) - T_in_val)

            g.append(Z_val - (Z_in_under + float(S_Z_minus[k])))
            g.append((Z_in_bar + float(S_Z_plus[k])) - Z_val)

            g.append(B_val - (B_under + float(S_B_minus[k])))
            g.append((B_bar + float(S_B_plus[k])) - B_val)

            g.append(float(L_delta[k]) + L_DLI_val - float(L_DLI_star))

        return np.asarray(g, dtype=float)

    # =========================
    # [Block 9] Finite-difference derivatives (IPOPT)
    # =========================
    def _fd_grad(fun, x: np.ndarray, eps: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        fx = float(fun(x))
        g = np.zeros_like(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            g[i] = (float(fun(xp)) - fx) / eps
        return g

    def _fd_jac(fun, x: np.ndarray, eps: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        fx = np.asarray(fun(x), dtype=float).reshape(-1)
        J = np.zeros((fx.size, x.size), dtype=float)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            fp = np.asarray(fun(xp), dtype=float).reshape(-1)
            J[:, i] = (fp - fx) / eps
        return J

    # =========================
    # [Block 10] Solve with IPOPT
    # =========================
    cons = [
        {"type": "eq", "fun": g_eq, "jac": lambda z: _fd_jac(g_eq, z, fd_eps)},
        {"type": "ineq", "fun": g_ineq, "jac": lambda z: _fd_jac(g_ineq, z, fd_eps)},
    ]

    res = minimize_ipopt(
        fun=objective,
        x0=z0,
        jac=lambda z: _fd_grad(objective, z, fd_eps),
        bounds=bounds,
        constraints=cons,
        options=solver_options,
    )

    z_opt = np.asarray(res.x, dtype=float).reshape(-1)
    _X_opt, U_opt, *_ = unpack(z_opt)

    iters = getattr(res, "nit", None)
    if iters is None:
        iters = getattr(res, "iters", None)

    return NmpcSolveResult(
        U_opt=U_opt,
        success=bool(getattr(res, "success", False)),
        objective=float(getattr(res, "fun", np.nan)),
        status=int(getattr(res, "status", -1)),
        message=str(getattr(res, "message", "")),
        iters=None if iters is None else int(iters),
    )


def _L_DLI_star(kappa: int, steps_per_day: int, ramp_start_step: int, ramp_end_step: int, L_DLI_target: float) -> float:
    if kappa < ramp_start_step:
        return 0.0
    if kappa < ramp_end_step:
        return float(L_DLI_target) * float(kappa - ramp_start_step) / float(ramp_end_step - ramp_start_step)
    return float(L_DLI_target)
