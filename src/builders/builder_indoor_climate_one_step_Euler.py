# src/builders/builder_indoor_climate_one_step_Euler.py
"""
One-step Euler-forward discretization for the indoor climate ODE model.

Single-setup policy (PyCharm-friendly):
- Always load configs/variable_and_parameter_keeper.yaml
- No CLI arguments. No multiple choices.
- No explicit raise(...) in this module; errors are native Python exceptions.

Project structure assumed:
  <project_root>/
    configs/
      variable_and_parameter_keeper.yaml
    src/builders/builder_indoor_climate_one_step_Euler.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np
import yaml

PathLike = Union[str, Path]

KEEPER_REL = Path("configs/variable_and_parameter_keeper.yaml")


# =============================================================================
# [BLOCK] DATA LOADING
# =============================================================================
def _read_yaml(path: PathLike) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))  # type: ignore[return-value]


def _index_map(names: Sequence[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def _locate_project_root(start: Path) -> Path:
    start = start.resolve()
    for base in [start, *start.parents]:
        if (base / "configs").is_dir():
            return base
    return start.parents[-1] if start.parents else start


def _get_keeper_path() -> Path:
    root = _locate_project_root(Path(__file__).resolve().parent)
    return (root / KEEPER_REL).resolve()


@dataclass(frozen=True)
class IndoorClimateParams:
    A_floor: float
    A_roof: float
    A_wall: float
    V_gh: float

    c_vh: float
    upsilon_roof: float
    upsilon_wall: float
    V_bar_fan: float
    V_bar_nat: float
    lambda_leak: float
    Q_bar_heat: float
    P_bar_LED: float
    DeltaT_pad: float
    q_evap: float

    C_hat_in: float
    D_bar_dos: float
    D_bar_ass: float

    F_bar_hum: float
    F_bar_dehum: float
    nu_surf: float

    L: float
    tau: float
    rho: float
    r_b: float
    r_bar_s: float
    r_under_s: float
    T_sr: float
    sigma: float
    delta_tran: float
    delta_sat: float
    delta_vc: float
    delta_sr: float

    phi_solar: float
    phi_LED: float

    eta_LED_r: float
    eta_LED_cano: float
    eta_evap: float
    eta_cover: float
    eta_short: float
    eta_ext: float
    eta_shad: float
    eta_warm: float

    chi: float
    omega_Q: float
    omega_F: float
    H_tilde: float
    zeta: float


def _load_model_params(model_block: Mapping[str, Any]) -> IndoorClimateParams:
    keys = (
        "A_floor A_roof A_wall V_gh "
        "c_vh upsilon_roof upsilon_wall V_bar_fan V_bar_nat lambda_leak Q_bar_heat P_bar_LED DeltaT_pad q_evap "
        "C_hat_in D_bar_dos D_bar_ass "
        "F_bar_hum F_bar_dehum nu_surf "
        "L tau rho r_b r_bar_s r_under_s T_sr sigma delta_tran delta_sat delta_vc delta_sr "
        "phi_solar phi_LED "
        "eta_LED_r eta_LED_cano eta_evap eta_cover eta_short eta_ext eta_shad eta_warm "
        "chi omega_Q omega_F H_tilde zeta"
    ).split()
    return IndoorClimateParams(**{k: float(model_block[k]) for k in keys})


@dataclass(frozen=True)
class IndoorClimateVectorDefs:
    state_names: list[str]
    control_names: list[str]
    disturbance_names: list[str]
    state_idx: Dict[str, int]
    control_idx: Dict[str, int]
    dist_idx: Dict[str, int]


def _load_vector_defs(model_block: Mapping[str, Any]) -> IndoorClimateVectorDefs:
    sx = list(model_block["state_vector"])
    su = list(model_block["control_vector"])
    sd = list(model_block["disturbance_vector"])
    return IndoorClimateVectorDefs(
        state_names=sx,
        control_names=su,
        disturbance_names=sd,
        state_idx=_index_map(sx),
        control_idx=_index_map(su),
        dist_idx=_index_map(sd),
    )


def load_one_step_euler_from_yamls(*, dt_override_s: float | None = None) -> "IndoorClimateEulerOneStep":
    keeper_path = _get_keeper_path()

    keeper = _read_yaml(keeper_path)
    model_block = keeper["indoor_climate_model_parameters"]

    params = _load_model_params(model_block)
    defs = _load_vector_defs(model_block)

    dt_s = float(dt_override_s) if dt_override_s is not None else float(
        keeper["mpc_global_parameters"]["Delta_t"]
    )
    return IndoorClimateEulerOneStep(dt_s=dt_s, params=params, defs=defs)


# =============================================================================
# [BLOCK] ODE BUILDING (equation-ordered)
# =============================================================================
def _safe_exp(x: np.ndarray | float) -> np.ndarray | float:
    return np.exp(np.clip(x, -80.0, 80.0))


def _ode_rhs(
    x: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    p: IndoorClimateParams,
    idx: IndoorClimateVectorDefs,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    d = np.asarray(d, dtype=float).reshape(-1)

    # -------------------------------------------------------------------------
    # [BLOCK] Unpack
    # -------------------------------------------------------------------------
    T_in = float(x[idx.state_idx["T_in"]])
    H_in = float(x[idx.state_idx["H_in"]])
    C_in = float(x[idx.state_idx["C_in"]])

    U_heat = float(u[idx.control_idx["U_heat"]])
    U_fan = float(u[idx.control_idx["U_fan"]])
    U_nat = float(u[idx.control_idx["U_nat"]])
    U_pad = float(u[idx.control_idx["U_pad"]])
    U_dos = float(u[idx.control_idx["U_dos"]])
    U_LED = float(u[idx.control_idx["U_LED"]])
    U_hum = float(u[idx.control_idx["U_hum"]])
    U_dehum = float(u[idx.control_idx["U_dehum"]])
    U_scre_shad = float(u[idx.control_idx["U_scre_shad"]])
    U_scre_warm = float(u[idx.control_idx["U_scre_warm"]])

    T_out = float(d[idx.dist_idx["T_out"]])
    H_out = float(d[idx.dist_idx["H_out"]])
    C_out = float(d[idx.dist_idx["C_out"]])
    R_out = float(d[idx.dist_idx["R_out"]])

    # -------------------------------------------------------------------------
    # [BLOCK] Intermediate variables (match LaTeX order)
    # -------------------------------------------------------------------------
    DeltaT = T_in - T_out
    V_vent = p.V_bar_fan * U_fan + p.V_bar_nat * U_nat + (p.lambda_leak * p.V_gh) / 3600.0

    R_cano_solar = p.eta_cover * (1.0 - p.eta_shad * U_scre_shad) * R_out
    R_cano_LED = (p.eta_LED_r * p.eta_LED_cano * p.P_bar_LED * U_LED) / p.A_floor
    R_cano_glo = R_cano_solar + R_cano_LED
    exp_extL = _safe_exp(p.eta_ext * p.L)
    R_cano_abs = p.eta_short * ((exp_extL - 1.0) / exp_extL) * R_cano_glo

    X_cano = p.phi_solar * R_cano_solar + p.phi_LED * R_cano_LED

    r_s = (p.r_under_s + p.r_bar_s * _safe_exp(-(p.tau * R_cano_abs) / p.L)) * (
        1.0 + p.delta_sr * (T_in - p.T_sr) ** 2
    )
    g_tran = 2.0 * p.L / ((1.0 + 0.7584 * _safe_exp(p.delta_tran * T_in)) * p.r_b + r_s)

    H_in_sat = p.H_tilde * _safe_exp(p.delta_sat * T_in)
    H_cano = H_in_sat + p.chi * (p.r_b * R_cano_abs) / (2.0 * p.L * p.q_evap)

    T_cover = (2.0 * T_out + T_in) / 3.0
    smooth = ((T_in - T_cover) + float(np.sqrt((T_in - T_cover) ** 2 + p.sigma**2))) / 2.0
    g_vc = p.nu_surf * float(smooth ** (1.0 / 3.0))

    # -------------------------------------------------------------------------
    # [BLOCK] Temperature dynamics (Q)
    # -------------------------------------------------------------------------
    Q_heat = p.Q_bar_heat * U_heat
    Q_vent = p.c_vh * DeltaT * V_vent

    F_pad = (p.c_vh * p.V_bar_fan * U_fan * p.DeltaT_pad * U_pad) / p.q_evap
    Q_cool = p.q_evap * (F_pad + p.eta_evap * p.F_bar_hum * U_hum)

    Q_solar = R_cano_solar * p.A_floor
    Q_DeltaT = (p.upsilon_roof * p.A_roof * (1.0 - p.eta_warm * U_scre_warm) + p.upsilon_wall * p.A_wall) * DeltaT
    Q_LED = p.P_bar_LED * U_LED
    Q_tran = p.q_evap * g_tran * (H_cano - H_in) * p.A_floor

    dT_dt = (Q_heat - Q_vent - Q_cool + Q_solar - Q_DeltaT + Q_LED - Q_tran) / (
        p.omega_Q * p.V_gh * p.c_vh
    )

    # -------------------------------------------------------------------------
    # [BLOCK] Humidity dynamics (H)
    # -------------------------------------------------------------------------
    F_cool = F_pad + p.eta_evap * p.F_bar_hum * U_hum
    F_dehum = p.F_bar_dehum * U_dehum
    F_vent = (H_in - H_out) * V_vent
    F_tran = g_tran * (H_cano - H_in) * p.A_floor
    F_vc = g_vc * (p.zeta * _safe_exp(p.delta_vc * T_in) * DeltaT - (H_in_sat - H_in)) * p.A_floor

    dH_dt = (F_cool - F_dehum - F_vent + F_tran - F_vc) / (p.omega_F * p.V_gh)

    # -------------------------------------------------------------------------
    # [BLOCK] CO2 dynamics (C)
    # -------------------------------------------------------------------------
    D_dos = p.D_bar_dos * U_dos
    D_vent = (C_in - C_out) * V_vent
    C_in_safe = max(C_in, 1e-9)
    D_ass = (
        p.D_bar_ass
        * (C_in_safe / (C_in_safe + p.C_hat_in))
        * (1.0 - _safe_exp(-p.rho * X_cano))
        * p.A_floor
    )
    dC_dt = (D_dos - D_vent - D_ass) / p.V_gh

    # -------------------------------------------------------------------------
    # [BLOCK] DLI dynamics (DLI)
    # -------------------------------------------------------------------------
    dL_dt = X_cano / 1e6

    dxdt = np.zeros((len(idx.state_names),), dtype=float)
    dxdt[idx.state_idx["T_in"]] = dT_dt
    dxdt[idx.state_idx["H_in"]] = dH_dt
    dxdt[idx.state_idx["C_in"]] = dC_dt
    dxdt[idx.state_idx["L_DLI"]] = dL_dt
    return dxdt


# =============================================================================
# [BLOCK] EULER DISCRETIZATION
# =============================================================================
@dataclass(frozen=True)
class IndoorClimateEulerOneStep:
    dt_s: float
    params: IndoorClimateParams
    defs: IndoorClimateVectorDefs

    @property
    def n_x(self) -> int:
        return len(self.defs.state_names)

    @property
    def n_u(self) -> int:
        return len(self.defs.control_names)

    @property
    def n_d(self) -> int:
        return len(self.defs.disturbance_names)

    def rhs(self, x_k: np.ndarray, u_k: np.ndarray, d_k: np.ndarray) -> np.ndarray:
        return _ode_rhs(x_k, u_k, d_k, self.params, self.defs)

    def step(self, x_k: np.ndarray, u_k: np.ndarray, d_k: np.ndarray) -> np.ndarray:
        x_k = np.asarray(x_k, dtype=float).reshape(-1)
        return x_k + self.dt_s * _ode_rhs(x_k, u_k, d_k, self.params, self.defs)


# =============================================================================
# [BLOCK] PYCHARM SMOKE RUN (no CLI)
# =============================================================================
if __name__ == "__main__":
    m = load_one_step_euler_from_yamls()

    x = np.array(
        [
            19.00,  # T_in
             7.41,  # H_in
             0.92,  # C_in
             0.00,  # L_DLI
        ],
        dtype=float,
    )

    u = np.array(
        [
            0.0,  # U_heat
            0.0,  # U_fan
            0.0,  # U_nat
            0.0,  # U_pad
            0.0,  # U_dos
            0.0,  # U_LED
            0.0,  # U_hum
            0.0,  # U_dehum
            0.0,  # U_scre_shad
            1.0,  # U_scre_warm
        ],
        dtype=float,
    )

    d = np.array(
        [
            3.00,  # T_out
            4.56,  # H_out
            0.78,  # C_out
            0.00,  # R_out
        ],
        dtype=float,
    )

    x_next = m.step(x, u, d)

    def _print_named(title: str, names: list[str], values: np.ndarray) -> None:
        print(f"\n{title}")
        for name, val in zip(names, values.tolist()):
            print(f"  {name:>12s} : {val: .6g}")

    print(f"dt_s = {m.dt_s:g}")

    _print_named("x_k (state)", m.defs.state_names, x)
    _print_named("u_k (control)", m.defs.control_names, u)
    _print_named("d_k (disturbance)", m.defs.disturbance_names, d)
    _print_named("x_k+1 (next state)", m.defs.state_names, x_next)
