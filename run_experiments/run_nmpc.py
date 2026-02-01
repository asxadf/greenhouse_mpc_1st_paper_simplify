# run_experiments/run_nmpc.py
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.builders.builder_1_indoor_climate_one_step_Euler import load_one_step_euler_from_yamls
from src.builders.builder_2_nmpc_modeling_and_solving import modeling_solving_nmpc_problem

_THIS = Path(__file__).resolve()
REPO = _THIS.parents[1]
SRC = REPO / "src"

# -----------------------------
# All path-loading definitions
# -----------------------------
CONFIGS_DIR = REPO / "configs"
DATA_PROCESSED_DIR = REPO / "data" / "processed"
RESULTS_NMPC_DIR = REPO / "results" / "nmpc"

SETTING_NMPC_YAML = CONFIGS_DIR / "setting_nmpc.yaml"
KEEPER_YAML = CONFIGS_DIR / "variable_and_parameter_keeper.yaml"

OUTDOOR_PREDICTION_CSV = (DATA_PROCESSED_DIR / "outdoor_prediction.csv").resolve()
OUTDOOR_REALIZATION_CSV = (DATA_PROCESSED_DIR / "outdoor_realization.csv").resolve()

CURRENT_STEP_INFORMATION_JSON = RESULTS_NMPC_DIR / "current_step_information.json"

CSV_COLS = {
    "T_out": "T_Outdoor(C)",
    "H_out": "H_Outdoor(g/m3)",
    "C_out": "CO2_Outdoor(g/m3)",
    "R_out": "Radiation_Outdoor(w/m2)",
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _current_step_information_path() -> Path:
    return CURRENT_STEP_INFORMATION_JSON


def _load_or_init_current_step_information(defs, keeper: Dict[str, Any]) -> Dict[str, Any]:
    path = _current_step_information_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    x_ini_map = keeper["indoor_climate_model_parameters"]["x_ini"]
    x = np.asarray(
        [float(x_ini_map[name]) for name in defs.state_names],
        dtype=float,
    ).reshape(len(defs.state_names)).tolist()

    info = {"step_index": 0, "x": x, "u_warm_start": None}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)
    return info


def _save_current_step_information(info: Dict[str, Any]) -> None:
    path = _current_step_information_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)


def _shift_warm_start(u_ws: Any, K: int, n_u: int) -> Optional[np.ndarray]:
    if u_ws is None:
        return None
    arr = np.asarray(u_ws, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != n_u:
        return None
    if arr.shape[0] != K:
        if arr.shape[0] > K:
            arr = arr[:K]
        else:
            pad = np.repeat(arr[-1:], repeats=K - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
    return np.vstack([arr[1:], arr[-1:]])


def _load_disturbance_csv(path: Path, defs) -> Tuple[np.ndarray, List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)

    timestamps = [row[0] for row in rows]
    cols = header[1:]
    idx = {name: i for i, name in enumerate(cols)}

    data = np.array([[float(x) for x in row[1:]] for row in rows], dtype=float)

    d = np.zeros((data.shape[0], len(defs.disturbance_names)), dtype=float)
    for j, dname in enumerate(defs.disturbance_names):
        col = CSV_COLS[dname]
        d[:, j] = data[:, idx[col]]

    return d, timestamps


def _slice_horizon(D: np.ndarray, i: int, K: int) -> Tuple[np.ndarray, int]:
    if i + K <= D.shape[0]:
        return D[i : i + K], K
    K_eff = max(0, D.shape[0] - i)
    pad = np.repeat(D[-1:], repeats=i + K - D.shape[0], axis=0)
    return np.concatenate([D[i:], pad], axis=0), K_eff


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if ts is None:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _fmt_dt(dt_s: float) -> str:
    mins = dt_s / 60.0
    if abs(mins - round(mins)) < 1e-9:
        return f"{int(round(mins))}min"
    return f"{mins:.3g}min"


def _status_name(status: int) -> str:
    return "Solve_Succeeded" if status == 0 else "Unknown"


def _L_DLI_star(kappa: int, steps_per_day: int, day_start: int, day_end: int, target: float) -> float:
    if kappa < day_start:
        return 0.0
    if kappa < day_end:
        return float(target) * float(kappa - day_start) / float(day_end - day_start)
    return float(target)


def _horizon_end_time(
    timestamps: List[str],
    i: int,
    K: int,
    dt_s: float,
    t0: Optional[datetime],
) -> Optional[datetime]:
    """
    Returns the end time of the NMPC look-ahead window.

    Preference order:
      1) timestamps[i + K] if available and parseable
      2) t0 + K * dt_s (fallback)
    """
    if t0 is None:
        return None

    j = i + int(K)
    if 0 <= j < len(timestamps):
        tK = _parse_iso(timestamps[j])
        if tK is not None:
            return tK

    return t0 + timedelta(seconds=float(K) * float(dt_s))


def _print_step_block(
    *,
    step_i: int,
    steps_per_day: int,
    K: int,
    K_eff: int,
    dt_s: float,
    controller: str,
    t0: Optional[datetime],
    t1: Optional[datetime],
    tK: Optional[datetime],
    is_day: bool,
    solve_status: int,
    solve_iters: Optional[int],
    objective: float,
    solve_time_s: float,
    u0: np.ndarray,
    control_names: List[str],
    x_next: np.ndarray,
    defs,
    mpc: Dict[str, Any],
    model,
    d_real_i: np.ndarray,
    d_real_ts: Optional[datetime],
) -> None:
    sep = "-" * 78
    kappa = step_i % steps_per_day

    t0s = t0.strftime("%Y-%m-%d %H:%M:%S") if t0 else "N/A"
    t1s = t1.strftime("%Y-%m-%d %H:%M:%S") if t1 else "N/A"
    tKs = tK.strftime("%Y-%m-%d %H:%M:%S") if tK else "N/A"
    tod = "day" if is_day else "night"

    print(sep)
    print(f"Step-of-Simulation : {step_i + 1} / {steps_per_day}")
    print(f"Step-of-Day        : kappa = {kappa} / {steps_per_day}")
    print(f"Target period      : {t0s} → {t1s}")
    print(f"Time-of-day        : {tod}")
    print(f"Controller         : {controller}")
    print(f"Look-ahead window  : [{t0s} → {tKs}]  (K={K}, K_eff={K_eff}, Δt={_fmt_dt(dt_s)})")
    print()
    print("[1] Solving MPC (NMPC)...")
    iters_str = "N/A" if solve_iters is None else str(solve_iters)
    print(
        f"Finished solving (NMPC).  status={solve_status} ({_status_name(solve_status)})"
        f"  iters={iters_str}  objective= {objective: .6f}  solve_time={solve_time_s:.3f}s"
    )
    print()
    print(f"[2] Control action U (apply over [{t0s} → {t1s}))")
    w = max(len(n) for n in control_names)
    for name, val in zip(control_names, u0.tolist()):
        v = float(val)
        print(f"  {name:<{w}s} : {v: .6f}" if abs(v) >= 1e-4 else f"  {name:<{w}s} : {v: .3e}")

    print()
    print("[3] Apply control to digital twin + integrate one step")
    dts = d_real_ts.strftime("%Y-%m-%d %H:%M:%S") if d_real_ts else "N/A"
    print(f"  Using realized disturbance with timestamp: {dts}")
    print(f"  Integrating over [{t0s} → {t1s}] ...")
    print()
    print(f"[4] Tracking output at {t1s}")

    H_tilde_sat = float(mpc["H_tilde_sat"])
    xi = float(mpc["xi"])
    beta_C = float(mpc["beta_C"])

    T_in = float(x_next[defs.state_idx["T_in"]])
    H_in = float(x_next[defs.state_idx["H_in"]])
    C_in = float(x_next[defs.state_idx["C_in"]])
    L_dli = float(x_next[defs.state_idx["L_DLI"]])

    Z_in = (H_tilde_sat - H_in) * xi
    B_in = beta_C * C_in

    day_start = int(mpc["L_DLI_profile"]["ramp_start_step"])
    day_end = int(mpc["L_DLI_profile"]["ramp_end_step"])
    dli_target = float(mpc["L_DLI_target"])

    kappa_next = (kappa + 1) % steps_per_day
    is_day_next = day_start <= kappa_next < day_end

    if is_day_next:
        T_lo = float(mpc["T_in_under"]["day"])
        T_hi = float(mpc["T_in_bar"]["day"])
        B_lo = float(mpc["B_in_under"]["day"])
        B_hi = float(mpc["B_in_bar"]["day"])
        lamB = float(mpc["lambda_B"]["day"])
    else:
        T_lo = float(mpc["T_in_under"]["night"])
        T_hi = float(mpc["T_in_bar"]["night"])
        B_lo = float(mpc["B_in_under"]["night"])
        B_hi = float(mpc["B_in_bar"]["night"])
        lamB = float(mpc["lambda_B"]["night"])

    Z_lo = float(mpc["Z_in_under"])
    Z_hi = float(mpc["Z_in_bar"])

    T_ref = 0.5 * (T_lo + T_hi)
    Z_ref = 0.5 * (Z_lo + Z_hi)
    B_ref = 0.5 * (B_lo + B_hi) if lamB != 0.0 else 0.5 * (B_lo + B_hi)
    L_ref = _L_DLI_star(kappa_next, steps_per_day, day_start, day_end, dli_target)

    print("  Outputs:")
    print(f"    T_Indoor (C)          : {T_in:6.2f}   [{T_lo:.2f}, {T_hi:.2f}], ref is {T_ref:.2f}")
    print(f"    VPD_Indoor (kPa)      : {Z_in:6.3f}   [{Z_lo:.3f}, {Z_hi:.3f}], ref is {Z_ref:.3f}")
    print(f"    CO2_Indoor (ppm)      : {B_in:6.1f}   [{B_lo:.1f}, {B_hi:.1f}],  ref is {B_ref:.1f}")
    print(f"    L_DLI (mol/m2/day)    : {L_dli:6.2f}   [bounds N/A],  ref is {L_ref:.2f}")

    print(f"  Outdoor disturbance (realized, timestamp={dts}):")
    print(f"    T_Outdoor (C)           : {d_real_i[defs.dist_idx['T_out']]: .6f}")
    print(f"    H_Outdoor (g/m3)        : {d_real_i[defs.dist_idx['H_out']]: .6f}")
    print(f"    CO2_Outdoor (g/m3)      : {d_real_i[defs.dist_idx['C_out']]: .6f}")
    print(f"    Radiation_Outdoor (w/m2): {d_real_i[defs.dist_idx['R_out']]: .6f}")
    print(sep)
    print()


def main() -> None:
    cfg_nmpc = _read_yaml(SETTING_NMPC_YAML)
    keeper = _read_yaml(KEEPER_YAML)

    mpc = keeper["mpc_global_parameters"]
    solver_options = cfg_nmpc["solver"]["options"]
    fd_eps = float(cfg_nmpc["solver"]["fd_eps"])

    model = load_one_step_euler_from_yamls()
    defs = model.defs

    d_pred_full, ts_pred = _load_disturbance_csv(OUTDOOR_PREDICTION_CSV, defs)
    d_real_full, ts_real = _load_disturbance_csv(OUTDOOR_REALIZATION_CSV, defs)
    timestamps = ts_real if ts_real else ts_pred

    current = _load_or_init_current_step_information(defs, keeper)

    K = int(mpc["K"])
    steps_per_day = int(mpc["steps_per_day"])
    dt_s = float(model.dt_s)

    remaining = int(d_real_full.shape[0]) - int(current["step_index"])

    day_start = int(mpc["L_DLI_profile"]["ramp_start_step"])
    day_end = int(mpc["L_DLI_profile"]["ramp_end_step"])

    for _ in range(remaining):
        i = int(current["step_index"])
        x_ini = np.asarray(current["x"], dtype=float).reshape(len(defs.state_names))

        d_pred_h, K_eff = _slice_horizon(d_pred_full, i, K)
        d_real_i = d_real_full[i]

        t0 = _parse_iso(timestamps[i]) if i < len(timestamps) else None
        t1 = (
            _parse_iso(timestamps[i + 1])
            if (i + 1) < len(timestamps)
            else (t0 + timedelta(seconds=dt_s) if t0 else None)
        )
        tK = _horizon_end_time(timestamps, i, K, dt_s, t0)

        kappa = i % steps_per_day
        is_day_now = day_start <= kappa < day_end

        warm_u = _shift_warm_start(current.get("u_warm_start"), K, len(defs.control_names))

        t_solve0 = time.perf_counter()
        sol = modeling_solving_nmpc_problem(
            model=model,
            x_ini=x_ini,
            d_pred_h=d_pred_h,
            step_index=i,
            mpc_global_parameters=mpc,
            solver_options=solver_options,
            K_eff=K_eff,
            fd_eps=fd_eps,
            u_warm_start=warm_u,
        )
        solve_time_s = time.perf_counter() - t_solve0

        u0 = sol.U_opt[0].copy()
        x_next = np.asarray(model.step(x_ini, u0, d_real_i), dtype=float).reshape(x_ini.shape)

        _print_step_block(
            step_i=i,
            steps_per_day=steps_per_day,
            K=K,
            K_eff=K_eff,
            dt_s=dt_s,
            controller="nmpc",
            t0=t0,
            t1=t1,
            tK=tK,
            is_day=is_day_now,
            solve_status=sol.status,
            solve_iters=sol.iters,
            objective=sol.objective,
            solve_time_s=solve_time_s,
            u0=u0,
            control_names=defs.control_names,
            x_next=x_next,
            defs=defs,
            mpc=mpc,
            model=model,
            d_real_i=d_real_i,
            d_real_ts=t0,
        )

        current = {
            "step_index": i + 1,
            "x": x_next.astype(float).tolist(),
            "u_warm_start": sol.U_opt.astype(float).tolist(),
        }
        _save_current_step_information(current)


if __name__ == "__main__":
    main()
