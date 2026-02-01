# run_experiments/run_qmpc_deterministic.py
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.builders.builder_indoor_climate_one_step_Euler import load_one_step_euler_from_yamls
from src.builders.builder_qmpc_deterministic_modeling_and_solving import (
    AffineDynamicsMatrices,
    modeling_solving_qmpc_deterministic_problem,
)

_THIS = Path(__file__).resolve()
REPO = _THIS.parents[1]
CONFIGS_DIR = REPO / "configs"
DATA_PROCESSED_DIR = REPO / "data" / "processed"
RESULTS_DIR = REPO / "results" / "qmpc_deterministic"

SETTING_QMPC_YAML = CONFIGS_DIR / "setting_qmpc_deterministic.yaml"
KEEPER_YAML = CONFIGS_DIR / "variable_and_parameter_keeper.yaml"

OUTDOOR_PREDICTION_CSV = (DATA_PROCESSED_DIR / "outdoor_prediction.csv").resolve()
OUTDOOR_REALIZATION_CSV = (DATA_PROCESSED_DIR / "outdoor_realization.csv").resolve()

CURRENT_STEP_INFORMATION_JSON = RESULTS_DIR / "current_step_information.json"

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
    x = np.asarray([float(x_ini_map[name]) for name in defs.state_names], dtype=float).tolist()

    info = {"step_index": 0, "x": x}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)
    return info


def _save_current_step_information(info: Dict[str, Any]) -> None:
    path = _current_step_information_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)


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


def _slice_horizon(D: np.ndarray, i: int, K: int) -> np.ndarray:
    if i + K <= D.shape[0]:
        return D[i : i + K]
    pad = np.repeat(D[-1:], repeats=i + K - D.shape[0], axis=0)
    return np.concatenate([D[i:], pad], axis=0)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if ts is None:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _horizon_end_time(
    timestamps: List[str],
    i: int,
    K: int,
    dt_s: float,
    t0: Optional[datetime],
) -> Optional[datetime]:
    if t0 is None:
        return None

    j = i + int(K)
    if 0 <= j < len(timestamps):
        tK = _parse_iso(timestamps[j])
        if tK is not None:
            return tK

    return t0 + timedelta(seconds=float(K) * float(dt_s))


def _L_DLI_star(kappa: int, steps_per_day: int, ramp_start: int, ramp_end: int, target: float) -> float:
    if kappa < ramp_start:
        return 0.0
    if kappa < ramp_end:
        return float(target) * float(kappa - ramp_start) / float(ramp_end - ramp_start)
    return float(target)


def _resolve_affine_dynamics_dir(model_block: Dict[str, Any]) -> Path:
    raw = Path(model_block["affine_dynamics_dir"])
    base = raw if raw.is_absolute() else (REPO / raw).resolve()
    candidates = [
        base,
        REPO / "data" / "for_fitting_qmpc" / "fitting_results",
        REPO / "data" / "for_fitting_qmpc" / "fitting_results" / "readable_exports",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return base


def _load_csv_matrix(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    if len(header) > 1:
        return np.array([[float(x) for x in row[1:]] for row in rows], dtype=float)
    return np.array([[float(x) for x in row] for row in rows], dtype=float)


def _load_affine_matrices(model_block: Dict[str, Any]) -> AffineDynamicsMatrices:
    base = _resolve_affine_dynamics_dir(model_block)
    preference = list(model_block.get("matrix_preference_order", []))

    def try_npz(path: Path) -> Optional[AffineDynamicsMatrices]:
        if not path.exists():
            return None
        data = np.load(path)
        return AffineDynamicsMatrices(
            M=np.asarray(data["M"], dtype=float),
            N=np.asarray(data["N"], dtype=float),
            O=np.asarray(data["O"], dtype=float),
            m=np.asarray(data["m"], dtype=float),
        )

    def try_csv(directory: Path) -> Optional[AffineDynamicsMatrices]:
        if not directory.exists():
            return None
        m_path = directory / "M_matrix.csv"
        n_path = directory / "N_matrix.csv"
        o_path = directory / "O_matrix.csv"
        b_path = directory / "m_bias.csv"
        if not (m_path.exists() and n_path.exists() and o_path.exists() and b_path.exists()):
            return None
        return AffineDynamicsMatrices(
            M=_load_csv_matrix(m_path),
            N=_load_csv_matrix(n_path),
            O=_load_csv_matrix(o_path),
            m=_load_csv_matrix(b_path).reshape(-1),
        )

    for pref in preference:
        if pref.endswith(".npz"):
            path = base / pref
            if "*" in pref:
                matches = list(base.glob(pref))
                if matches:
                    path = matches[0]
            matrices = try_npz(path)
            if matrices is not None:
                return matrices
        if pref.endswith(".csv") or "readable_exports" in pref:
            if "*" in pref:
                candidate = base
            else:
                candidate = base / pref
            if candidate.is_dir():
                matrices = try_csv(candidate)
            else:
                matrices = try_csv(candidate.parent)
            if matrices is not None:
                return matrices

    matrices = try_npz(base / "model_matrices.npz")
    if matrices is not None:
        return matrices

    csv_dir = base / "readable_exports" if (base / "readable_exports").exists() else base
    matrices = try_csv(csv_dir)
    if matrices is not None:
        return matrices

    raise FileNotFoundError(f"No affine dynamics matrices found in {base}")


def _print_named(title: str, names: List[str], values: np.ndarray) -> None:
    print(f"\n{title}")
    for name, val in zip(names, values.tolist()):
        print(f"  {name:>12s} : {val: .6g}")


def main() -> None:
    settings = _read_yaml(SETTING_QMPC_YAML)
    keeper = _read_yaml(KEEPER_YAML)

    model = load_one_step_euler_from_yamls()
    defs = model.defs

    info = _load_or_init_current_step_information(defs, keeper)
    step_index = int(info["step_index"])
    x_ini = np.asarray(info["x"], dtype=float)

    D_pred, pred_ts = _load_disturbance_csv(OUTDOOR_PREDICTION_CSV, defs)
    D_real, real_ts = _load_disturbance_csv(OUTDOOR_REALIZATION_CSV, defs)

    K = int(keeper["mpc_global_parameters"]["K"])
    d_pred_h = _slice_horizon(D_pred, step_index, K)
    d_real_i = D_real[min(step_index, D_real.shape[0] - 1)]

    affine = _load_affine_matrices(settings["model"])

    result = modeling_solving_qmpc_deterministic_problem(
        model=model,
        x_ini=x_ini,
        d_pred_h=d_pred_h,
        d_realization=d_real_i,
        step_index=step_index,
        mpc_global_parameters=keeper["mpc_global_parameters"],
        affine=affine,
        solver_options=settings.get("solver", {}).get("options", {}),
    )

    info = {
        "step_index": step_index + 1,
        "x": result.x_next.tolist(),
    }
    _save_current_step_information(info)

    t0 = _parse_iso(pred_ts[step_index]) if step_index < len(pred_ts) else None
    t1 = _parse_iso(real_ts[step_index]) if step_index < len(real_ts) else None
    tK = _horizon_end_time(pred_ts, step_index, K, keeper["mpc_global_parameters"]["Delta_t"], t0)

    print("\n=== QMPC Deterministic Step ===")
    print(f"step_index : {step_index}")
    if t0:
        print(f"t0         : {t0.isoformat()}")
    if t1:
        print(f"t_real     : {t1.isoformat()}")
    if tK:
        print(f"t_horizon  : {tK.isoformat()}")
    print(f"solve_ok   : {result.success}")
    print(f"status     : {result.status} ({result.message})")
    if result.solve_time_s is not None:
        print(f"solve_time : {result.solve_time_s:.3f}s")
    print(f"objective  : {result.objective}")

    _print_named("u0 (control)", defs.control_names, result.u0)
    _print_named("x_next (digital twin)", defs.state_names, result.x_next)


if __name__ == "__main__":
    main()
