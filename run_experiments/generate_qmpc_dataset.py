# run_experiments/generate_qmpc_dataset.py
from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.builders.builder_indoor_climate_one_step_Euler import load_one_step_euler_from_yamls


# =============================================================================
# Helpers: sampling + validation
# =============================================================================
def _bounds_for(names: list[str], table: dict[str, tuple[float, float]], default=(0.0, 1.0)) -> np.ndarray:
    b = np.empty((len(names), 2), dtype=float)
    for i, n in enumerate(names):
        b[i] = table.get(n, default)
    return b


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.random() * (hi - lo) + lo)


def _sample_row(rng: np.random.Generator, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return rng.random(bounds.shape[0]) * (hi - lo) + lo


def _validate_vec(vec: np.ndarray, names: list[str], limits: dict[str, tuple[float, float]], prefix: str) -> str | None:
    if not np.all(np.isfinite(vec)):
        return f"{prefix}_nonfinite"

    for v, n in zip(vec, names):
        lo, hi = limits.get(n, (-np.inf, np.inf))
        if v < lo:
            return f"{prefix}_{n}_below_min"
        if v > hi:
            return f"{prefix}_{n}_above_max"
    return None


def _print_stats(title: str, names: list[str], mat: np.ndarray) -> None:
    print(f"{title}:")
    width = max(len(n) for n in names) if names else 10
    for i, n in enumerate(names):
        col = mat[:, i]
        print(f"  {n:>{width}}: min={np.min(col):10.4f} max={np.max(col):10.4f} mean={np.mean(col):10.4f}")


def _clear_random_samples_dir(root: Path) -> Path:
    """
    Clears <root>/data/for_fitting_qmpc/random_samples safely.

    Safety: only deletes if the resolved path is exactly the expected target.
    """
    target = (root / "data/for_fitting_qmpc/random_samples").resolve()

    expected = ("data", "for_fitting_qmpc", "random_samples")
    if target.parts[-3:] != expected:
        raise RuntimeError(f"Refusing to delete unexpected directory: {target}")

    target.mkdir(parents=True, exist_ok=True)

    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return target


# =============================================================================
# Humidity physics: H_sat(T) in g/m^3 (absolute humidity at saturation)
# =============================================================================
def _h_sat_g_m3(T_c: float) -> float:
    """
    Saturation absolute humidity (g/m^3) as a function of temperature in Celsius.

    Steps:
      1) Saturation vapor pressure e_s(T) using Magnus formula (hPa)
      2) Convert to absolute humidity rho_v (g/m^3) via ideal gas:
           rho_v = 216.7 * e(hPa) / (T(K))
    """
    T_k = T_c + 273.15
    e_hpa = 6.112 * np.exp((17.62 * T_c) / (243.12 + T_c))
    rho_g_m3 = 216.7 * e_hpa / T_k
    return float(rho_g_m3)


# =============================================================================
# CO2-aware control sampling (minimal, but improves physicality)
# =============================================================================
def _maybe_index(names: list[str], key: str) -> int | None:
    try:
        return names.index(key)
    except ValueError:
        return None


def _apply_co2_aware_controls(
    rng: np.random.Generator,
    u_i: np.ndarray,
    control_names: list[str],
    C_in: float,
    C_out: float,
    C_in_bounds: tuple[float, float],
) -> None:
    """
    Overwrite a few CO2-relevant controls if they exist:
      - If indoor CO2 is low: increase dosing (U_dos), reduce ventilation (U_nat, U_fan).
      - If indoor CO2 is high: allow more ventilation and reduce dosing.

    This is intentionally simple, to keep code changes minimal.
    """
    idx_dos = _maybe_index(control_names, "U_dos")
    idx_nat = _maybe_index(control_names, "U_nat")
    idx_fan = _maybe_index(control_names, "U_fan")

    lo_C, hi_C = C_in_bounds
    span = max(1e-9, hi_C - lo_C)
    # normalized CO2 level in [0,1]
    c_norm = float(np.clip((C_in - lo_C) / span, 0.0, 1.0))

    # Consider outdoor CO2: if outdoor is low, ventilation is more likely to reduce indoor CO2
    # Make a simple "penalty" when C_out is below C_in (vent tends to decrease C_in).
    vent_penalty = 0.15 if (C_out < C_in) else 0.0

    # Low CO2 => higher dosing, lower ventilation
    if c_norm < 0.35:
        if idx_dos is not None:
            u_i[idx_dos] = _uniform(rng, 0.6, 1.0)
        if idx_nat is not None:
            u_i[idx_nat] = _uniform(rng, 0.0, max(0.25 - vent_penalty, 0.05))
        if idx_fan is not None:
            u_i[idx_fan] = _uniform(rng, 0.0, max(0.35 - vent_penalty, 0.10))

    # Mid CO2 => moderate dosing and ventilation
    elif c_norm < 0.75:
        if idx_dos is not None:
            u_i[idx_dos] = _uniform(rng, 0.2, 0.8)
        if idx_nat is not None:
            u_i[idx_nat] = _uniform(rng, 0.0, max(0.7 - vent_penalty, 0.30))
        if idx_fan is not None:
            u_i[idx_fan] = _uniform(rng, 0.0, max(0.8 - vent_penalty, 0.40))

    # High CO2 => low dosing, allow higher ventilation
    else:
        if idx_dos is not None:
            u_i[idx_dos] = _uniform(rng, 0.0, 0.4)
        if idx_nat is not None:
            u_i[idx_nat] = _uniform(rng, 0.3, 1.0)
        if idx_fan is not None:
            u_i[idx_fan] = _uniform(rng, 0.3, 1.0)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # ===== EDIT IN PYCHARM =====
    num_sample = 60_000
    seed = 0
    dt_override_s = None

    # Relative humidity sampling ranges (0..1)
    RH_in_bounds = (0.20, 1.00)
    RH_out_bounds = (0.20, 1.00)

    # -------------------------
    # CO2 sampling + guards (C unit is your model's C unit, consistent with C_out)
    # -------------------------
    # Recommended “more physical” band for typical greenhouse:
    # - Outdoor ~0.6–1.0 (as you use)
    # - Indoor often ~0.7–2.0 (enrichment)
    C_IN_BOUNDS = (0.7, 2.0)          # sampling range for x0 C_in
    C_OUT_BOUNDS = (0.6, 1.0)         # sampling range for d C_out

    # Hard-guard to avoid hidden clipping/negative CO2 in x1
    C_IN_MIN_GUARD = 0.4              # reject if x0/x1 C_in < this
    C_IN_MAX_GUARD = 5.0              # reject if x0/x1 C_in > this (safety)

    # -------------------------
    # Sampling ranges (x0/u/d). Unknown names fall back to defaults.
    # NOTE:
    # - H_in and H_out are NOT directly sampled anymore.
    # - They are computed via RH * H_sat(T).
    # -------------------------
    x0_bounds = {
        "T_in": (16.0, 25.0),
        "C_in": C_IN_BOUNDS,
        "L_DLI": (0.0, 44.0),
        # "H_in": handled by RH + H_sat(T_in)
    }
    u_bounds = {}  # usually [0,1]
    d_bounds = {
        "T_out": (-5.0, 15.0),
        "C_out": C_OUT_BOUNDS,
        "R_out": (0.0, 800.0),
        # "H_out": handled by RH + H_sat(T_out)
    }

    x_default = (0.0, 1.0)
    u_default = (0.0, 1.0)
    d_default = (0.0, 1.0)

    # Physical guards (hard constraints). If violated => reject sample.
    state_limits = {
        "T_in": (-5.0, 50.0),
        "H_in": (0.0, 40.0),
        "C_in": (C_IN_MIN_GUARD, C_IN_MAX_GUARD),   # tightened
        "L_DLI": (0.0, 60.0),
    }
    # ===========================

    root = Path(__file__).resolve().parents[1]
    out_dir = _clear_random_samples_dir(root)

    model = load_one_step_euler_from_yamls(dt_override_s=dt_override_s)
    rng = np.random.default_rng(seed)

    # Bounds arrays aligned to model name ordering
    bx0 = _bounds_for(model.defs.state_names, x0_bounds, default=x_default)
    bu = _bounds_for(model.defs.control_names, u_bounds, default=u_default)
    bd = _bounds_for(model.defs.disturbance_names, d_bounds, default=d_default)

    # Indices for humidity & temperature (must exist)
    try:
        idx_T_in = model.defs.state_names.index("T_in")
        idx_H_in = model.defs.state_names.index("H_in")
    except ValueError as e:
        raise KeyError(f"State names must include 'T_in' and 'H_in'. Got: {model.defs.state_names}") from e

    try:
        idx_C_in = model.defs.state_names.index("C_in")
    except ValueError as e:
        raise KeyError(f"State names must include 'C_in'. Got: {model.defs.state_names}") from e

    try:
        idx_T_out = model.defs.disturbance_names.index("T_out")
        idx_H_out = model.defs.disturbance_names.index("H_out")
    except ValueError:
        raise KeyError(
            f"Disturbance names must include 'T_out' and 'H_out'. Got: {model.defs.disturbance_names}"
        )

    try:
        idx_C_out = model.defs.disturbance_names.index("C_out")
    except ValueError as e:
        raise KeyError(f"Disturbance names must include 'C_out'. Got: {model.defs.disturbance_names}") from e

    x0 = np.empty((num_sample, model.n_x), dtype=float)
    u = np.empty((num_sample, model.n_u), dtype=float)
    d = np.empty((num_sample, model.n_d), dtype=float)
    x1 = np.empty((num_sample, model.n_x), dtype=float)

    reasons = Counter()
    attempts = 0
    accepted = 0
    report_every = max(1, num_sample // 10)

    rh_in_lo, rh_in_hi = RH_in_bounds
    rh_out_lo, rh_out_hi = RH_out_bounds
    if not (0.0 <= rh_in_lo <= rh_in_hi <= 1.0):
        raise ValueError(f"RH_in_bounds must be within [0,1]. Got: {RH_in_bounds}")
    if not (0.0 <= rh_out_lo <= rh_out_hi <= 1.0):
        raise ValueError(f"RH_out_bounds must be within [0,1]. Got: {RH_out_bounds}")

    while accepted < num_sample:
        attempts += 1

        # -------------------------
        # Sample x0 (states) then overwrite H_in using RH*Hsat(T_in)
        # -------------------------
        x0_i = _sample_row(rng, bx0)

        # Explicit order: sample T_in -> RH_in -> H_in
        T_in = float(x0_i[idx_T_in])
        RH_in = _uniform(rng, rh_in_lo, rh_in_hi)
        x0_i[idx_H_in] = RH_in * _h_sat_g_m3(T_in)

        # CO2 guard for x0 (avoid too-low initial CO2)
        if float(x0_i[idx_C_in]) < C_IN_MIN_GUARD:
            reasons["x0_C_in_below_guard"] += 1
            continue

        r = _validate_vec(x0_i, model.defs.state_names, state_limits, prefix="x0")
        if r is not None:
            reasons[r] += 1
            continue

        # -------------------------
        # Sample u (controls) – start uniform, then apply CO2-aware overwrite
        # -------------------------
        u_i = _sample_row(rng, bu)

        # -------------------------
        # Sample d (disturbances) then overwrite H_out using RH*Hsat(T_out)
        # -------------------------
        d_i = _sample_row(rng, bd)

        # Explicit order: sample T_out -> RH_out -> H_out
        T_out = float(d_i[idx_T_out])
        RH_out = _uniform(rng, rh_out_lo, rh_out_hi)
        d_i[idx_H_out] = RH_out * _h_sat_g_m3(T_out)

        # Apply simple CO2-aware control policy to avoid venting CO2 to negative
        C_in_now = float(x0_i[idx_C_in])
        C_out_now = float(d_i[idx_C_out])
        _apply_co2_aware_controls(
            rng=rng,
            u_i=u_i,
            control_names=model.defs.control_names,
            C_in=C_in_now,
            C_out=C_out_now,
            C_in_bounds=C_IN_BOUNDS,
        )

        # -------------------------
        # Step the simulator
        # -------------------------
        x1_i = model.step(x0_i, u_i, d_i)

        # CO2 guard for x1 (reject “goes negative then clips to 0” behavior)
        c1 = float(x1_i[idx_C_in])
        if c1 < C_IN_MIN_GUARD:
            reasons["x1_C_in_below_guard"] += 1
            continue
        if c1 > C_IN_MAX_GUARD:
            reasons["x1_C_in_above_guard"] += 1
            continue

        r = _validate_vec(x1_i, model.defs.state_names, state_limits, prefix="x1")
        if r is not None:
            reasons[r] += 1
            continue

        # Accept
        x0[accepted] = x0_i
        u[accepted] = u_i
        d[accepted] = d_i
        x1[accepted] = x1_i
        accepted += 1

        if accepted == 1 or accepted % report_every == 0 or accepted == num_sample:
            print(f"Generated {accepted}/{num_sample} samples (attempts: {attempts})")

    dataset_path = out_dir / "dataset.npz"
    meta_path = out_dir / "dataset_meta.json"

    np.savez_compressed(
        dataset_path,
        x_k=x0,
        u_k=u,
        d_k=d,
        x_kp1=x1,
        state_names=np.array(model.defs.state_names, dtype=object),
        control_names=np.array(model.defs.control_names, dtype=object),
        disturbance_names=np.array(model.defs.disturbance_names, dtype=object),
        dt_s=float(model.dt_s),
        seed=int(seed),
    )

    invalid = attempts - num_sample
    invalid_pct = (invalid / attempts * 100.0) if attempts else 0.0
    accept_rate = (num_sample / attempts * 100.0) if attempts else 0.0

    meta = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "num_sample": int(num_sample),
        "attempts": int(attempts),
        "invalid": int(invalid),
        "invalid_pct": float(invalid_pct),
        "accept_rate_pct": float(accept_rate),
        "seed": int(seed),
        "dt_s": float(model.dt_s),
        "state_limits": state_limits,
        "rejection_reasons": dict(reasons),
        "names": {
            "state": model.defs.state_names,
            "control": model.defs.control_names,
            "disturbance": model.defs.disturbance_names,
        },
        "sampling": {
            "x0_bounds": x0_bounds,
            "u_bounds": u_bounds,
            "d_bounds": d_bounds,
            "RH_in_bounds": RH_in_bounds,
            "RH_out_bounds": RH_out_bounds,
            "H_in_rule": "H_in = RH_in * H_sat(T_in)",
            "H_out_rule": "H_out = RH_out * H_sat(T_out)",
            "H_sat_units": "g/m^3",
            "H_sat_method": "Magnus e_s(T) + ideal gas conversion",
            "CO2": {
                "C_in_bounds": C_IN_BOUNDS,
                "C_out_bounds": C_OUT_BOUNDS,
                "C_in_min_guard": C_IN_MIN_GUARD,
                "C_in_max_guard": C_IN_MAX_GUARD,
                "control_bias": "if C_in low => higher U_dos, lower U_nat/U_fan (if present)",
            },
        },
        "model_form": "x_{k+1} = f(x_k, u_k, d_k) (later fit as M,N,O,m)",
        "model_params_snapshot": asdict(model.params),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved dataset to: {dataset_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Invalid samples: {invalid} / {attempts} ({invalid_pct:.2f}%)")
    if reasons:
        print("Top numeric rejection/clipping reasons:")
        for k, v in reasons.most_common(10):
            print(f"  {k}: {v}")
    print(f"Acceptance rate: {accept_rate:.2f}%")

    _print_stats("X0 stats", model.defs.state_names, x0)
    _print_stats("U stats", model.defs.control_names, u)
    _print_stats("D stats", model.defs.disturbance_names, d)
    _print_stats("X1 stats", model.defs.state_names, x1)
