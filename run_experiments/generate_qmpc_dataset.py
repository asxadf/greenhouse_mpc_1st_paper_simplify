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


def _bounds_for(names: list[str], table: dict[str, tuple[float, float]], default=(0.0, 1.0)) -> np.ndarray:
    b = np.empty((len(names), 2), dtype=float)
    for i, n in enumerate(names):
        b[i] = table.get(n, default)
    return b


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


if __name__ == "__main__":
    # ===== EDIT IN PYCHARM =====
    num_sample = 60_000
    seed = 0
    dt_override_s = None

    # Sampling ranges (x0/u/d). Unknown names fall back to defaults.
    x0_bounds = {"T_in": (9.0, 29.0), "H_in": (0.0, 25.5), "C_in": (0.0, 3.7), "L_DLI": (0.0, 44.0)}
    u_bounds = {}  # usually [0,1]
    d_bounds = {"T_out": (-5.0, 15.0), "H_out": (3.0, 9.0), "C_out": (0.6, 1.0), "R_out": (0.0, 800.0)}

    x_default = (0.0, 1.0)
    u_default = (0.0, 1.0)
    d_default = (0.0, 1.0)

    # Physical guards (hard constraints). If violated => reject sample.
    state_limits = {
        "T_in": (-5.0, 50.0),
        "H_in": (0.0, 40.0),
        "C_in": (0.0, 10.0),
        "L_DLI": (0.0, 60.0),
    }
    # ===========================

    root = Path(__file__).resolve().parents[1]
    out_dir = _clear_random_samples_dir(root)

    model = load_one_step_euler_from_yamls(dt_override_s=dt_override_s)
    rng = np.random.default_rng(seed)

    bx0 = _bounds_for(model.defs.state_names, x0_bounds, default=x_default)
    bu = _bounds_for(model.defs.control_names, u_bounds, default=u_default)
    bd = _bounds_for(model.defs.disturbance_names, d_bounds, default=d_default)

    x0 = np.empty((num_sample, model.n_x), dtype=float)
    u = np.empty((num_sample, model.n_u), dtype=float)
    d = np.empty((num_sample, model.n_d), dtype=float)
    x1 = np.empty((num_sample, model.n_x), dtype=float)

    reasons = Counter()
    attempts = 0
    accepted = 0
    report_every = max(1, num_sample // 10)

    while accepted < num_sample:
        attempts += 1

        x0_i = _sample_row(rng, bx0)
        r = _validate_vec(x0_i, model.defs.state_names, state_limits, prefix="x0")
        if r is not None:
            reasons[r] += 1
            continue

        u_i = _sample_row(rng, bu)
        d_i = _sample_row(rng, bd)

        x1_i = model.step(x0_i, u_i, d_i)
        r = _validate_vec(x1_i, model.defs.state_names, state_limits, prefix="x1")
        if r is not None:
            reasons[r] += 1
            continue

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
        "model_form": "x_{k+1} = M x_k + N u_k + O d_k + m",
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
