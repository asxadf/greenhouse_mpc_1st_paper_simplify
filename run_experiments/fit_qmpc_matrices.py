# run_experiments/fit_qmpc_matrices.py
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _latest_dataset_npz(base_dir: Path) -> Path:
    base_dir = base_dir.resolve()
    candidates = list(base_dir.glob("dataset*.npz")) + list(base_dir.glob("*/dataset*.npz"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No dataset*.npz found under: {base_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _load_dataset(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clear_dir_safely(root: Path, rel_parts: tuple[str, ...]) -> Path:
    """
    Clears <root>/<rel_parts...> and returns the resolved path.

    Safety: refuses to delete if the resolved dir does not end with rel_parts.
    """
    target = (root / Path(*rel_parts)).resolve()
    if target.parts[-len(rel_parts) :] != rel_parts:
        raise RuntimeError(f"Refusing to delete unexpected directory: {target}")

    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return target


def _print_stats(title: str, names: list[str], mat: np.ndarray) -> None:
    print(f"{title}:")
    width = max(len(n) for n in names) if names else 10
    for i, n in enumerate(names):
        col = mat[:, i]
        print(
            f"  {n:>{width}}: "
            f"min={np.min(col):9.4f} max={np.max(col):9.4f} mean={np.mean(col):9.4f} std={np.std(col):9.4f}"
        )


def _metrics_table(title: str, names: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    r2 = np.array([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])], dtype=float)
    agg_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    print(title)
    print(f"{'State':>12} | {'RMSE':>10} | {'MAE':>10} | {'R^2':>8}")
    print("-" * 49)
    for i, n in enumerate(names):
        print(f"{n:>12} | {rmse[i]:10.4f} | {mae[i]:10.4f} | {r2[i]:8.4f}")
    print(f"Aggregate RMSE (all states): {agg_rmse:.4f}")

    return {
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "r2": r2.tolist(),
        "aggregate_rmse": agg_rmse,
    }


def _write_labeled_matrix_csv(
    path: Path,
    mat: np.ndarray,
    row_label: str,
    row_names: list[str],
    col_names: list[str],
) -> None:
    if mat.shape != (len(row_names), len(col_names)):
        raise ValueError(
            f"Shape mismatch for {path.name}: mat={mat.shape}, rows={len(row_names)}, cols={len(col_names)}"
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row_label] + list(col_names))
        for r_name, row in zip(row_names, mat, strict=True):
            w.writerow([r_name] + [f"{float(v):.16g}" for v in row])


def _write_labeled_vector_csv(
    path: Path,
    vec: np.ndarray,
    row_label: str,
    row_names: list[str],
    col_name: str,
) -> None:
    v = np.asarray(vec, dtype=float).reshape(-1)
    if len(v) != len(row_names):
        raise ValueError(f"Length mismatch for {path.name}: vec={len(v)}, rows={len(row_names)}")

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row_label, col_name])
        for r_name, val in zip(row_names, v, strict=True):
            w.writerow([r_name, f"{float(val):.16g}"])


def _export_csv_exports(
    out_dir: Path,
    M: np.ndarray,
    N: np.ndarray,
    O: np.ndarray,
    m: np.ndarray,
    state_names: list[str],
    control_names: list[str],
    disturbance_names: list[str],
) -> dict:
    exp_dir = out_dir / "readable_exports"
    _ensure_dir(exp_dir)

    paths = {
        "M": exp_dir / "M_matrix.csv",
        "N": exp_dir / "N_matrix.csv",
        "O": exp_dir / "O_matrix.csv",
        "m": exp_dir / "m_bias.csv",
    }

    _write_labeled_matrix_csv(
        path=paths["M"],
        mat=M,
        row_label="x_state",
        row_names=state_names,
        col_names=state_names,
    )
    _write_labeled_matrix_csv(
        path=paths["N"],
        mat=N,
        row_label="x_state",
        row_names=state_names,
        col_names=control_names,
    )
    _write_labeled_matrix_csv(
        path=paths["O"],
        mat=O,
        row_label="x_state",
        row_names=state_names,
        col_names=disturbance_names,
    )
    _write_labeled_vector_csv(
        path=paths["m"],
        vec=m,
        row_label="x_state",
        row_names=state_names,
        col_name="bias",
    )

    return {k: str(v) for k, v in paths.items()}


def _spectral_radius(M: np.ndarray) -> float:
    eig = np.linalg.eigvals(M)
    return float(np.max(np.abs(eig)))


def _fit_affine_ls_with_scaling(Phi: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Y = Phi @ B^T + b with scaling for numeric stability.
    Returns unscaled (B, b): B (n_y, n_feat), b (n_y,)
    """
    sx = StandardScaler(with_mean=True, with_std=True)
    sy = StandardScaler(with_mean=True, with_std=True)

    Xs = sx.fit_transform(Phi)
    Ys = sy.fit_transform(Y)

    reg = LinearRegression(fit_intercept=True)
    reg.fit(Xs, Ys)

    coef_s = reg.coef_  # (n_y, n_feat)
    intercept_s = reg.intercept_  # (n_y,)

    x_mean, x_std = sx.mean_, sx.scale_
    y_mean, y_std = sy.mean_, sy.scale_

    B = (coef_s * y_std[:, None]) / x_std[None, :]
    b = y_mean + y_std * intercept_s - (x_mean @ B.T)

    return B, b


if __name__ == "__main__":
    # ===== EDIT IN PYCHARM =====
    root = Path(__file__).resolve().parents[1]

    random_samples_dir = root / "data/for_fitting_qmpc/random_samples"
    results_dir_rel = ("data", "for_fitting_qmpc", "fitting_results")

    test_size = 0.2
    random_state = 0
    # ===========================

    dataset_path = _latest_dataset_npz(random_samples_dir)
    print(f"Info: using most recent dataset under random_samples: {dataset_path} (from {random_samples_dir})")

    ds = _load_dataset(dataset_path)

    x0 = np.asarray(ds["x_k"], dtype=float)
    u = np.asarray(ds["u_k"], dtype=float)
    d = np.asarray(ds["d_k"], dtype=float)
    x1 = np.asarray(ds["x_kp1"], dtype=float)

    if not (x0.ndim == u.ndim == d.ndim == x1.ndim == 2):
        raise ValueError("Expected 2D arrays for x_k, u_k, d_k, x_kp1")
    if not (len(x0) == len(u) == len(d) == len(x1)):
        raise ValueError("x_k, u_k, d_k, x_kp1 must have the same number of rows")

    state_names = [str(s) for s in ds.get("state_names", np.array([f"x{i}" for i in range(x0.shape[1])], dtype=object)).tolist()]
    control_names = [str(s) for s in ds.get("control_names", np.array([f"u{i}" for i in range(u.shape[1])], dtype=object)).tolist()]
    disturbance_names = [str(s) for s in ds.get("disturbance_names", np.array([f"d{i}" for i in range(d.shape[1])], dtype=object)).tolist()]

    _print_stats("X0 stats", state_names, x0)
    _print_stats("U stats", control_names, u)
    _print_stats("D stats", disturbance_names, d)
    _print_stats("X1 stats", state_names, x1)

    Y = x1
    Phi = np.hstack([x0, u, d])

    idx = np.arange(len(Phi))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)

    Phi_tr, Phi_te = Phi[idx_tr], Phi[idx_te]
    Y_tr, Y_te = Y[idx_tr], Y[idx_te]
    x0_tr, x0_te = x0[idx_tr], x0[idx_te]
    x1_tr, x1_te = x1[idx_tr], x1[idx_te]

    B, b = _fit_affine_ls_with_scaling(Phi_tr, Y_tr)
    n_x = x0.shape[1]
    n_u = u.shape[1]

    M = B[:, :n_x]
    N = B[:, n_x : n_x + n_u]
    O = B[:, n_x + n_u :]
    m = b

    Yhat_te = Phi_te @ B.T + b
    Yhat_tr = Phi_tr @ B.T + b
    x1hat_te = Yhat_te
    x1hat_tr = Yhat_tr

    _metrics_table("Test metrics (x1 prediction):", state_names, Y_te, Yhat_te)
    print()
    _metrics_table("Test metrics (x1 prediction, direct):", state_names, x1_te, x1hat_te)
    print()

    rho = _spectral_radius(M)
    print(f"M spectral radius: {rho:.4f}")
    if rho > 1.0:
        print("Warning: M spectral radius > 1.0 (unstable dynamics).")
    print()

    _metrics_table("Train metrics (x1 prediction):", state_names, Y_tr, Yhat_tr)
    print()
    _metrics_table("Train metrics (x1 prediction, direct):", state_names, x1_tr, x1hat_tr)

    out_dir = _clear_dir_safely(root, results_dir_rel)

    mats_path = out_dir / "fitted_matrices.npz"
    meta_path = out_dir / "fitted_matrices_meta.json"

    np.savez_compressed(
        mats_path,
        M=M,
        N=N,
        O=O,
        m=m,
        dt_s=np.array(float(ds.get("dt_s", np.nan))),
        state_names=np.array(state_names, dtype=object),
        control_names=np.array(control_names, dtype=object),
        disturbance_names=np.array(disturbance_names, dtype=object),
        dataset_path=np.array(str(dataset_path), dtype=object),
        train_idx=idx_tr,
        test_idx=idx_te,
    )

    export_paths = _export_csv_exports(
        out_dir=out_dir,
        M=M,
        N=N,
        O=O,
        m=m,
        state_names=state_names,
        control_names=control_names,
        disturbance_names=disturbance_names,
    )
    print("Readable CSV exports:")
    for _, p in export_paths.items():
        print(f"  {p}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "created_at": stamp,
        "dataset_path": str(dataset_path),
        "random_samples_dir": str(random_samples_dir),
        "results_dir": str(out_dir),
        "n_samples": int(len(x0)),
        "n_train": int(len(idx_tr)),
        "n_test": int(len(idx_te)),
        "n_x": int(n_x),
        "n_u": int(n_u),
        "n_d": int(d.shape[1]),
        "spectral_radius_M": float(rho),
        "exports": export_paths,
        "scores": {
            "test_x1_mse": float(mean_squared_error(Y_te, Yhat_te)),
            "test_x1_mae": float(mean_absolute_error(Y_te, Yhat_te)),
            "train_x1_mse": float(mean_squared_error(Y_tr, Yhat_tr)),
            "train_x1_mae": float(mean_absolute_error(Y_tr, Yhat_tr)),
            "test_x1_r2_avg": float(r2_score(Y_te, Yhat_te, multioutput="uniform_average")),
            "train_x1_r2_avg": float(r2_score(Y_tr, Yhat_tr, multioutput="uniform_average")),
        },
        "saved": {
            "matrices": str(mats_path),
            "metadata": str(meta_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved matrices to: {mats_path}")
    print(f"Saved metadata to: {meta_path}")