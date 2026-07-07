#!/usr/bin/env python3
"""Compare CPU and GPU deskew outputs with intensity and structure metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.optimize import curve_fit
import tifffile
import yaml
import zarr


FWHM_FACTOR = float(2.0 * np.sqrt(2.0 * np.log(2.0)))


def _image_stem(path: str | Path) -> str:
    text = str(path)
    if text.lower().endswith(".ome.zarr"):
        return Path(text[:-len(".ome.zarr")]).name
    return Path(text).stem


def load_yaml_params(path: Path) -> dict[str, Any]:
    """Read a Nextflow/Astrocyte params YAML file."""
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(params, dict):
        raise ValueError(f"Expected mapping in params file: {path}")
    return params


def _first_input_path(params: dict[str, Any], *, params_path: Path) -> str:
    value = params.get("input")
    if value is None:
        value = params.get("image_path")
    if isinstance(value, list):
        if not value:
            raise ValueError(f"No input entries found in params file: {params_path}")
        value = value[0]
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing input or image_path in params file: {params_path}")
    return str(value)


def _output_path_from_params(params: dict[str, Any], *, params_path: Path) -> Path:
    output_dir = params.get("output_dir")
    if output_dir is None or str(output_dir).strip() == "":
        raise ValueError(f"Missing output_dir in params file: {params_path}")
    input_path = _first_input_path(params, params_path=params_path)
    output_suffix = ".ome.zarr" if "input" in params else Path(input_path).suffix
    if not output_suffix:
        output_suffix = ".ome.zarr"
    return Path(str(output_dir)) / "Top_shear" / f"{_image_stem(input_path)}{output_suffix}"


def comparison_from_yaml_files(cpu_params_path: Path, gpu_params_path: Path) -> dict[str, Any]:
    """Resolve comparison inputs from CPU and GPU workflow params YAML files."""
    cpu_params = load_yaml_params(cpu_params_path)
    gpu_params = load_yaml_params(gpu_params_path)
    lateral_pixel_size = cpu_params.get("dx", gpu_params.get("dx", 1.0))
    return {
        "reference_path": _output_path_from_params(cpu_params, params_path=cpu_params_path),
        "candidate_path": _output_path_from_params(gpu_params, params_path=gpu_params_path),
        "lateral_pixel_size": float(lateral_pixel_size),
    }


def sample_indices(size: int, count: int) -> list[int]:
    """Return up to ``count`` evenly spaced indices including endpoints."""
    size = int(size)
    count = max(1, int(count))
    if size <= 0:
        return []
    if count >= size:
        return list(range(size))
    return [int(v) for v in np.rint(np.linspace(0, size - 1, count)).astype(int)]


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return array[np.isfinite(array)]


def gaussian_intensity_stats(values: np.ndarray, *, ignore_zero: bool = False) -> dict[str, float | int]:
    """Fit a simple Gaussian brightness model and report mean, sigma, and FWHM."""
    finite = _finite_values(values)
    if ignore_zero:
        finite = finite[finite != 0.0]
    if finite.size == 0:
        return {"count": 0, "mean": float("nan"), "sigma": float("nan"), "fwhm": float("nan")}
    sigma = float(np.std(finite))
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "sigma": sigma,
        "fwhm": float(FWHM_FACTOR * sigma),
    }


def gaussian_profile(
    x: np.ndarray, amplitude: float, x_offset: float, sigma: float, y_offset: float
) -> np.ndarray:
    """Return the same Gaussian line-profile model used by ClearEx."""
    return amplitude * np.exp(-((x - x_offset) ** 2) / (2 * sigma**2)) + y_offset


def _failed_line_profile_fit(count: int) -> dict[str, float | int]:
    return {
        "count": int(count),
        "fwhm": float("nan"),
        "amplitude": float("nan"),
        "x_offset": float("nan"),
        "sigma": float("nan"),
        "y_offset": float("nan"),
        "r2": float("nan"),
    }


def fit_line_profile(
    x_axis: np.ndarray,
    line_profile: np.ndarray,
    lateral_pixel_size: float = 1.0,
) -> dict[str, float | int]:
    """Fit a ClearEx-style Gaussian line profile and report FWHM plus R^2."""
    x = np.asarray(x_axis, dtype=np.float64).ravel()
    y = np.asarray(line_profile, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 4:
        return _failed_line_profile_fit(int(x.size))

    try:
        # Match ClearEx's initial guesses in clearex.fit.points.fit_line_profile.
        initial_amplitude = float(np.max(y) - np.min(y))
        initial_x_offset = float(np.argmax(y))
        initial_sigma = 3.0
        initial_y_offset = float(np.min(y))
        params, _ = curve_fit(
            gaussian_profile,
            x,
            y,
            p0=[initial_amplitude, initial_x_offset, initial_sigma, initial_y_offset],
            maxfev=10000,
        )
    except (RuntimeError, TypeError, ValueError, FloatingPointError):
        return _failed_line_profile_fit(int(x.size))

    amplitude, x_offset, sigma, y_offset = [float(value) for value in params]
    predicted = gaussian_profile(x, amplitude, x_offset, sigma, y_offset)
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - float(np.mean(y))) ** 2))
    if total_sum == 0.0:
        r2 = 1.0 if np.allclose(y, predicted) else float("nan")
    else:
        r2 = float(1.0 - residual_sum / total_sum)
    return {
        "count": int(x.size),
        "fwhm": float(FWHM_FACTOR * sigma * float(lateral_pixel_size)),
        "amplitude": amplitude,
        "x_offset": x_offset,
        "sigma": sigma,
        "y_offset": y_offset,
        "r2": r2,
    }


def normalized_cross_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute Pearson-style normalized cross-correlation over finite pixels."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cand = np.asarray(candidate, dtype=np.float64).ravel()
    valid = np.isfinite(ref) & np.isfinite(cand)
    if int(np.count_nonzero(valid)) == 0:
        return float("nan")
    ref = ref[valid] - float(np.mean(ref[valid]))
    cand = cand[valid] - float(np.mean(cand[valid]))
    denominator = float(np.sqrt(np.sum(ref * ref) * np.sum(cand * cand)))
    if denominator == 0.0:
        return 1.0 if np.allclose(ref, cand) else float("nan")
    return float(np.sum(ref * cand) / denominator)


def global_ssim(reference: np.ndarray, candidate: np.ndarray, *, data_range: float | None = None) -> float:
    """Compute whole-image SSIM for one 2-D page."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(cand)
    if int(np.count_nonzero(valid)) == 0:
        return float("nan")
    ref = ref[valid]
    cand = cand[valid]
    if data_range is None:
        maximum = float(max(np.max(ref), np.max(cand)))
        minimum = float(min(np.min(ref), np.min(cand)))
        data_range = max(1.0, maximum - minimum)
    c1 = float((0.01 * float(data_range)) ** 2)
    c2 = float((0.03 * float(data_range)) ** 2)
    mu_ref = float(np.mean(ref))
    mu_cand = float(np.mean(cand))
    var_ref = float(np.mean((ref - mu_ref) ** 2))
    var_cand = float(np.mean((cand - mu_cand) ** 2))
    cov = float(np.mean((ref - mu_ref) * (cand - mu_cand)))
    numerator = (2.0 * mu_ref * mu_cand + c1) * (2.0 * cov + c2)
    denominator = (mu_ref**2 + mu_cand**2 + c1) * (var_ref + var_cand + c2)
    return float(numerator / denominator) if denominator != 0.0 else float("nan")


def _peak_anchor(page: np.ndarray) -> tuple[int, int] | None:
    array = np.asarray(page, dtype=np.float64)
    if array.ndim != 2:
        return None
    finite = np.isfinite(array)
    if int(np.count_nonzero(finite)) == 0:
        return None
    filled = np.where(finite, array, -np.inf)
    y_index, x_index = np.unravel_index(int(np.argmax(filled)), filled.shape)
    return int(y_index), int(x_index)


def line_profile_metrics(
    reference_page: np.ndarray,
    candidate_page: np.ndarray,
    *,
    lateral_pixel_size: float = 1.0,
) -> list[dict[str, Any]]:
    """Compare x/y Gaussian line profiles through the reference page peak."""
    reference = np.asarray(reference_page)
    candidate = np.asarray(candidate_page)
    if reference.shape != candidate.shape or reference.ndim != 2:
        return []
    anchor = _peak_anchor(reference)
    if anchor is None:
        return []

    y_index, x_index = anchor
    profile_specs = (
        ("x", y_index, reference[y_index, :], candidate[y_index, :]),
        ("y", x_index, reference[:, x_index], candidate[:, x_index]),
    )
    rows: list[dict[str, Any]] = []
    for profile_axis, profile_index, reference_profile, candidate_profile in profile_specs:
        x_axis = np.arange(reference_profile.size, dtype=np.float64)
        reference_fit = fit_line_profile(
            x_axis, reference_profile, lateral_pixel_size=lateral_pixel_size
        )
        candidate_fit = fit_line_profile(
            x_axis, candidate_profile, lateral_pixel_size=lateral_pixel_size
        )
        rows.append({
            "axis": profile_axis,
            "index": int(profile_index),
            "anchor": {"y": y_index, "x": x_index},
            "ncc": normalized_cross_correlation(reference_profile, candidate_profile),
            "ssim": global_ssim(reference_profile, candidate_profile),
            "cpu": reference_fit,
            "gpu": candidate_fit,
            "fwhm_delta": float(candidate_fit["fwhm"] - reference_fit["fwhm"]),
            "r2_delta": float(candidate_fit["r2"] - reference_fit["r2"]),
        })
    return rows


def _nan_summary(values: list[float]) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _open_volume(path: Path, *, level: str = "0") -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Comparison input not found: {path}. Run both deskew workflows first "
            "or pass explicit --cpu/--gpu output paths."
        )
    if path.name.lower().endswith(".ome.zarr"):
        return zarr.open(str(path / str(level)), mode="r")
    return tifffile.memmap(str(path), mode="r")


def _slice_page(volume: Any, axis: int, index: int) -> np.ndarray:
    selection: list[Any] = [slice(None), slice(None), slice(None)]
    selection[int(axis)] = int(index)
    return np.asarray(volume[tuple(selection)])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compare_outputs(
    *,
    reference_path: Path,
    candidate_path: Path,
    output_json: Optional[Path] = None,
    sample_axis: int = 0,
    sample_count: int = 32,
    level: str = "0",
    ignore_zero: bool = False,
    lateral_pixel_size: float = 1.0,
) -> dict[str, Any]:
    """Compare two 3-D deskew outputs and write compact JSON metrics."""
    reference = _open_volume(reference_path, level=level)
    candidate = _open_volume(candidate_path, level=level)
    reference_shape = tuple(int(v) for v in reference.shape)
    candidate_shape = tuple(int(v) for v in candidate.shape)
    if reference_shape != candidate_shape:
        raise ValueError(
            f"Output shapes differ: reference={reference_shape}, candidate={candidate_shape}"
        )
    if len(reference_shape) != 3:
        raise ValueError(f"Expected 3-D outputs, got shape {reference_shape}")

    axis = int(sample_axis)
    if axis < 0 or axis >= 3:
        raise ValueError(f"sample_axis must be 0, 1, or 2; got {sample_axis}")

    indices = sample_indices(reference_shape[axis], int(sample_count))
    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    all_reference_values: list[np.ndarray] = []
    all_candidate_values: list[np.ndarray] = []

    for index in indices:
        reference_page = _slice_page(reference, axis, index)
        candidate_page = _slice_page(candidate, axis, index)
        reference_stats = gaussian_intensity_stats(reference_page, ignore_zero=ignore_zero)
        candidate_stats = gaussian_intensity_stats(candidate_page, ignore_zero=ignore_zero)
        page_profiles = line_profile_metrics(
            reference_page,
            candidate_page,
            lateral_pixel_size=float(lateral_pixel_size),
        )
        profile_rows.extend(page_profiles)
        rows.append({
            "i": int(index),
            "ncc": normalized_cross_correlation(reference_page, candidate_page),
            "ssim": global_ssim(reference_page, candidate_page),
            "cpu_fwhm": reference_stats["fwhm"],
            "gpu_fwhm": candidate_stats["fwhm"],
            "line_profiles": page_profiles,
        })
        all_reference_values.append(np.asarray(reference_page).ravel())
        all_candidate_values.append(np.asarray(candidate_page).ravel())

    reference_values = np.concatenate(all_reference_values) if all_reference_values else np.asarray([])
    candidate_values = np.concatenate(all_candidate_values) if all_candidate_values else np.asarray([])
    reference_stats = gaussian_intensity_stats(reference_values, ignore_zero=ignore_zero)
    candidate_stats = gaussian_intensity_stats(candidate_values, ignore_zero=ignore_zero)
    ncc_values = [row["ncc"] for row in rows]
    ssim_values = [row["ssim"] for row in rows]
    summary = {
        "shape": reference_shape,
        "axis": axis,
        "samples": len(indices),
        "fwhm": {
            "cpu": reference_stats["fwhm"],
            "gpu": candidate_stats["fwhm"],
            "delta": float(candidate_stats["fwhm"] - reference_stats["fwhm"]),
        },
        "brightness": {
            "cpu_mean": reference_stats["mean"],
            "gpu_mean": candidate_stats["mean"],
            "delta": float(candidate_stats["mean"] - reference_stats["mean"]),
        },
        "ncc": {
            "mean": float(np.nanmean(ncc_values)) if ncc_values else float("nan"),
            "min": float(np.nanmin(ncc_values)) if ncc_values else float("nan"),
        },
        "ssim": {
            "mean": float(np.nanmean(ssim_values)) if ssim_values else float("nan"),
            "min": float(np.nanmin(ssim_values)) if ssim_values else float("nan"),
        },
        "line_profiles": {
            "count": len(profile_rows),
            "ncc": _nan_summary([float(row["ncc"]) for row in profile_rows]),
            "ssim": _nan_summary([float(row["ssim"]) for row in profile_rows]),
            "cpu_r2": _nan_summary([float(row["cpu"]["r2"]) for row in profile_rows]),
            "gpu_r2": _nan_summary([float(row["gpu"]["r2"]) for row in profile_rows]),
            "cpu_fwhm": _nan_summary([float(row["cpu"]["fwhm"]) for row in profile_rows]),
            "gpu_fwhm": _nan_summary([float(row["gpu"]["fwhm"]) for row in profile_rows]),
            "fwhm_delta": _nan_summary([float(row["fwhm_delta"]) for row in profile_rows]),
            "r2_delta": _nan_summary([float(row["r2_delta"]) for row in profile_rows]),
        },
        "pages": rows,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(_json_safe(summary), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return summary


def _resolve_cli_comparison(args: argparse.Namespace) -> dict[str, Any]:
    using_direct_paths = args.cpu is not None or args.gpu is not None
    using_param_files = args.cpu_params is not None or args.gpu_params is not None
    if using_direct_paths and using_param_files:
        raise ValueError("Use either --cpu/--gpu or --cpu_params/--gpu_params, not both.")
    if using_param_files:
        if args.cpu_params is None or args.gpu_params is None:
            raise ValueError("--cpu_params and --gpu_params must be provided together.")
        return comparison_from_yaml_files(args.cpu_params, args.gpu_params)
    if args.cpu is None or args.gpu is None:
        raise ValueError("Provide either --cpu and --gpu, or --cpu_params and --gpu_params.")
    return {
        "reference_path": args.cpu,
        "candidate_path": args.gpu,
        "lateral_pixel_size": 1.0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, help="CPU output TIFF or OME-Zarr")
    parser.add_argument("--gpu", type=Path, help="GPU output TIFF or OME-Zarr")
    parser.add_argument("--cpu_params", type=Path, help="CPU workflow params YAML")
    parser.add_argument("--gpu_params", type=Path, help="GPU workflow params YAML")
    parser.add_argument("--output_json", type=Path, help="Optional path to write compact JSON")
    parser.add_argument("--sample_axis", type=int, default=0, help="Axis to sample pages along; default 0")
    parser.add_argument("--sample_count", type=int, default=32, help="Number of pages to sample")
    parser.add_argument("--level", default="0", help="OME-Zarr pyramid level to compare")
    parser.add_argument("--ignore_zero", action="store_true", help="Ignore zero-valued pixels in Gaussian stats")
    parser.add_argument(
        "--lateral_pixel_size",
        type=float,
        default=None,
        help="Lateral pixel size used to scale Gaussian line-profile FWHM",
    )
    args = parser.parse_args(argv)
    comparison = _resolve_cli_comparison(args)
    lateral_pixel_size = (
        float(args.lateral_pixel_size)
        if args.lateral_pixel_size is not None
        else float(comparison["lateral_pixel_size"])
    )

    summary = compare_outputs(
        reference_path=comparison["reference_path"],
        candidate_path=comparison["candidate_path"],
        output_json=args.output_json,
        sample_axis=args.sample_axis,
        sample_count=args.sample_count,
        level=str(args.level),
        ignore_zero=bool(args.ignore_zero),
        lateral_pixel_size=lateral_pixel_size,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
