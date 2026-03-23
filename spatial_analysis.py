from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Coordinate = Tuple[float, float]
EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between two lat/lon points in kilometers."""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def compute_spatial_errors(
    predictions: Sequence[Coordinate],
    ground_truth: Sequence[Coordinate],
    outlier_threshold_km: float = 100.0,
) -> List[float]:
    """Compute per-sample spatial error (km) and print outlier warnings."""
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, ground_truth={len(ground_truth)}"
        )

    errors: List[float] = []
    for idx, ((pred_lat, pred_lon), (true_lat, true_lon)) in enumerate(
        zip(predictions, ground_truth)
    ):
        error_km = haversine(pred_lat, pred_lon, true_lat, true_lon)
        errors.append(error_km)

        if error_km > outlier_threshold_km:
            print(f"Outlier detected: user index {idx}, error {error_km:.3f} km")

    return errors


def compute_statistics(errors: Sequence[float]) -> Dict[str, float]:
    """Return summary statistics for spatial errors."""
    if not errors:
        raise ValueError("errors is empty; cannot compute statistics")

    return {
        "mean_error": statistics.fmean(errors),
        "median_error": statistics.median(errors),
        "min_error": min(errors),
        "max_error": max(errors),
    }


def compute_distribution(errors: Sequence[float]) -> Dict[str, float]:
    """Return percentage distribution across fixed spatial-error bins."""
    if not errors:
        raise ValueError("errors is empty; cannot compute distribution")

    total = len(errors)
    count_lt_1 = sum(1 for e in errors if e < 1)
    count_1_3 = sum(1 for e in errors if 1 <= e < 3)
    count_3_10 = sum(1 for e in errors if 3 <= e <= 10)
    count_gt_10 = sum(1 for e in errors if e > 10)

    return {
        "<1 km": (count_lt_1 / total) * 100.0,
        "1-3 km": (count_1_3 / total) * 100.0,
        "3-10 km": (count_3_10 / total) * 100.0,
        ">10 km": (count_gt_10 / total) * 100.0,
    }


def print_statistics(stats: Dict[str, float]) -> None:
    """Print spatial error summary in a presentation-friendly format."""
    print(f"Mean Spatial Error: {stats['mean_error']:.3f} km")
    print(f"Median Spatial Error: {stats['median_error']:.3f} km")
    print(f"Min Error: {stats['min_error']:.3f} km")
    print(f"Max Error: {stats['max_error']:.3f} km")


def print_distribution(distribution: Dict[str, float]) -> None:
    """Print percentage distribution of spatial errors."""
    print(f"<1 km     : {distribution['<1 km']:.2f}%")
    print(f"1-3 km    : {distribution['1-3 km']:.2f}%")
    print(f"3-10 km   : {distribution['3-10 km']:.2f}%")
    print(f">10 km    : {distribution['>10 km']:.2f}%")


def plot_error_histogram(errors: Sequence[float], bins: int = 30) -> None:
    """Plot histogram of spatial errors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Install it with: pip install matplotlib"
        ) from exc

    plt.figure(figsize=(9, 5))
    plt.hist(errors, bins=bins, edgecolor="black", alpha=0.8)
    plt.title("Spatial Error Distribution")
    plt.xlabel("Distance Error (km)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()


def run_spatial_analysis(
    predictions: Sequence[Coordinate],
    ground_truth: Sequence[Coordinate],
    plot_histogram: bool = False,
) -> Dict[str, object]:
    """End-to-end analysis helper for direct integration with existing pipelines."""
    errors = compute_spatial_errors(predictions, ground_truth)
    stats = compute_statistics(errors)
    distribution = compute_distribution(errors)

    print_statistics(stats)
    print_distribution(distribution)

    if plot_histogram:
        plot_error_histogram(errors)

    return {
        "errors": list(errors),
        "statistics": stats,
        "distribution": distribution,
    }


def _load_coords(path: Path) -> List[Coordinate]:
    """Load coordinate pairs from JSON: [[lat, lon], ...] or [{"lat":..,"lon":..}, ...]."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in {path}")

    coords: List[Coordinate] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            if "lat" not in item or "lon" not in item:
                raise ValueError(f"Item {i} in {path} missing 'lat' or 'lon'")
            lat = float(item["lat"])
            lon = float(item["lon"])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            lat = float(item[0])
            lon = float(item[1])
        else:
            raise ValueError(
                f"Invalid item format at index {i} in {path}; expected [lat, lon] or {{'lat', 'lon'}}"
            )
        coords.append((lat, lon))

    return coords


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spatial error analysis for mobility prediction")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Path to JSON file containing predicted coordinates",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Path to JSON file containing ground-truth coordinates",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot histogram of spatial errors",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.predictions and args.ground_truth:
        predictions = _load_coords(args.predictions)
        ground_truth = _load_coords(args.ground_truth)
    else:
        # Minimal demo data for quick smoke test when no files are provided.
        predictions = [(12.9716, 77.5946), (28.7041, 77.1025), (19.0760, 72.8777)]
        ground_truth = [(12.9721, 77.5933), (28.6139, 77.2090), (19.0728, 72.8826)]
        print("Using built-in demo data. Provide --predictions and --ground-truth to analyze your dataset.")

    run_spatial_analysis(predictions, ground_truth, plot_histogram=args.plot)


if __name__ == "__main__":
    main()
