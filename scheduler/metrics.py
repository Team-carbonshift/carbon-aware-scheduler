"""측정 지표 집계 및 비교 출력."""

from .config import MODES


def aggregate(results):
    n = len(results)
    total_carbon = sum(r["carbon_emitted"] for r in results)
    avg_delay = sum(r["delay"] for r in results) / n if n else 0.0
    violations = sum(1 for r in results if not r["slo_satisfied"])
    slo_violation_rate = violations / n if n else 0.0
    return {
        "n_jobs": n,
        "total_carbon": total_carbon,
        "avg_delay": avg_delay,
        "slo_violation_rate": slo_violation_rate,
    }


def compare_modes(results_by_mode):
    return {mode: aggregate(results) for mode, results in results_by_mode.items()}


def print_comparison(results_by_mode):
    comparison = compare_modes(results_by_mode)
    header = f"{'mode':22s} {'n_jobs':>8s} {'total_carbon':>15s} {'avg_delay(h)':>14s} {'slo_violation':>14s}"
    print(header)
    print("-" * len(header))
    for mode, m in comparison.items():
        label = MODES.get(mode, mode)
        print(f"{label:22s} {m['n_jobs']:8d} {m['total_carbon']:15.2f} {m['avg_delay']:14.3f} {m['slo_violation_rate']:14.4f}")
    return comparison
