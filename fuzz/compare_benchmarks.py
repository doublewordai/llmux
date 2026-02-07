#!/usr/bin/env python3
"""Compare two benchmark reports side-by-side."""

import json
import sys
import os


def load_report(path):
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        # Auto-find latest two reports
        results_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
        if os.path.isdir(results_dir):
            reports = sorted(
                [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".json")],
                key=os.path.getmtime,
            )
            if len(reports) >= 2:
                a_path, b_path = reports[-2], reports[-1]
            else:
                print("Usage: compare_benchmarks.py <report_a.json> <report_b.json>")
                sys.exit(1)
        else:
            print("Usage: compare_benchmarks.py <report_a.json> <report_b.json>")
            sys.exit(1)
    else:
        a_path, b_path = sys.argv[1], sys.argv[2]

    a = load_report(a_path)
    b = load_report(b_path)

    label_a = a["label"]
    label_b = b["label"]

    print(f"Comparing: {label_a} vs {label_b}")
    print(f"  A: {os.path.basename(a_path)}")
    print(f"  B: {os.path.basename(b_path)}")
    print()

    # Index profiles by name
    a_profiles = {p["profile"]: p for p in a["profiles"]}
    b_profiles = {p["profile"]: p for p in b["profiles"]}

    all_profiles = list(dict.fromkeys(list(a_profiles.keys()) + list(b_profiles.keys())))

    def delta(va, vb, lower_better=True):
        if va == 0 and vb == 0:
            return "  ="
        if va == 0:
            return "  N/A"
        pct = (vb - va) / abs(va) * 100
        direction = "better" if (pct < 0) == lower_better else "worse"
        return f"{pct:+.1f}% ({direction})"

    # Per-profile comparison
    header = f"{'Profile':<20} {'Metric':<15} {label_a:>12} {label_b:>12} {'Delta':>20}"
    print(header)
    print("-" * len(header))

    totals_a = {"switches": 0, "switch_time_s": 0, "requests": 0, "wall_time_s": 0, "avg_wait_weighted": 0, "wait_count": 0}
    totals_b = {"switches": 0, "switch_time_s": 0, "requests": 0, "wall_time_s": 0, "avg_wait_weighted": 0, "wait_count": 0}

    for profile in all_profiles:
        pa = a_profiles.get(profile)
        pb = b_profiles.get(profile)
        if not pa or not pb:
            print(f"{profile:<20} (missing in one report)")
            continue

        for key, label, lower in [
            ("switches", "Switches", True),
            ("switch_time_s", "Switch Time", True),
            ("gpu_serving_pct", "GPU Serving %", False),
            ("avg_wait_s", "Avg Wait", True),
            ("requests", "Requests", False),
        ]:
            va = float(pa.get(key, 0))
            vb = float(pb.get(key, 0))
            d = delta(va, vb, lower_better=lower)
            pfx = profile if key == "switches" else ""
            print(f"{pfx:<20} {label:<15} {va:>12.1f} {vb:>12.1f} {d:>20}")

        # Accumulate totals
        for t, p in [(totals_a, pa), (totals_b, pb)]:
            t["switches"] += p["switches"]
            t["switch_time_s"] += p["switch_time_s"]
            t["requests"] += p["requests"]
            t["wall_time_s"] += p["wall_time_s"]
            t["avg_wait_weighted"] += p["avg_wait_s"] * p["requests"]
            t["wait_count"] += p["requests"]

        print()

    # Overall summary
    print("=" * len(header))
    print("OVERALL SUMMARY")
    print()

    for metric, key, lower in [
        ("Total Switches", "switches", True),
        ("Total Switch Time", "switch_time_s", True),
        ("Total Requests", "requests", False),
    ]:
        va = totals_a[key]
        vb = totals_b[key]
        d = delta(va, vb, lower_better=lower)
        print(f"  {metric:<25} {va:>10.1f} {vb:>10.1f} {d:>20}")

    # Overall GPU serving
    gpu_a = (1 - totals_a["switch_time_s"] / totals_a["wall_time_s"]) * 100 if totals_a["wall_time_s"] > 0 else 0
    gpu_b = (1 - totals_b["switch_time_s"] / totals_b["wall_time_s"]) * 100 if totals_b["wall_time_s"] > 0 else 0
    d = delta(gpu_a, gpu_b, lower_better=False)
    print(f"  {'Overall GPU Serving %':<25} {gpu_a:>10.1f} {gpu_b:>10.1f} {d:>20}")

    # Overall avg wait
    avg_a = totals_a["avg_wait_weighted"] / totals_a["wait_count"] if totals_a["wait_count"] > 0 else 0
    avg_b = totals_b["avg_wait_weighted"] / totals_b["wait_count"] if totals_b["wait_count"] > 0 else 0
    d = delta(avg_a, avg_b, lower_better=True)
    print(f"  {'Overall Avg Wait':<25} {avg_a:>10.2f} {avg_b:>10.2f} {d:>20}")

    print()

    # Verdict
    switch_improvement = (totals_a["switches"] - totals_b["switches"]) / totals_a["switches"] * 100 if totals_a["switches"] > 0 else 0
    gpu_improvement = gpu_b - gpu_a
    wait_improvement = (avg_a - avg_b) / avg_a * 100 if avg_a > 0 else 0

    print("VERDICT:")
    print(f"  Switch reduction:    {switch_improvement:+.1f}%")
    print(f"  GPU serving change:  {gpu_improvement:+.1f} percentage points")
    print(f"  Wait time change:    {wait_improvement:+.1f}%")

    if gpu_improvement > 0 and switch_improvement > 0:
        print(f"  >> {label_b} is BETTER than {label_a}")
    elif gpu_improvement < -5:
        print(f"  >> {label_b} is WORSE than {label_a}")
    else:
        print(f"  >> Results are MIXED — review per-profile data")


if __name__ == "__main__":
    main()
