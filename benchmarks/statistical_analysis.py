#!/usr/bin/env python3
"""
Statistical significance analysis for MemMe benchmark results.
Computes bootstrap confidence intervals and permutation tests for ablation comparisons.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

np.random.seed(42)
N_BOOTSTRAP = 10_000
ALPHA = 0.05

# --- File paths ---
FILES = {
    "Full System":        "locomo/results_rerank_full/run_20260326_162109.jsonl",
    "No Rerank":          "locomo/results_ablation_no_rerank/run_20260326_171648.jsonl",
    "No FTS":             "locomo/results_ablation_no_fts/run_20260326_182058.jsonl",
    "No Temporal":        "locomo/results_ablation_no_temporal/run_20260326_233239.jsonl",
    "No Entity":          "locomo/results_ablation_no_entity/run_20260326_174603.jsonl",
    "Vector Only":        "locomo/results_ablation_vector_only/run_20260326_192808.jsonl",
    "Vector+Rerank":      "locomo/results_ablation_vector_rerank_only/run_20260326_195657.jsonl",
}

BASE_DIR = Path(__file__).parent


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = ALPHA):
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    mean = np.mean(values)
    n = len(values)
    boot_means = np.array([
        np.mean(values[np.random.randint(0, n, size=n)])
        for _ in range(n_boot)
    ])
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return mean, ci_low, ci_high


def paired_bootstrap_test(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOTSTRAP):
    """
    Paired bootstrap test: is mean(a) > mean(b)?
    Returns delta, (ci_low, ci_high), p_value.
    a and b must be aligned (same questions in same order).
    """
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    diff = a - b
    observed_delta = np.mean(diff)
    n = len(diff)

    boot_deltas = np.array([
        np.mean(diff[np.random.randint(0, n, size=n)])
        for _ in range(n_boot)
    ])
    ci_low = np.percentile(boot_deltas, 100 * ALPHA / 2)
    ci_high = np.percentile(boot_deltas, 100 * (1 - ALPHA / 2))

    # Permutation-style p-value: fraction of bootstrap samples where delta <= 0
    p_value = np.mean(boot_deltas <= 0)

    return observed_delta, (ci_low, ci_high), p_value


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = N_BOOTSTRAP):
    """
    Two-sided permutation test on paired differences.
    H0: no difference between a and b.
    """
    assert len(a) == len(b)
    diff = a - b
    observed = np.abs(np.mean(diff))
    n = len(diff)

    count = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=n)
        perm_mean = np.abs(np.mean(diff * signs))
        if perm_mean >= observed:
            count += 1
    return count / n_perm


def align_by_question(records_a, records_b):
    """Align two result sets by (sample_id, question) and return paired arrays."""
    key_fn = lambda r: (r["sample_id"], r["question"])
    map_a = {key_fn(r): r["judge_mean"] for r in records_a}
    map_b = {key_fn(r): r["judge_mean"] for r in records_b}
    common_keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(common_keys) < len(map_a):
        print(f"  Warning: {len(map_a)} vs {len(map_b)} records, {len(common_keys)} in common")
    a_vals = np.array([map_a[k] for k in common_keys])
    b_vals = np.array([map_b[k] for k in common_keys])
    return a_vals, b_vals


def main():
    # Load all data
    all_data = {}
    for name, rel_path in FILES.items():
        path = BASE_DIR / rel_path
        all_data[name] = load_jsonl(path)

    # ===== Table 1: Overall means and CIs =====
    print("=" * 80)
    print("TABLE 1: Overall Judge Accuracy (judge_mean) with 95% Bootstrap CI")
    print("=" * 80)
    print(f"{'System':<22} {'N':>5}  {'Mean':>7}  {'95% CI':>18}  {'SE':>7}")
    print("-" * 65)

    overall_stats = {}
    for name, records in all_data.items():
        values = np.array([r["judge_mean"] for r in records])
        mean, ci_lo, ci_hi = bootstrap_ci(values)
        se = np.std(values) / np.sqrt(len(values))
        overall_stats[name] = (mean, ci_lo, ci_hi, se)
        print(f"{name:<22} {len(values):>5}  {mean*100:>6.2f}%  [{ci_lo*100:>6.2f}%, {ci_hi*100:>6.2f}%]  {se*100:>6.3f}%")

    # ===== Table 2: Per-category breakdown =====
    print()
    print("=" * 80)
    print("TABLE 2: Per-Category Judge Accuracy with 95% Bootstrap CI")
    print("=" * 80)

    # Collect all categories
    all_cats = sorted(set(r["category_name"] for records in all_data.values() for r in records))

    for cat in all_cats:
        print(f"\n--- {cat.upper()} ---")
        print(f"{'System':<22} {'N':>5}  {'Mean':>7}  {'95% CI':>18}")
        print("-" * 58)
        for name, records in all_data.items():
            values = np.array([r["judge_mean"] for r in records if r["category_name"] == cat])
            if len(values) == 0:
                print(f"{name:<22}     0     N/A")
                continue
            mean, ci_lo, ci_hi = bootstrap_ci(values)
            print(f"{name:<22} {len(values):>5}  {mean*100:>6.2f}%  [{ci_lo*100:>6.2f}%, {ci_hi*100:>6.2f}%]")

    # ===== Table 3: Pairwise comparisons =====
    print()
    print("=" * 80)
    print("TABLE 3: Statistical Significance Tests (Full System vs Ablations)")
    print("=" * 80)
    print(f"{'Comparison':<35} {'Delta':>7}  {'95% CI of Delta':>22}  {'p-value (boot)':>14}  {'p-value (perm)':>14}  {'Sig?':>5}")
    print("-" * 105)

    comparisons = [
        ("Full System", "No Rerank"),
        ("Full System", "No FTS"),
        ("Full System", "No Temporal"),
        ("Full System", "No Entity"),
        ("Full System", "Vector Only"),
        ("Full System", "Vector+Rerank"),
    ]

    full_records = all_data["Full System"]

    for sys_a, sys_b in comparisons:
        rec_a = all_data[sys_a]
        rec_b = all_data[sys_b]
        a_vals, b_vals = align_by_question(rec_a, rec_b)

        delta, (ci_lo, ci_hi), p_boot = paired_bootstrap_test(a_vals, b_vals)
        p_perm = permutation_test(a_vals, b_vals)
        sig = "***" if p_perm < 0.001 else "**" if p_perm < 0.01 else "*" if p_perm < 0.05 else "ns"

        label = f"Full vs {sys_b}"
        print(f"{label:<35} {delta*100:>+6.2f}%  [{ci_lo*100:>+6.2f}%, {ci_hi*100:>+6.2f}%]  "
              f"{p_boot:>14.4f}  {p_perm:>14.4f}  {sig:>5}")

    # ===== Table 4: Per-category significance for key comparisons =====
    print()
    print("=" * 80)
    print("TABLE 4: Per-Category Significance (Full System vs No Rerank)")
    print("=" * 80)
    print(f"{'Category':<15} {'Delta':>7}  {'95% CI of Delta':>22}  {'p-perm':>8}  {'Sig?':>5}")
    print("-" * 65)

    key_ablation = "No Rerank"
    for cat in all_cats:
        rec_a_cat = [r for r in full_records if r["category_name"] == cat]
        rec_b_cat = [r for r in all_data[key_ablation] if r["category_name"] == cat]
        if not rec_a_cat or not rec_b_cat:
            continue
        a_vals, b_vals = align_by_question(rec_a_cat, rec_b_cat)
        if len(a_vals) < 5:
            continue
        delta, (ci_lo, ci_hi), _ = paired_bootstrap_test(a_vals, b_vals)
        p_perm = permutation_test(a_vals, b_vals)
        sig = "***" if p_perm < 0.001 else "**" if p_perm < 0.01 else "*" if p_perm < 0.05 else "ns"
        print(f"{cat:<15} {delta*100:>+6.2f}%  [{ci_lo*100:>+6.2f}%, {ci_hi*100:>+6.2f}%]  {p_perm:>8.4f}  {sig:>5}")

    # ===== LLM Judge variance analysis =====
    print()
    print("=" * 80)
    print("TABLE 5: LLM Judge Variance Analysis (Full System)")
    print("=" * 80)

    values = np.array([r["judge_mean"] for r in full_records])
    n = len(values)
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    # Since judge_mean is binary (0 or 1), variance = p*(1-p)
    p_hat = mean
    se_bernoulli = np.sqrt(p_hat * (1 - p_hat) / n)

    print(f"  N questions:                 {n}")
    print(f"  Overall mean:                {mean*100:.2f}%")
    print(f"  Empirical SE:                {se*100:.3f}%")
    print(f"  Bernoulli SE (sqrt(pq/n)):   {se_bernoulli*100:.3f}%")
    print(f"  95% CI (normal approx):      [{(mean - 1.96*se)*100:.2f}%, {(mean + 1.96*se)*100:.2f}%]")

    # Check how many have multiple judge scores
    multi_judge = [r for r in full_records if len(r.get("judge_scores", [])) > 1]
    print(f"  Questions with >1 judge run:  {len(multi_judge)} / {n}")
    if multi_judge:
        intra_vars = [np.var(r["judge_scores"]) for r in multi_judge]
        print(f"  Mean intra-question variance: {np.mean(intra_vars)*100:.3f}%")

    # Effect size (Cohen's h for proportions)
    print()
    print("=" * 80)
    print("TABLE 6: Effect Sizes (Cohen's h)")
    print("=" * 80)
    print(f"{'Comparison':<35} {'p1':>7}  {'p2':>7}  {'Cohen h':>8}  {'Magnitude':>10}")
    print("-" * 72)

    for sys_a, sys_b in comparisons:
        p1 = overall_stats[sys_a][0]
        p2 = overall_stats[sys_b][0]
        # Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
        h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
        mag = "large" if abs(h) >= 0.8 else "medium" if abs(h) >= 0.5 else "small" if abs(h) >= 0.2 else "negligible"
        label = f"Full vs {sys_b}"
        print(f"{label:<35} {p1*100:>6.2f}% {p2*100:>6.2f}%  {h:>+7.4f}  {mag:>10}")

    print()
    print("=" * 80)
    print("Analysis complete. All CIs are 95% percentile bootstrap with 10,000 resamples.")
    print("Permutation test: two-sided sign-flip test on paired differences.")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("=" * 80)


if __name__ == "__main__":
    main()
