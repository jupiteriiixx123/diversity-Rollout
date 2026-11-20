#!/usr/bin/env python3
"""Compare paired validation JSONL outputs written by verl's val_only mode."""

"""
python scripts/tool/compare_babyai_eval.py \
  runs/eval_babyai/babyai_test_v2
"""

"""Compare verl BabyAI validation: reward rates + pass@k + paired CIs.

Input: <root>/{base,lora}/trajectories/*.jsonl
Task grouping requires verl val_kwargs.n with interleave=True and matching
validation task ordering. Identical prompt text does not prove task-ID equality.
"""
import argparse
import collections
import json
import math
import random
from pathlib import Path
from statistics import mean

MARKER = "the task is completed"


def load_rows(folder):
    paths = sorted((folder / "trajectories").glob("*.jsonl"))
    if len(paths) != 1:
        raise SystemExit(f"Expected exactly one JSONL in {folder / 'trajectories'}; found {len(paths)}")
    rows = []
    for num, line in enumerate(paths[0].read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not {"input", "output", "score"}.issubset(row):
                raise ValueError("missing input/output/score")
            row["score"] = float(row["score"])
            if not math.isfinite(row["score"]):
                raise ValueError("score must be finite")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"Invalid row in {paths[0]} line {num}: {exc}") from exc
    if not rows:
        raise SystemExit(f"Empty validation file: {paths[0]}")
    return rows


def group_tasks(rows, n):
    if len(rows) % n:
        raise SystemExit(f"Rows={len(rows)} is not divisible by samples per task n={n}")
    groups = []
    for offset in range(0, len(rows), n):
        block = rows[offset:offset + n]
        prompt = str(block[0]["input"])
        if any(str(row["input"]) != prompt for row in block):
            raise SystemExit(f"Task grouping failed at row {offset}: verify --n and interleave=True")
        scores = [row["score"] for row in block]
        markers = [MARKER in str(row["output"]).lower() for row in block]
        positives = [score > 0 for score in scores]
        groups.append({"input": prompt, "scores": scores, "markers": markers, "positives": positives})
    return groups


def pass_at_k(n, successes, k):
    """Without-replacement pass@k estimator; k <= n."""
    if k < 1 or k > n or successes < 0 or successes > n:
        raise ValueError(f"Invalid n={n}, successes={successes}, k={k}")
    return 1.0 if n - successes < k else 1.0 - math.comb(n - successes, k) / math.comb(n, k)


def metric_arrays(tasks, n, ks, success):
    metrics = {
        "mean_reward": [],
        "positive_reward_rate": [],
        "completion_marker_rate": [],
    }
    metrics.update({f"pass@{k}": [] for k in ks})
    for task in tasks:
        metrics["mean_reward"].append(mean(task["scores"]))
        metrics["positive_reward_rate"].append(sum(task["positives"]) / n)
        metrics["completion_marker_rate"].append(sum(task["markers"]) / n)
        c = sum(task["markers"] if success == "marker" else task["positives"])
        for k in ks:
            metrics[f"pass@{k}"].append(pass_at_k(n, c, k))
    return metrics


def paired_ci(diffs, reps=2000, seed=42):
    rng = random.Random(seed)
    n = len(diffs)
    sims = sorted(mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(reps))
    return [sims[int(0.025 * (reps - 1))], sims[int(0.975 * (reps - 1))]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, help="runs/eval_babyai/<EVAL_NAME>")
    p.add_argument("--n", type=int, default=8, help="val_kwargs.n; default 8")
    p.add_argument("--expected-tasks", type=int, default=90, help="0 disables count check")
    p.add_argument("--success", choices=("marker", "positive-reward"), default="marker")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    if args.n < 1 or args.expected_tasks < 0 or args.bootstrap < 100:
        p.error("Require n>=1, expected-tasks>=0, bootstrap>=100")

    b_rows = load_rows(args.root / "base")
    l_rows = load_rows(args.root / "lora")
    base = group_tasks(b_rows, args.n)
    lora = group_tasks(l_rows, args.n)
    if len(base) != len(lora):
        raise SystemExit(f"Task counts differ: base={len(base)}, lora={len(lora)}")
    if args.expected_tasks and len(base) != args.expected_tasks:
        raise SystemExit(f"Expected {args.expected_tasks} tasks, got {len(base)}")
    for i, (b, l) in enumerate(zip(base, lora)):
        if b["input"] != l["input"]:
            raise SystemExit(f"Task order mismatch at index {i}")

    repeated = sum(c - 1 for c in collections.Counter(t["input"] for t in base).values() if c > 1)
    if repeated:
        print(f"WARNING: {repeated} repeated prompt texts; ordered task ID/seed parity cannot be verified from JSONL alone.")

    ks = sorted(set([k for k in (1, 2, 4, 8) if k <= args.n] + [args.n]))
    metrics = ["mean_reward", "positive_reward_rate", "completion_marker_rate"] + [f"pass@{k}" for k in ks]
    b_arr = metric_arrays(base, args.n, ks, args.success)
    l_arr = metric_arrays(lora, args.n, ks, args.success)
    b_stats = {key: mean(b_arr[key]) for key in metrics}
    l_stats = {key: mean(l_arr[key]) for key in metrics}
    delta = {key: l_stats[key] - b_stats[key] for key in metrics}
    ci = {key: paired_ci([v - u for u, v in zip(b_arr[key], l_arr[key])], args.bootstrap) for key in metrics}
    disagree = {
        "base": sum(a != b for t in base for a, b in zip(t["markers"], t["positives"])),
        "lora": sum(a != b for t in lora for a, b in zip(t["markers"], t["positives"])),
    }

    print(f"Tasks={len(base)} samples/task={args.n} trajectories/model={len(b_rows)} success={args.success}")
    print(f"Marker/positive-reward disagreements: {disagree}")
    print(f"{'Metric':<27} {'Base':>10} {'LoRA':>10} {'Delta':>11} {'Paired 95% CI':>24}")
    for key in metrics:
        lo, hi = ci[key]
        print(f"{key:<27} {b_stats[key]:>10.4f} {l_stats[key]:>10.4f} {delta[key]:>+11.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("Note: pass@1 equals completion_marker_rate when --success=marker; all rates are fractions in [0,1].")
    print("Note: success-marker and positive-reward are distinct criteria; do not treat reward>0 as guaranteed success.")

    report = {
        "n_tasks": len(base), "samples_per_task": args.n,
        "success_criterion": args.success, "marker_reward_disagreements": disagree,
        "base": {"n": len(b_rows), **b_stats},
        "lora": {"n": len(l_rows), **l_stats},
        "delta": delta, "paired_bootstrap_delta_ci95": ci,
    }
    outfile = args.root / "passk_comparison.json"
    outfile.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
