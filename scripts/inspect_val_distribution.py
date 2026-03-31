"""
Inspect task/suite distribution of a dumped val pkl file.

Suite classification uses a ground-truth lookup table built from the LeRobot
dataset's episodes.jsonl (episode_index → suite by known index ranges), so
no keyword guessing is needed.

If a companion *_meta.json exists (produced by dump_libero_val_balanced.sbatch),
suite counts are also verified against those batch-level labels.

Usage:
    JAX_PLATFORMS=cpu python scripts/inspect_val_distribution.py libero_val_balanced.pkl
    JAX_PLATFORMS=cpu python scripts/inspect_val_distribution.py libero_val.pkl libero_val_balanced.pkl
"""

import argparse
import collections
import json
import os
import pickle

import numpy as np
import sentencepiece

TOKENIZER_PATH = "/storage/yukaichengLab/lishiwen/.cache/openpi/big_vision/paligemma_tokenizer.model"
EPISODES_JSONL = "/storage/yukaichengLab/lishiwen/jiayusun/huggingface/lerobot/meta/episodes.jsonl"

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]

# Episode index ranges per suite (from libero_suite_episodes.py)
SUITE_RANGES = {
    "libero_10":      (0,    379),
    "libero_goal":    (379,  807),
    "libero_object":  (807,  1261),
    "libero_spatial": (1261, 1693),
}


def build_task_to_suite(episodes_jsonl: str) -> dict[str, str]:
    """
    Read episodes.jsonl and build an exact task_text -> suite mapping.
    Each episode has a known suite determined by its episode_index range.
    """
    task_to_suite: dict[str, str] = {}
    with open(episodes_jsonl) as f:
        for line in f:
            ep = json.loads(line)
            idx = ep["episode_index"]
            suite = next(
                (s for s, (lo, hi) in SUITE_RANGES.items() if lo <= idx < hi),
                "unknown",
            )
            for task in ep["tasks"]:
                task_text = task.strip()
                if task_text in task_to_suite and task_to_suite[task_text] != suite:
                    # Task appears in multiple suites — mark ambiguous
                    task_to_suite[task_text] = "ambiguous"
                else:
                    task_to_suite[task_text] = suite
    return task_to_suite


def load_meta(pkl_path: str) -> list[str] | None:
    """Load per-batch suite labels from companion _meta.json if it exists."""
    meta_path = pkl_path.replace(".pkl", "_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("suite_labels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl", nargs="+", help="One or more .pkl files to inspect")
    parser.add_argument("--top-tasks", type=int, default=20, help="Show top N tasks")
    args = parser.parse_args()

    # Build ground-truth task→suite lookup from the dataset
    task_to_suite = build_task_to_suite(EPISODES_JSONL)
    print(f"Loaded task→suite lookup: {len(task_to_suite)} unique tasks from episodes.jsonl")

    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)

    for pkl_path in args.pkl:
        print(f"\n{'='*60}")
        print(f"File: {pkl_path}")
        print(f"{'='*60}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        n_batches = len(data)
        batch_size = data[0][0].tokenized_prompt.shape[0]
        n_samples = n_batches * batch_size
        print(f"Batches: {n_batches},  batch_size: {batch_size},  total samples: {n_samples}")

        suite_counter = collections.Counter()
        task_counter = collections.Counter()

        for obs, _ in data:
            tokens = np.array(obs.tokenized_prompt)
            mask   = np.array(obs.tokenized_prompt_mask)
            for i in range(tokens.shape[0]):
                valid = tokens[i][mask[i]].tolist()
                text  = sp.decode(valid).strip()
                task_counter[text] += 1
                suite_counter[task_to_suite.get(text, "unknown")] += 1

        # Suite distribution (ground truth from episodes.jsonl)
        print(f"\n--- Suite distribution ({n_samples} samples) [episodes.jsonl lookup] ---")
        all_suites = SUITES + [s for s in ["ambiguous", "unknown"] if s in suite_counter]
        for suite in all_suites:
            cnt = suite_counter.get(suite, 0)
            pct = 100 * cnt / n_samples if n_samples > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {suite:<18} {cnt:5d}  ({pct:5.1f}%)  {bar}")

        # Cross-check against meta labels if available
        meta_labels = load_meta(pkl_path)
        if meta_labels:
            meta_counter = collections.Counter(meta_labels)
            print(f"\n--- Cross-check vs _meta.json (per-batch labels) ---")
            match = True
            for suite in SUITES:
                meta_batches = meta_counter.get(suite, 0)
                lookup_samples = suite_counter.get(suite, 0)
                expected_samples = meta_batches * batch_size
                status = "✓" if lookup_samples == expected_samples else "✗ MISMATCH"
                print(f"  {suite:<18} meta={meta_batches} batches ({expected_samples} samples)  "
                      f"lookup={lookup_samples} samples  {status}")
                if lookup_samples != expected_samples:
                    match = False
            if match:
                print("  All suites match.")

        print(f"\n--- Unique tasks: {len(task_counter)} ---")
        print(f"Top {args.top_tasks} by sample count:")
        for task, cnt in task_counter.most_common(args.top_tasks):
            suite = task_to_suite.get(task, "unknown")
            print(f"  [{cnt:4d}] [{suite:<16}]  {task}")


if __name__ == "__main__":
    main()
