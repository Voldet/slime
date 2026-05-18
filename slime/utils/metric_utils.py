import logging
import math
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def _score_sample_by_rm_type(sample, rm_type: str) -> float:
    """Score a single sample by rm_type (math/dapo/gpqa/etc)."""
    response = sample.response
    label = sample.label

    if rm_type.startswith("boxed_"):
        from slime.rollout.rm_hub.math_utils import extract_boxed_answer

        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    if rm_type == "math":
        from slime.rollout.rm_hub.math_utils import grade_answer_verl

        return 1.0 if grade_answer_verl(response, label) else 0.0
    elif rm_type == "dapo":
        from slime.rollout.rm_hub.math_dapo_utils import compute_score_dapo

        out = compute_score_dapo(response, str(label))
        return 1.0 if out.get("acc") else 0.0
    elif rm_type == "gpqa":
        from slime.rollout.rm_hub import compute_gpqa_reward

        return float(compute_gpqa_reward(response, label, metadata=getattr(sample, "metadata", {})))
    elif rm_type == "deepscaler":
        from slime.rollout.rm_hub import get_deepscaler_rule_based_reward

        return float(get_deepscaler_rule_based_reward(response, label))
    elif rm_type == "f1":
        from slime.rollout.rm_hub import f1_score

        return float(f1_score(response, label)[0])
    else:
        return 0.0


def compute_eval_group_metrics(
    samples: list,
    group_size: int,
    args=None,
) -> dict[str, float]:
    """Compute eval metrics at prompt group level (mean/maj/best-of-N).

    Args:
        samples: Flat list of eval samples (ordered by prompt, then by repeat).
        group_size: Number of samples per prompt (e.g. 16 for n_samples_per_eval_prompt=16).
        args: Optional args containing eval_verifiable_rm_type. If None, falls back to
              sample.metadata['rm_type'] or default "math".

    Returns:
        Dict with keys acc/mean@N, acc/maj@N/mean, acc/maj@N/std,
        acc/best@N/mean, acc/best@N/std.
    """
    scores = []
    for sample in samples:
        if getattr(sample, "label", None) is None:
            scores.append(0.0)
            continue

        # Determine rm_type: sample.metadata > args.eval_verifiable_rm_type > default "math"
        metadata = getattr(sample, "metadata", {}) or {}
        rm_type = metadata.get("rm_type")
        if not rm_type and args is not None:
            rm_type = getattr(args, "eval_verifiable_rm_type", None)
        if not rm_type:
            rm_type = "math"

        scores.append(_score_sample_by_rm_type(sample, rm_type))

    num_groups = len(scores) // group_size
    assert len(scores) == num_groups * group_size, (
        f"len(scores)={len(scores)} must be divisible by group_size={group_size}"
    )
    group_scores = np.array(scores).reshape(num_groups, group_size)

    # Mean accuracy across all samples
    mean_acc = np.mean(group_scores == 1.0).item()

    # Majority vote: more correct than incorrect within each group
    maj_accs = []
    for group in group_scores:
        correct = np.sum(group == 1.0)
        incorrect = group_size - correct
        maj_accs.append(1.0 if correct > incorrect else 0.0)
    maj_mean = np.mean(maj_accs).item()
    maj_std = np.std(maj_accs).item()

    # Best-of-N: at least one correct in each group
    best_accs = []
    for group in group_scores:
        best_accs.append(1.0 if np.max(group) >= 1.0 else 0.0)
    best_mean = np.mean(best_accs).item()
    best_std = np.std(best_accs).item()

    return {
        f"acc/mean@{group_size}": mean_acc,
        f"acc/maj@{group_size}/mean": maj_mean,
        f"acc/maj@{group_size}/std": maj_std,
        f"acc/best@{group_size}/mean": best_mean,
        f"acc/best@{group_size}/std": best_std,
    }


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
