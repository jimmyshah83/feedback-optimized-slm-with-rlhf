"""Custom PubMedQA classification accuracy metric."""

from __future__ import annotations

import re


def extract_decision(text: str) -> str | None:
    """Extract yes/no/maybe decision from model output.

    Looks for patterns like 'Decision: yes', '**Decision:** no',
    or a standalone yes/no/maybe at the end.
    """
    if not text:
        return None

    text_lower = text.lower()

    pattern = r"\*?\*?decision\*?\*?\s*:?\s*(yes|no|maybe)"
    match = re.search(pattern, text_lower)
    if match:
        return match.group(1)

    for label in ("yes", "no", "maybe"):
        if text_lower.rstrip().endswith(label):
            return label

    return None


def pubmedqa_accuracy(
    responses: list[str],
    ground_truths: list[str],
) -> dict:
    """Compute PubMedQA classification accuracy.

    Returns dict with 'accuracy', 'correct', 'total',
    'unparsed', and per-class counts.
    """
    correct = 0
    unparsed = 0
    class_counts: dict[str, dict[str, int]] = {
        "yes": {"correct": 0, "total": 0},
        "no": {"correct": 0, "total": 0},
        "maybe": {"correct": 0, "total": 0},
    }

    for resp, gt in zip(responses, ground_truths):
        gt_label = gt.strip().lower()
        if gt_label in class_counts:
            class_counts[gt_label]["total"] += 1

        pred = extract_decision(resp)
        if pred is None:
            unparsed += 1
            continue

        if pred == gt_label:
            correct += 1
            if gt_label in class_counts:
                class_counts[gt_label]["correct"] += 1

    total = len(responses)
    parsed = total - unparsed

    return {
        "accuracy": correct / parsed if parsed > 0 else 0.0,
        "correct": correct,
        "total": total,
        "parsed": parsed,
        "unparsed": unparsed,
        "per_class": class_counts,
    }
