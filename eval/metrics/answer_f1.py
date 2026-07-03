"""Answer F1 — standard SQuAD-style token-overlap F1 (docs/DESIGN.md §5.1,
MuSiQue's own metric), best-of over gold answer + aliases."""

from __future__ import annotations

import re
import string


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(predicted: str, gold: str) -> float:
    pred_tokens = _normalize(predicted).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common: dict[str, int] = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_answer_f1(predicted: str, gold_answers: list[str]) -> float:
    return max((token_f1(predicted, gold) for gold in gold_answers), default=0.0)


def gold_contained(predicted: str, gold: str) -> bool:
    """Cheap supplementary signal alongside answer_f1: our agent produces a
    full cited report, not a short-form answer, which token F1 penalizes for
    length even when the report states the fact correctly. This just checks
    whether the gold answer string appears (normalized) anywhere in the
    report — a blunter but less length-biased check."""
    return _normalize(gold) in _normalize(predicted)
