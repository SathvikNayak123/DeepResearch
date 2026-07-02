"""MuSiQue benchmark loader — docs/DESIGN.md §5.1.

Answerable subset, HF mirror `bdsaglam/musique` (CC BY 4.0), `validation`
split (test gold is held out). Ships each question's own candidate
paragraphs (gold + distractors) directly, so it's usable as a fixed local
corpus with zero external fetching — unlike FRAMES, which only ships
Wikipedia links and needs an ingestion step (see frames.py).

Pinned to a specific dataset revision so a silent upstream update can't
change what a stored score means (docs/DESIGN.md risk table: "benchmark
drift").
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

DATASET = "bdsaglam/musique"
DATASET_CONFIG = "answerable"
DATASET_SPLIT = "validation"
DATASET_REVISION = "22873a405dd809893b22ada0b499299fb612d2df"

CORPUS_DIR = Path(__file__).parent.parent.parent / "data" / "corpus" / "musique"


@dataclass
class MusiqueExample:
    question_id: str
    question: str
    answer: str
    answer_aliases: list[str]
    hop_count: int
    corpus_path: Path  # one JSON file per question, for LocalCorpusBackend

    @property
    def gold_answers(self) -> list[str]:
        return [self.answer, *self.answer_aliases]


def _hop_count(example_id: str) -> int:
    # ids look like "2hop__...", "3hop__...", "4hop__...", or with a variant
    # suffix before the double underscore ("3hop1__...") — the leading
    # digits before "hop" are always the hop count.
    match = re.match(r"(\d+)hop", example_id)
    if not match:
        raise ValueError(f"unrecognized MuSiQue example id format: {example_id!r}")
    return int(match.group(1))


def _write_corpus(question_id: str, paragraphs: list[dict]) -> Path:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    path = CORPUS_DIR / f"{question_id}.json"
    documents = [
        {"doc_id": f"p{p['idx']}", "title": p["title"], "text": p["paragraph_text"]} for p in paragraphs
    ]
    path.write_text(json.dumps(documents), encoding="utf-8")
    return path


def load_subset(n: int, seed: int, *, stratify_by_hop: bool = True) -> list[MusiqueExample]:
    """Deterministic sample. Writes each selected question's corpus JSON to
    disk as a side effect (idempotent — same seed always writes the same
    files)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION)
    rng = random.Random(seed)

    if stratify_by_hop:
        by_hop: dict[int, list[int]] = {}
        for i, row in enumerate(ds):
            if not row["answerable"]:
                continue
            by_hop.setdefault(_hop_count(row["id"]), []).append(i)
        for idxs in by_hop.values():
            rng.shuffle(idxs)
        hops = sorted(by_hop.keys())
        cursors = {h: 0 for h in hops}
        selected_idxs: list[int] = []
        while len(selected_idxs) < n:
            progressed = False
            for h in hops:
                if len(selected_idxs) >= n:
                    break
                if cursors[h] < len(by_hop[h]):
                    selected_idxs.append(by_hop[h][cursors[h]])
                    cursors[h] += 1
                    progressed = True
            if not progressed:
                break
    else:
        idxs = [i for i, row in enumerate(ds) if row["answerable"]]
        rng.shuffle(idxs)
        selected_idxs = idxs[:n]

    examples = []
    for i in selected_idxs:
        row = ds[i]
        corpus_path = _write_corpus(row["id"], row["paragraphs"])
        examples.append(
            MusiqueExample(
                question_id=row["id"],
                question=row["question"],
                answer=row["answer"],
                answer_aliases=row["answer_aliases"],
                hop_count=_hop_count(row["id"]),
                corpus_path=corpus_path,
            )
        )
    return examples
