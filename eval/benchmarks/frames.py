"""FRAMES benchmark loader — docs/DESIGN.md §5.1.

HF `google/frames-benchmark`, 824 questions, single "test" split. No
official small subset exists — sampled ourselves: fixed seed, stratified by
the dataset's `reasoning_types` column, pinned to a specific dataset
revision (docs/DESIGN.md risk table: "benchmark drift").

Unlike MuSiQue, FRAMES only ships gold Wikipedia *links*, not article text —
each question's local corpus has to be built by actually fetching those
pages once (ingest_corpus below, via the MediaWiki action API — the REST
`page/plain` endpoint 403'd/404'd against this project's User-Agent when
checked; the classic `action=query&prop=extracts` endpoint worked cleanly),
then cached to disk so re-running the harness doesn't re-fetch.
"""

from __future__ import annotations

import ast
import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

DATASET = "google/frames-benchmark"
DATASET_SPLIT = "test"
DATASET_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"

CORPUS_DIR = Path(__file__).parent.parent.parent / "data" / "corpus" / "frames"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia's edge WAF appears to filter on User-Agent *content*, not just
# presence/rate — a differently-worded UA string got 403s while this one
# didn't, in back-to-back testing against the same endpoint. Replace the
# contact placeholder with a real one before any sustained use, per
# Wikipedia's bot policy (https://w.wiki/4wJS).
USER_AGENT = "DeepResearchEvalBot/0.1 (https://example.com/contact; research-eval-harness)"


@dataclass
class FramesExample:
    example_id: str
    question: str
    answer: str
    reasoning_type: str
    wikipedia_links: list[str]
    corpus_path: Path


def _title_from_wikipedia_url(url: str) -> str:
    path = urlparse(url).path  # "/wiki/Some_Article_Title"
    return unquote(path.rsplit("/", 1)[-1]).replace("_", " ")


def _wiki_links(row: dict) -> list[str]:
    """`wiki_links` comes out of the CSV-backed HF dataset as the *string*
    repr of a Python list (`"['https://...', ...]"`), not an actual list —
    iterating it directly iterates characters. Parse it properly."""
    raw = row.get("wiki_links")
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    return [link for link in (raw or []) if link]


def load_subset(n: int, seed: int, *, stratify_by_reasoning_type: bool = True) -> list[dict]:
    """Deterministic sample of raw dataset rows — NOT yet ingested. Call
    ingest_corpus() separately (the slow, network-bound step) once you've
    decided how many articles fetching this subset implies."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=DATASET_SPLIT, revision=DATASET_REVISION)
    rng = random.Random(seed)

    if stratify_by_reasoning_type:
        by_type: dict[str, list[int]] = {}
        for i, row in enumerate(ds):
            by_type.setdefault(row["reasoning_types"], []).append(i)
        for idxs in by_type.values():
            rng.shuffle(idxs)
        types = sorted(by_type.keys())
        cursors = {t: 0 for t in types}
        selected_idxs: list[int] = []
        while len(selected_idxs) < n:
            progressed = False
            for t in types:
                if len(selected_idxs) >= n:
                    break
                if cursors[t] < len(by_type[t]):
                    selected_idxs.append(by_type[t][cursors[t]])
                    cursors[t] += 1
                    progressed = True
            if not progressed:
                break
    else:
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        selected_idxs = idxs[:n]

    return [ds[i] for i in selected_idxs]


def estimate_articles_to_fetch(rows: list[dict]) -> int:
    """How many (example, article) pairs ingest_corpus would fetch, counting
    only examples not already cached on disk — print this before running a
    nightly-scale ingestion; it can be in the hundreds."""
    count = 0
    for row in rows:
        example_id = str(row["Unnamed: 0"])
        if (CORPUS_DIR / f"{example_id}.json").exists():
            continue
        count += len(_wiki_links(row))
    return count


async def _fetch_wikipedia_extract(client: httpx.AsyncClient, title: str) -> str:
    resp = await client.get(
        WIKIPEDIA_API,
        params={"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json"},
    )
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    for page in pages.values():
        return page.get("extract", "") or ""
    return ""


async def ingest_corpus(rows: list[dict], *, delay_seconds: float = 0.2) -> list[FramesExample]:
    """Fetches each row's gold Wikipedia articles (skipped if already cached
    on disk for that example id) and returns FramesExample objects ready for
    LocalCorpusBackend."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        for row in rows:
            example_id = str(row["Unnamed: 0"])
            corpus_path = CORPUS_DIR / f"{example_id}.json"
            links = _wiki_links(row)

            if not corpus_path.exists():
                documents = []
                for link in links:
                    title = _title_from_wikipedia_url(link)
                    text = await _fetch_wikipedia_extract(client, title)
                    documents.append(
                        {"doc_id": link, "title": title, "text": text or f"(no content fetched for {title})"}
                    )
                    await asyncio.sleep(delay_seconds)
                corpus_path.write_text(json.dumps(documents), encoding="utf-8")

            examples.append(
                FramesExample(
                    example_id=example_id,
                    question=row["Prompt"],
                    answer=row["Answer"],
                    reasoning_type=row["reasoning_types"],
                    wikipedia_links=links,
                    corpus_path=corpus_path,
                )
            )
    return examples
