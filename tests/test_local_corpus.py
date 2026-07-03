from __future__ import annotations

import json

import pytest

from deepresearch.backends.local_corpus import LocalCorpusBackend


DOCS = [
    {"doc_id": "d1", "title": "Paris", "text": "Paris is the capital of France and its largest city."},
    {"doc_id": "d2", "title": "Berlin", "text": "Berlin is the capital of Germany, known for its history."},
    {"doc_id": "d3", "title": "Bananas", "text": "Bananas are a good source of potassium and are yellow."},
]


@pytest.mark.asyncio
async def test_search_returns_most_relevant_document_first():
    backend = LocalCorpusBackend.from_dicts(DOCS)
    results = await backend.search("capital of France", max_results=3)
    assert results[0].title == "Paris"
    assert results[0].url == "corpus://d1"


@pytest.mark.asyncio
async def test_fetch_returns_full_document_text():
    backend = LocalCorpusBackend.from_dicts(DOCS)
    result = await backend.fetch("corpus://d2")
    assert "Berlin" in result.content
    assert "capital of Germany" in result.content


@pytest.mark.asyncio
async def test_fetch_unknown_doc_id_raises():
    backend = LocalCorpusBackend.from_dicts(DOCS)
    with pytest.raises(KeyError):
        await backend.fetch("corpus://does-not-exist")


def test_empty_document_list_raises():
    with pytest.raises(ValueError):
        LocalCorpusBackend.from_dicts([])


@pytest.mark.asyncio
async def test_from_json_file_round_trips(tmp_path):
    path = tmp_path / "q1.json"
    path.write_text(json.dumps(DOCS), encoding="utf-8")
    backend = LocalCorpusBackend.from_json_file(path)
    results = await backend.search("potassium yellow fruit", max_results=1)
    assert results[0].title == "Bananas"
