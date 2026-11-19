"""Both real scoring entry points must keep Judge labels inside the current batch."""

import os

import pytest
from langchain_core.messages import AIMessage

from evals.judge_labels import claim_labels
from research_engine.runconfig import RunConfig

_before = dict(os.environ)
from evals import benchmark, harness, metrics  # noqa: E402

os.environ.clear()
os.environ.update(_before)


@pytest.mark.parametrize("size", [1, 2, 4])
def test_only_local_numbers_are_accepted(size):
    text = "\n".join(f"Claim {n}: YES" for n in (-1, 0, 1, size, size + 1, 999))
    assert claim_labels(text, size) == {0: True, size - 1: True}


def test_duplicate_conflicting_labels_preserve_last_valid_wins():
    assert claim_labels("Claim 1: YES\nClaim 1: NO\nClaim 2: YES\nClaim 0: NO", 1) == {0: False}


@pytest.mark.parametrize("endpoint", ["harness", "benchmark"])
@pytest.mark.parametrize(
    "first,second,expected",
    [
        ("Claim 5: YES\nClaim 0: NO\nClaim -1: NO\nClaim 99: YES", "", [None] * 5),
        ("Claim 1: YES\nClaim 4: NO", "Claim 2: YES\nClaim 0: NO", [True, None, None, False, None]),
        ("Claim 1: YES\nClaim 1: NO", "Claim 1: YES", [False, None, None, None, True]),
    ],
)
async def test_current_batch_and_short_tail(monkeypatch, endpoint, first, second, expected):
    replies = iter((first, second))

    class Judge:
        async def ainvoke(self, messages):
            return AIMessage(content=next(replies))

    report = "\n".join(f"Fact number {n} is present [1]." for n in range(5))
    sources = [{"index": 1, "url": "https://example.org/a", "snippet": "Evidence"}]
    if endpoint == "harness":
        monkeypatch.setattr(harness, "RUN_CONFIG", RunConfig(llm_mode="fake"))
        monkeypatch.setattr(harness, "_judge_llm", Judge)
        rate, rows = await harness.judge_citation_support(report, sources)
        assert [r["supported"] for r in rows] == expected
        judged = [v for v in expected if v is not None]
        assert rate == (round(sum(judged) / len(judged), 4) if judged else None)
    else:
        monkeypatch.setattr(benchmark, "_build_judge", lambda _: (Judge(), "ollama:controlled"))
        rows = await benchmark.score_system_claims("controlled", "bounds", report, sources)
        labels = {None: None, "SUPPORTED": True, "UNSUPPORTED": False}
        assert [labels[r["judge_verdict"]] for r in rows] == expected


async def test_reasoning_zero_does_not_index_the_last_claim(monkeypatch):
    class Judge:
        async def ainvoke(self, messages):
            return AIMessage(content="Claim 0: NO\nReasoning 0: wrong\nReasoning 2: overflow")

    monkeypatch.setattr(benchmark, "_build_judge", lambda _: (Judge(), "ollama:controlled"))
    rows = await benchmark.score_system_claims("controlled", "tail", "One fact exists [1].", [])
    assert rows[0]["judge_verdict"] is None
    assert not rows[0].get("judge_reasoning")
    assert metrics.METRICS_VERSION == 5
