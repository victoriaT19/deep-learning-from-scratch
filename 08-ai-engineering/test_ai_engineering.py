import json

from agent import calculate, run
from evals import evaluate, exact_match
from rag import build_prompt
from retrieval import TfidfIndex, chunk

DOCS = [
    "BatchNorm normalises activations with the batch mean and variance and keeps running statistics for inference.",
    "Byte pair encoding merges the most frequent pair of tokens repeatedly.",
    "The KV cache stores keys and values so decoding a new token is linear in the context length.",
]


def test_retrieval_ranks_the_relevant_chunk_first():
    index = TfidfIndex(DOCS)
    assert "BatchNorm" in index.search("running statistics batch variance", 1)[0][0]
    assert "KV cache" in index.search("decoding keys values cache", 1)[0][0]


def test_chunks_overlap():
    words = " ".join(str(i) for i in range(200))
    chunks = chunk(words, size=80, overlap=20)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_prompt_numbers_the_context():
    msgs = build_prompt("q?", [(DOCS[0], 1.0), (DOCS[1], 0.5)])
    assert "[1] BatchNorm" in msgs[1]["content"] and "[2] Byte" in msgs[1]["content"]


def test_calculator_is_safe():
    assert calculate("17 * 23 + 4") == 395
    try:
        calculate("__import__('os')")
        assert False
    except ValueError:
        pass


def test_agent_loop_with_a_scripted_model():
    replies = iter([
        {"content": "", "tool_calls": [{"id": "1", "function": {"name": "calculate", "arguments": json.dumps({"expression": "6*7"})}}]},
        {"content": "42"},
    ])
    answer, trace = run("what is 6*7?", lambda messages, tools=None: next(replies))
    assert answer == "42" and trace[0]["result"] == 42


def test_eval_harness_counts_errors():
    cases = [{"id": 1, "input": "a", "gold": "A"}, {"id": 2, "input": "b", "gold": "B"}]

    def system(x):
        if x == "b":
            raise RuntimeError("boom")
        return x

    _, summary = evaluate(cases, system, metric=exact_match)
    assert summary["accuracy"] == 0.5 and summary["errors"] == 1
