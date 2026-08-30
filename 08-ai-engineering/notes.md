# AI engineering — notes (rohitg00/ai-engineering-from-scratch)

Everything around the model: getting the right context into the prompt, letting the model act through tools, and measuring whether the system works. Built from scratch, without frameworks, so each piece is visible. The LLM is any OpenAI-compatible endpoint (`llm.py`; a local Ollama model by default, configured with `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`).

## Retrieval and RAG (`retrieval.py`, `rag.py`)
- Chunking with overlap so a sentence split at a boundary still appears whole in one chunk.
- TF-IDF with cosine similarity, written by hand. Dense embeddings capture synonyms better, but lexical search is a strong baseline and needs no model. Hybrid search (both, then rerank) is what production systems usually do.
- The prompt numbers each chunk and asks for citations, and tells the model to say "I don't know" when the context does not contain the answer.
- `rag.py` indexes the notes of this repository, so I can ask questions about what I studied.

## Tool calling (`agent.py`)
- The model returns `tool_calls`; the loop runs the tool, appends the result as a `tool` message and calls the model again until it answers in plain text, with a step limit.
- Tool errors go back to the model as messages instead of crashing the loop; models can often fix their own arguments.
- The calculator parses the AST and only allows arithmetic nodes: never `eval` model output.

## Evals (`evals.py`)
- A case is `{id, input, gold}`; the harness records the prediction, correctness, errors and latency per case and summarises accuracy and median latency.
- Normalising before comparing (case, punctuation, whitespace) avoids counting formatting as failures; `contains` is more forgiving than `exact_match`.
- The tests in `test_ai_engineering.py` use a scripted fake model so the agent loop is tested without any API.

Lessons:
- Retrieval quality bounds RAG quality; look at what was retrieved before blaming the model.
- Every change to the prompt or the tools should be checked against the same eval set.
