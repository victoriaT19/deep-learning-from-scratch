import sys
from pathlib import Path

from llm import chat
from retrieval import TfidfIndex, chunk

SYSTEM = ("Answer using only the context. Cite the chunk numbers you used like [1]. "
          "If the answer is not in the context, say you don't know.")


def build_index(folder):
    chunks = []
    for path in sorted(Path(folder).rglob("*.md")):
        chunks += [f"({path.parent.name}/{path.name}) {c}" for c in chunk(path.read_text(encoding="utf-8"))]
    return TfidfIndex(chunks)


def build_prompt(question, hits):
    context = "\n\n".join(f"[{i + 1}] {doc}" for i, (doc, _) in enumerate(hits))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]


def answer(index, question, k=4):
    hits = index.search(question, k)
    return chat(build_prompt(question, hits))["content"], hits


if __name__ == "__main__":
    index = build_index(Path(__file__).resolve().parents[1])
    question = " ".join(sys.argv[1:]) or "Why does BatchNorm need running statistics?"
    for doc, score in index.search(question, 4):
        print(f"{score:.3f}  {doc[:100]}")
    if "--offline" not in sys.argv:
        print(answer(index, question)[0])
