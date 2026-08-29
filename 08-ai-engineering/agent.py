import ast
import json
import operator
from pathlib import Path

from rag import build_index

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.USub: operator.neg}


def calculate(expression):
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expression, mode="eval").body)


_index = None


def search_notes(query):
    global _index
    _index = _index or build_index(Path(__file__).resolve().parents[1])
    return [doc[:300] for doc, _ in _index.search(query, 3)]


TOOLS = {"calculate": calculate, "search_notes": search_notes}
SCHEMAS = [
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate an arithmetic expression.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search_notes", "description": "Search my study notes.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def run(question, chat, max_steps=5):
    messages = [{"role": "system", "content": "Use the tools when they help. Answer briefly."},
                {"role": "user", "content": question}]
    trace = []
    for _ in range(max_steps):
        msg = chat(messages, tools=SCHEMAS)
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
        if not calls:
            return msg.get("content", ""), trace
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
                result = TOOLS[name](**args)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            trace.append({"tool": name, "args": call["function"].get("arguments"), "result": result})
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": json.dumps(result, ensure_ascii=False)})
    return "stopped: too many steps", trace


if __name__ == "__main__":
    import sys

    from llm import chat as llm_chat

    answer, trace = run(" ".join(sys.argv[1:]) or "What is 17 * 23 + 4?", llm_chat)
    for t in trace:
        print("tool:", t)
    print(answer)
