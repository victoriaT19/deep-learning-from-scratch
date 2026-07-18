import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03-transformer"))

from data import load_text

from basic import BasicTokenizer
from regex_tokenizer import RegexTokenizer

vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
text = load_text()[:200_000]
out = Path(__file__).resolve().parent / "models"
out.mkdir(exist_ok=True)
for name, tok in [("basic", BasicTokenizer()), ("regex", RegexTokenizer())]:
    t0 = time.time()
    tok.train(text, vocab_size)
    ids = tok.encode_ordinary(text) if hasattr(tok, "encode_ordinary") else tok.encode(text)
    tok.save(str(out / name))
    print(f"{name}: {len(text.encode('utf-8'))} bytes -> {len(ids)} tokens "
          f"(ratio {len(text.encode('utf-8')) / len(ids):.2f}) in {time.time() - t0:.1f}s")
