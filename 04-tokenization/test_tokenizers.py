import pytest

from basic import BasicTokenizer
from regex_tokenizer import RegexTokenizer

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Tokenização em português também precisa funcionar: ação, coração, não. "
    "Numbers like 123456 and emoji 🤖 are bytes too.\n" * 20
)
SAMPLES = ["", "hello world", "ação e reação 🤖", "   multiple   spaces\n\nand lines", "1234567890"]


@pytest.mark.parametrize("cls", [BasicTokenizer, RegexTokenizer])
@pytest.mark.parametrize("text", SAMPLES)
def test_roundtrip(cls, text):
    tok = cls()
    tok.train(TEXT, 300)
    assert tok.decode(tok.encode(text)) == text


def test_training_compresses():
    tok = BasicTokenizer()
    tok.train(TEXT, 320)
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8")) / 2


def test_regex_never_merges_across_chunks():
    tok = RegexTokenizer()
    tok.train("dog. dog. dog. dog.", 270)
    tokens = [tok.vocab[i] for i in tok.encode("dog. dog.")]
    assert not any(b"g." in t for t in tokens)


def test_special_tokens_and_save_load(tmp_path):
    tok = RegexTokenizer()
    tok.train(TEXT, 300)
    tok.register_special_tokens({"<|endoftext|>": 300})
    ids = tok.encode("hello<|endoftext|>world", allowed_special="all")
    assert 300 in ids
    assert tok.decode(ids) == "hello<|endoftext|>world"
    tok.save(str(tmp_path / "tok"))
    loaded = RegexTokenizer()
    loaded.load(str(tmp_path / "tok.model"))
    assert loaded.encode("ação 🤖") == tok.encode("ação 🤖")
