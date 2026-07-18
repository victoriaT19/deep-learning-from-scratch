# Tokenization — notes ("Let's build the GPT Tokenizer", karpathy/minbpe)

- Text → UTF-8 bytes → Byte Pair Encoding: repeatedly merge the most frequent adjacent pair into a new token. 256 byte tokens are the base, so any string can be encoded and nothing is ever out of vocabulary.
- `basic.py`: BPE directly on the byte stream. Encoding applies merges in the order they were learned (lowest new id first), not by frequency in the input.
- `regex_tokenizer.py`: GPT-4's split pattern first cuts the text into chunks (words with their leading space, runs of up to 3 digits, punctuation, whitespace), and merges never cross chunk boundaries. That is why "dog." never becomes a single token (`test_regex_never_merges_across_chunks`).
- Special tokens (`<|endoftext|>`, chat markers) bypass BPE entirely and must be allowed explicitly when encoding, otherwise user text could inject them.
- `train_tokenizer.py` trains both on 200 KB of Shakespeare and prints the compression ratio.

Things that explain odd LLM behaviour:
- Non-English text and code use more tokens per character, so they cost more context and compute. I checked this on Portuguese words with accents: each accented letter is 2 bytes before any merge.
- Arithmetic is hard partly because numbers are split into arbitrary 1–3 digit chunks.
- Trailing spaces and "SolidGoldMagikarp"-style tokens: tokens that were frequent in the tokenizer's training data but rare in the model's.

Vocabulary size is a trade-off: bigger vocab → shorter sequences but a larger embedding table and rarer tokens.
