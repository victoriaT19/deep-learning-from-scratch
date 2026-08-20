import random

TEMPLATES = ["What is {a} plus {b}?", "Add {a} and {b}.", "{a} + {b} = ?", "Quanto é {a} mais {b}?"]


def addition_conversation(rng, max_n=50):
    a, b = rng.randint(0, max_n), rng.randint(0, max_n)
    question = rng.choice(TEMPLATES).format(a=a, b=b)
    return [{"role": "user", "content": question}, {"role": "assistant", "content": str(a + b)}], str(a + b)


def dataset(n, seed=0):
    rng = random.Random(seed)
    return [addition_conversation(rng) for _ in range(n)]
