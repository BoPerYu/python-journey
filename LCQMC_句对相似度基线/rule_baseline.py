r"""LCQMC rule baseline: character-set overlap without sklearn.

The baseline deliberately ignores multiplicity and word order. It exists to
reveal where a hand-written similarity rule fails before introducing a model.

Run from this directory:
    ..\w5-d1\.venv\Scripts\python.exe -X utf8 rule_baseline.py
"""

from math import isclose


def character_overlap(sentence1: str, sentence2: str) -> float:
    """Return Jaccard overlap of the two character sets.

    Contract:
    - The result is between 0.0 and 1.0.
    - Identical non-empty sentences return 1.0.
    - Sentences without a shared character return 0.0.
    """
    s1=list(sentence1)
    s2=list(sentence2)
    if not s1 or not s2:
        return 0
    tog = s1 & s2
    all = s1 | s2
    return len(tog)/len(all)


def predict_pair(
    sentence1: str,
    sentence2: str,
    threshold: float = 0.60,
) -> int:
    """Return label 1 when character overlap reaches the threshold."""
    return int(character_overlap(sentence1, sentence2) >= threshold)


def _check(name, got, want):
    if isinstance(want, float):
        ok = isinstance(got, float) and isclose(got, want, abs_tol=1e-9)
    else:
        ok = got == want and type(got) is type(want)
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def _selftest():
    cases = [
        ("1 identical", "我喜欢猫", "我喜欢猫", 1.0),
        ("2 partial overlap", "喜欢篮球", "喜欢打篮球", 0.8),
        ("3 no overlap", "今天下雨", "这个苹果很甜", 0.0),
        ("4 negation adds one character", "我喜欢猫", "我不喜欢猫", 0.8),
        ("5 reversed characters", "猫追狗", "狗追猫", 1.0),
    ]
    results = []
    for name, sentence1, sentence2, want in cases:
        try:
            results.append(
                _check(
                    name,
                    character_overlap(sentence1, sentence2),
                    want,
                )
            )
        except Exception as error:
            print(f"[FAIL] {name}: {type(error).__name__}: {error}")
            results.append(False)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过" + ("" if passed == len(results) else "  ← 还没写对"))


def _show_failure_examples():
    examples = [
        ("same meaning", "如何学习Python", "Python怎么学", 1),
        ("same topic, different question", "如何学习Python", "Python怎么安装", 0),
        ("negation", "我喜欢猫", "我不喜欢猫", 0),
        ("word order", "猫追狗", "狗追猫", 0),
        ("dialogue wording", "今天天气很好", "今天天气不错", 1),
    ]

    print("\nrule baseline failure inspection")
    print(f"{'case':<32} {'overlap':>8} {'pred':>6} {'true':>6}")
    for name, sentence1, sentence2, true_label in examples:
        overlap = character_overlap(sentence1, sentence2)
        prediction = predict_pair(sentence1, sentence2)
        print(f"{name:<32} {overlap:>8.4f} {prediction:>6} {true_label:>6}")


if __name__ == "__main__":
    _selftest()
    try:
        _show_failure_examples()
    except NotImplementedError:
        print("\n完成 character_overlap 后，再观察这里的错例。")
