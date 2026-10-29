"""W6 ML practice: reconstruct the Iris train/test experiment."""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def make_split():
    """Return x_train, x_test, y_train, y_test for Iris.

    Required split:
    - test_size=0.30
    - random_state=42
    - stratify=y
    """
    iris=load_iris()
    x=iris.data
    y=iris.target
    x_train,x_test,y_train,y_test=train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test

def evaluate_depth(max_depth):
    """Train one tree and return train_accuracy, test_accuracy."""
    x_train, x_test, y_train, y_test = make_split()
    model = DecisionTreeClassifier(
    max_depth=max_depth,
    random_state=42,
    )
    model.fit(x_train,y_train)
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    return train_accuracy,test_accuracy

def collect_results():
    """Return one (label, train_accuracy, test_accuracy) tuple per depth."""
    experiments=[("depth 1",1),("depth 3",3),("unlimited",None)]
    rows=[]
    for label,max_depth in experiments:
        train_accuracy,test_accuracy = evaluate_depth(max_depth)
        rows.append((label,train_accuracy,test_accuracy))
    return rows



def main() -> None:
    rows = collect_results()
    print(f"{'model':<12} {'train_acc':>10} {'test_acc':>10} {'gap':>10}")
    for label, train_accuracy, test_accuracy in rows:
        gap = train_accuracy - test_accuracy
        print(
            f"{label:<12} {train_accuracy:>10.4f} "
            f"{test_accuracy:>10.4f} {gap:>10.4f}"
        )


# ================== 自测（这一段不要改） ==================
def _check(name, got, want):
    ok = got == want and type(got) is type(want)
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def _check_call(name, action, predicate):
    try:
        got = action()
    except Exception as error:
        print(f"[FAIL] {name}: {type(error).__name__}: {error}")
        return False

    ok = predicate(got)
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r}")
    return ok


def _selftest():
    results = []

    def split_is_correct(split):
        if not isinstance(split, tuple) or len(split) != 4:
            return False
        x_train, x_test, y_train, y_test = split
        return (
            x_train.shape == (105, 4)
            and x_test.shape == (45, 4)
            and y_train.shape == (105,)
            and y_test.shape == (45,)
        )

    def depth_one_is_correct(scores):
        if not isinstance(scores, tuple) or len(scores) != 2:
            return False
        return abs(scores[0] - 0.6666666667) < 1e-6 and abs(scores[1] - 0.6666666667) < 1e-6

    def depth_three_is_correct(scores):
        if not isinstance(scores, tuple) or len(scores) != 2:
            return False
        return abs(scores[0] - 0.9809523810) < 1e-6 and abs(scores[1] - 0.9777777778) < 1e-6

    def unlimited_is_correct(scores):
        if not isinstance(scores, tuple) or len(scores) != 2:
            return False
        return abs(scores[0] - 1.0) < 1e-6 and abs(scores[1] - 0.9333333333) < 1e-6

    def rows_are_correct(rows):
        if not isinstance(rows, list) or len(rows) != 3:
            return False
        labels = [row[0] for row in rows]
        return labels == ["depth 1", "depth 3", "unlimited"]

    results.append(_check_call("split shapes", make_split, split_is_correct))
    results.append(_check_call("depth 1 scores", lambda: evaluate_depth(1), depth_one_is_correct))
    results.append(_check_call("depth 3 scores", lambda: evaluate_depth(3), depth_three_is_correct))
    results.append(_check_call("unlimited scores", lambda: evaluate_depth(None), unlimited_is_correct))
    results.append(_check_call("three experiment rows", collect_results, rows_are_correct))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过" + ("" if passed == len(results) else "  ← 还没写对"))


if __name__ == "__main__":
    _selftest()
    try:
        main()
    except NotImplementedError:
        print("\n完成三个函数后，再运行完整实验。")
