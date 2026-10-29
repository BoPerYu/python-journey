"""Reference experiment: train/test split and decision-tree depth."""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def evaluate_depth(
    x_train,
    x_test,
    y_train,
    y_test,
    max_depth,
) -> tuple[float, float]:
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    return train_accuracy, test_accuracy


def main() -> None:
    iris = load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("train shape:", x_train.shape)
    print("test shape:", x_test.shape)
    print()

    experiments = [
        ("depth 1", 1),
        ("depth 3", 3),
        ("unlimited", None),
    ]
    rows = []
    for label, max_depth in experiments:
        train_accuracy, test_accuracy = evaluate_depth(
            x_train,
            x_test,
            y_train,
            y_test,
            max_depth,
        )
        rows.append((label, train_accuracy, test_accuracy))

    print(f"{'model':<12} {'train_acc':>10} {'test_acc':>10} {'gap':>10}")
    for label, train_accuracy, test_accuracy in rows:
        gap = train_accuracy - test_accuracy
        print(
            f"{label:<12} {train_accuracy:>10.4f} "
            f"{test_accuracy:>10.4f} {gap:>10.4f}"
        )


if __name__ == "__main__":
    main()
