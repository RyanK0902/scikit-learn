import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def tree_mnist(seed, is_histogram):
    np.random.seed(seed)
    digits = load_digits()

    msk = np.random.rand(len(digits.data)) < 0.8
    train_image, test_image = digits.data[msk].copy(), digits.data[~msk].copy()
    train_target, test_target = digits.target[msk].copy(), digits.target[~msk].copy()
    train_image = train_image.reshape((len(train_image), -1))
    test_image = test_image.reshape((len(test_image), -1))

    # fitting tree
    if is_histogram:
        print("\n------ training tree WITH histogram ------")
        tree = DecisionTreeClassifier(
            splitter="histogram",
            criterion="hist_gini",
            max_depth=2,
            random_state=seed)
    else:
        print("\n------ training tree WITHOUT histogram ------")
        tree = DecisionTreeClassifier(random_state=seed)

    # fit and test tree
    tree.fit(train_image, train_target)
    score = tree.score(test_image, test_target)
    print("=> score: ", score)


if __name__ == "__main__":
    tree_mnist(0, is_histogram=True)

