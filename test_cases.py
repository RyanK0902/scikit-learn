import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def forest_digits_vanilla(seed):
    np.random.seed(seed)
    digits = load_digits()

    # preparing the dataset
    msk = np.random.rand(len(digits.images)) < 0.8
    train_image, test_image = digits.images[msk].copy(), digits.images[~msk].copy()
    train_target, test_target = digits.target[msk].copy(), digits.target[~msk].copy()
    train_image = train_image.reshape((len(train_image), -1))
    test_image = test_image.reshape((len(test_image), -1))

    # training forest
    print("=> training forest")
    rf = RandomForestClassifier()
    rf.fit(train_image, train_target)

    # testing forest
    score = rf.score(test_image, test_target)
    print("=> score: ", score)


def tree_digits_vanilla(seed, is_histogram):
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
        tree = DecisionTreeClassifier(splitter="histogram", random_state=seed)
    else:
        print("\n------ training tree WITHOUT histogram ------")
        tree = DecisionTreeClassifier(random_state=seed)

    # fit and test tree
    tree.fit(train_image, train_target)
    score = tree.score(test_image, test_target)
    print("=> score: ", score)


if __name__ == "__main__":
    tree_digits_vanilla(0, is_histogram=False)
    tree_digits_vanilla(0, is_histogram=True)
