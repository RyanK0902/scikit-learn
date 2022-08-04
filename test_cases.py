import numpy as np
import time

from sklearn.datasets import load_digits
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import pyximport
pyximport.install()

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
            max_depth=1,    # setting to 2 will give you decision stump
            random_state=seed)
    else:
        print("\n------ training tree WITHOUT histogram ------")
        tree = DecisionTreeClassifier(
            splitter="best",
            criterion="gini",
            max_depth=1,  # setting to 1 will give you decision stump
            random_state=seed)

    # fit and test tree
    tree.fit(train_image, train_target)
    score = tree.score(test_image, test_target)
    print("=> score: ", score)

def tree_complexity(seed, size, is_histogram):
    np.random.seed(seed)
    data, target = make_classification(
        n_samples=size,
        n_features=100,
        n_classes=10,
        n_informative=10,
        random_state=seed
    )
    msk = np.random.rand(len(data)) < 0.8
    train_data, test_data = data[msk].copy(), data[~msk].copy()
    train_target, test_target = target[msk].copy(), target[~msk].copy()
    train_data = train_data.reshape((len(train_data), -1))
    test_data = test_data.reshape((len(test_data), -1))

    # fitting tree
    if is_histogram:
        print("\n------ training tree WITH histogram ------")
        tree = DecisionTreeClassifier(
            splitter="histogram",
            criterion="hist_gini",
            max_depth=1,  # setting to 1 will give you decision stump
            random_state=seed)
    else:
        print("\n------ training tree WITHOUT histogram ------")
        tree = DecisionTreeClassifier(
            splitter="best",
            criterion="gini",
            max_depth=1,  # setting to 1 will give you decision stump
            random_state=seed)

    # fit and test tree
    tree.fit(train_data, train_target)
    score = tree.score(test_data, test_target)
    print("=> score: ", score)


if __name__ == "__main__":
    np.random.seed(0)
    start = time.time()
    # remember to change batch_size to 50!
    tree_mnist(0, is_histogram=False)

    # remember to change batch_size to 1000!
    #tree_complexity(0, size=1000000, is_histogram=False)
    end = time.time()
    print("====>>>> RUNTIME sklearn: ", end - start)

    start = time.time()
    tree_mnist(0, is_histogram=True)

    tree_complexity(0, size=1000000, is_histogram=True)
    end = time.time()
    print("====>>>> RUNTIME histogram: ", end - start)

