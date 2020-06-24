import numpy as np
def train_test_split(X, y,test_ratio=0.2, seed=None):
    if seed:
        np.random.seed(seed)

    # 构建一定范围的随机排列
    shuffle_indexs = np.random.permutation(len(X))
    # 比例
    test_size = int(len(X) * test_ratio)
    test_indexs = shuffle_indexs[:test_size]
    train_indexs = shuffle_indexs[test_size:]
    X_train = X[train_indexs]
    y_train = y[train_indexs]
    X_test = X[test_indexs]
    y_test = y[test_indexs]
    return X_train, y_train, X_test, y_test