from collections import Counter
from math import sqrt
from .metrics import accuracy_score
import numpy as np
class KNeighborsClassifier():
    def __init__(self,k):
        assert k >= 1, 'k must be valid'
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self,X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."
        self._X_train = X_train
        self._y_train = y_train
        return self
    def predict(self,X_predict):
        """给定待预测数据集X_predict，返回标示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "mush fit before predict"
        assert self._X_train.shape[1] == X_predict.shape[1], "the feature number of x must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)


    def _predict(self, x):
        """单个的"""
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearset = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearset[:self.k]]
        return Counter(topK_y).most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据测试数据, 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

