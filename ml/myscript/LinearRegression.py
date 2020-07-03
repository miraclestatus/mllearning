import numpy as np
from .metrics import r2_score
class LinearRegression():
    def __init__(self):
        # 系数
        self.coef_ = None
        # 截距
        self.intercetion_ = None
        # 向量
        self._theta = None
    def fit_normal(self, X_train, y_train):
        """使用正规方程进行训练
        hstack  水平拼接"""
        X_b = np.hstack([np.ones((len(X_train),1)), X_train])
        # X_b.T  np.linalg.inv求矩阵的逆
        self._theta = np.linalg.inv((X_b.T.dot(X_b))).dot(X_b.T).dot(y_train)
        self.intercetion_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        # def dJ(theta, X_b, y):
        #     res = np.empty(len(theta))
        #     res[0] = np.sum(X_b.dot(theta) - y)
        #     for i in range(1, len(theta)):
        #         res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
        #     return res * 2 / len(X_b)
        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        def dJ_sgd(theta, X_b_i, y_i):
            return 2. * X_b_i.T.dot(X_b_i.dot(theta) - y_i)

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)
            # n_iters 代表对样本看了多少遍
            for cur_iter in range(n_iters):
            # for cur_iter in range(n_iters*m):
                # rand_i = np.random.randint(len(X_b))
                indexes = np.random.permutation(m)
                # 保证随机性
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter*m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta= sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]
        return self
    def predict(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
    def __repr__(self):
        return "LinearRegression()"