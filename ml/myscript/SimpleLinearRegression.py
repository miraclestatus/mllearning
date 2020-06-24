# -*- coding: utf-8 -*-
# @Time    : 2020/6/22 20:58
# @Author  : Eric Lee
# @Email   : li.yan_li@neusoft.com
# @File    : SimpleLinearRegression.py
# @Software: PyCharm
import numpy as np
from .metrics import r2_score

class SimpleLinearRegression2():
    """向量化"""
    def __init__(self):
        """
        定义a ， b
        """
        self.a_ = None
        self.b_ = None
    def fit(self, X_train, y_train):
        # 求均值
        x_mean = np.mean(X_train)
        y_mean = np.mean(y_train)
        self.a_ = (X_train-x_mean).dot(y_train -y_mean)/(X_train-x_mean).dot(X_train-x_mean)
        self.b_ = y_mean -self.a_*x_mean
        return self

    def predict(self, x_predict):
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_*x_single +self.b_

    def score(self, X_test, y_test):
        """根据测试数据, 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
    def __repr__(self):
        return "SimpleLinearRegression2()"
class SimpleLinearRegression1():
    def __init__(self):
        """
        定义a ， b
        """
        self.a_ = None
        self.b_ = None
    def fit(self, X_train, y_train):
        # 求均值
        x_mean = X_train.mean()
        y_mean = y_train.mean()
        num = 0.0
        d = 0.0
        for x_i, y_i in zip(X_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_*x_single +self.b_

    def score(self, X_test, y_test):
        """根据测试数据, 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression1()"





