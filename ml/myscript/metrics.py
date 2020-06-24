# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 21:30
# @Author  : Eric Lee
# @Email   : li.yan_li@neusoft.com
# @File    : metrics.py
# @Software: PyCharm
import  numpy as np
from math import sqrt
def accuracy_score(y_true, y_predict):
    return sum(y_predict==y_predict)/len(y_true)
def mean_squared_error(y_true, y_predict):
    return np.sum((y_true-y_predict)**2/len(y_true))
def root_mean_squared_error(y_true, y_predict):
    return sqrt(mean_squared_error(y_true, y_predict))
def mean_absolute_error(y_true, y_predict):
    return np.sum(np.absolute(y_true-y_predict)/len(y_true))
def r2_score(y_true, y_predict):
    return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)