# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 21:30
# @Author  : Eric Lee
# @Email   : li.yan_li@neusoft.com
# @File    : metrics.py
# @Software: PyCharm
def accuracy_score(y_true, y_predict):
    return sum(y_predict==y_predict)/len(y_true)