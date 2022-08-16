# -*- coding: utf-8 -*-
# @Time    : 2020/11/27
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import time

import numpy as np

def cal_nonzero(size):
    a = np.random.randn(size, size)
    a = a > 0
    start = time.time()
    print(np.count_nonzero(a), time.time() - start)
    start = time.time()
    print(np.sum(a), time.time() - start)
    start = time.time()
    print(len(np.nonzero(a)[0]), time.time() - start)


if __name__ == '__main__':
    cal_nonzero(1000)
