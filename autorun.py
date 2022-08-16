# -*- coding: utf-8 -*-
import os

train_cmd = 'python MyTrain.py'
test_cmd = 'python MyTest.py'
eval_cmd = 'python evaluation.py'

if __name__ == '__main__':
    # os.system(train_cmd)
    os.system(test_cmd)
    os.chdir('evaluation')
    os.system(eval_cmd)