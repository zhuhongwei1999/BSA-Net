# -*- coding: utf-8 -*-
import os
import sys
import time

cmd1 = 'python MyTrain.py'
cmd2 = 'python MyTest.py'
cmd4 = 'python evaluation.py'

if __name__ == '__main__':
    os.system(cmd1)
    os.system(cmd2)
    os.chdir('evaluation')
    os.system(cmd4)