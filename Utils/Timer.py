# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:54:10 2019

@author: 13383861
"""

import time

def timed(fn):
    def return_fn(*args, **kwargs):
        t1 = time.time()
        res = fn(*args, **kwargs)
        print("Time taken to run " + str(fn.__name__) + ": " + str(time.time() - t1) + " seconds.")
        return res
    return return_fn


if '__name__' == '__main__':
    @timed
    def sq(x):
        time.sleep(1)
        return x**2
    sq(100)

    