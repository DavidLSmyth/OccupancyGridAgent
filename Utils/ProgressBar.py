# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:49:48 2019

@author: 13383861
"""

import sys
import time

def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r [%s] %s%s ...%s \r' % (bar, percents, '%', status))
    sys.stdout.flush()
            
            
if '__name__' == '__main__':
    for i in range(10):
        time.sleep(1)
        progress(10*i, 100, 'good')