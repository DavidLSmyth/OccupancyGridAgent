# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:31:34 2018

@author: 13383861
"""

import os
import subprocess
import pathlib

if __name__ == '__main__':
    print("Running file: ", __file__)
    for file in os.listdir(os.path.dirname(os.path.realpath(__file__))):
        #print(pathlib.Path(os.path.dirname(os.path.realpath(__file__)),file))
        #print(pathlib.Path(os.path.dirname(os.path.realpath(__file__)),__file__))
        if pathlib.Path(os.path.dirname(os.path.realpath(__file__)),file) != pathlib.Path(os.path.dirname(os.path.realpath(__file__)),__file__) and file[-2:] == 'py':
            print(pathlib.Path(os.path.dirname(os.path.realpath(__file__)), file))
            subprocess.call(["python", str(pathlib.Path(os.path.dirname(os.path.realpath(__file__)), file))])

