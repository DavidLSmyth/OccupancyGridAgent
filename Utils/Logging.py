# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:20:10 2018

@author: 13383861
"""

import logging
import sys

def setup_file_logger(name, log_file, formatter, level=logging.INFO):
    handler = logging.FileHandler(log_file, "w+")        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    #for now don't propogate
    logger.propagate = False

    
def setup_command_line_logger(name, formatter, level=logging.DEBUG):
    handler = logging.StreamHandler(sys.stdout)        
    #handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    
def log_msg_to_file(msg, level, log_name, should_log = True): 
    log = logging.getLogger(log_name)
    if level == 'info'    : log.info(msg) 
    elif level == 'warning' : log.warning(msg)
    elif level == 'error'   : log.error(msg)
    elif level == 'debug'   : log.debug(msg)
    #if level == 'data'    : log.data(msg)
    
def log_msg_to_cmd(msg, level, log_name, should_log = True):
    if not should_log:
        return 
    log = logging.getLogger(log_name)
    if level == 'info'    : log.info(msg) 
    elif level == 'warning' : log.warning(msg)
    elif level == 'error'   : log.error(msg)
    elif level == 'debug'   : log.debug(msg)