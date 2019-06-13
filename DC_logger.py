# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:45:07 2019

@author: shlom
"""

import logging

#self.logger = logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []

# create a file handler
handler = logging.FileHandler('sim\\run.log', mode='w')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

def DC_LOG(module_name , massage):
    massage1 = module_name + " - " + massage
    logger.info(massage1)
    
    
if __name__ == '__main__':  
    #logger.info("test1")
    DC_LOG("test","this is a test")