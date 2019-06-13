# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:29:26 2019

@author: shlom

main function
""" 
import DC_logger
from DC_logger import DC_LOG
from DC_config import configuration
from VGG_D import cifar10vggD
from Deep_Compression import DeepCompression

if __name__ == '__main__':
    
    #build configuration class using config.ini file
    ini_file = ".\\config.ini"
    config = configuration(ini_file) 
   
    #buikd VGG16 model
    model = cifar10vggD(config, train = False)
    #create instance of DeepCompression class
    compress_model = DeepCompression(model,config)
    #compress the model
    compress_model.deep_compression()