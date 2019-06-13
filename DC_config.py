# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:21:47 2019

@author: shlom

Configuration
"""
import configparser
from DC_logger import DC_LOG

class configuration:
    def __init__(self,config_file = ""):
        
        self.name = "DC_config"
        DC_LOG(self.name, "DC_config constructor!")
        if not config_file:
            # ---- model parametes --- #
            self.trainable_layers = []
            self.retrain_epochs = 5
            
            # ---- pruning hyperparameters ---- #
            self.pruning_delta = 0.0005       #increase in the thershold
            self.pruning_accuracy_thr = 3   #determine accuracy threshold decline - 3%
           
            # ---- quantization hyperparameters ---- #
            self.quantization_index_bits = 5  #number of bits to represent the distance between non-zero elements in csr format
            self.quantization_clusters = 15   #number of clusters for K-menas
            
            # ---- RLE hyperparameters ---- #
            self.rle_function = "elias_gamma"
            
            # ---- Deep Compression ----- #
            self.P = True       #pruning
            self.Q = True       #quantization
            self.H = True       #huffman
            self.RLE = False
            self.csr = True
            
        else:
            config = configparser.ConfigParser()
            config.read(config_file)
            DC_LOG(self.name, "config DC_config using" + config_file + " file..." )
            # ---- model parametes --- #
            self.trainable_layers = config['MODEL']['TRAINABLE_LAYERS'].split(",")
            for i,layer in enumerate(self.trainable_layers):
                self.trainable_layers[i] = int(layer) 
            self.retrain_epochs = int(config['MODEL']['RETRAIN_EPOCHS'])
            
            # ---- pruning hyperparameters ---- #
            self.pruning_delta =float(config['PRUNING']['DELTA'])       #increase in the thershold
            self.pruning_accuracy_thr = float(config['PRUNING']['ACCURACY_THR'])   #determine accuracy threshold decline - 3%
           
            # ---- quantization hyperparameters ---- #
            self.quantization_index_bits = int(config['QUANTIZATION']['INDEX_BITS'])  #number of bits to represent the distance between non-zero elements in csr format
            self.quantization_clusters = int(config['QUANTIZATION']['CLUSTERS_NUM'])   #number of clusters for K-menas
            self.quantization_batch_size = int(config['QUANTIZATION']['BATCH_SIZE'])
            # ---- RLE hyperparameters ---- #
            self.rle_function = config['RLE']['RLE_FUNC'].split(",")
            
            # ---- Deep Compression ----- #
            self.P = False if config['DEEP_COMPRESSION']['P'] == 'False' else True      #pruning
            self.Q = False if config['DEEP_COMPRESSION']['Q'] == 'False' else True      #quantization               
            self.H = False if config['DEEP_COMPRESSION']['H'] == 'False' else True      #huffman         
            self.RLE = False if config['DEEP_COMPRESSION']['RLE'] == 'False' else True  #RLE
            self.csr = False if config['DEEP_COMPRESSION']['CSR'] == 'False' else True  #csr format
            
            # one of RLE & csr have to be set to True
            if((self.RLE == False) and (self.csr == False)):
               self.csr = True         
                   

      
