# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:57:08 2019

Deep compression

@author: sambar

version: 1.0
"""
#sambar
import numpy as np
from keras import backend as K
from sklearn.cluster import KMeans
from UseHuffman import use_huffman
from dectobinary import dec2bin,txt2binfile
from VGG_D import cifar10vggD
import rle
from rle import golomb
#from rle import elias_gamma,elias_delta,golomb
from DC_config import configuration
import math
from DC_logger import DC_LOG
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

class DeepCompression:
    def __init__(self, model,config):
        
        self.name = "Deep_Compression"
        self.model = model
        
        # ---- model parametes --- #
        self.trainable_layers = config.trainable_layers      #list with the number of trainable layers
        self.epochs = config.retrain_epochs                 #nuber of epochs fro retraining
        
        # ---- pruning hyperparameters ---- #
        self.delta = config.pruning_delta                   #increase in the thershold
        self.accuracy_thr = config.pruning_accuracy_thr     #determine accuracy threshold decline
       
        # ---- quantization hyperparameters ---- #
        self.index_bits = config.quantization_index_bits    #number of bits to represent the distance between non-zero elements in csr format
        self.clusters = config.quantization_clusters        #number of clusters for K-menas
        self.batch_size = config.quantization_batch_size
        # ---- RLE hyperparameters ---- #
        self.rle_function = config.rle_function
        
        # ---- Deep Compression ----- #
        self.P = config.P       #pruning
        self.Q = config.Q       #quantization
        self.H = config.H       #huffman
        self.RLE = config.RLE
        self.csr = config.csr
        
        self.GraphBaseDir = "sim\\graphs\\"
          
        DC_LOG(self.name, "DeepCompression class constructor!")
        self.print_config()

    def deep_compression(self):
        DC_LOG(self.name, "starting deep compression the network...\n\n")      
        if self.P:
            self.pruning()
        if self.Q:
            self.quantiztion()

    #===============================================#
    #                   Pruning                     #
    #===============================================#   
    
    def pruning(self):
        DC_LOG(self.name, "Pruning stage starts!\n")
        prune_number = 1
        weights = self.model.get_weights()
        orig_weights = self.model.get_weights()
        accuracy = self.model.get_accuracy() #get the model accuracy for the current weights
        threshold = [] #list that contain the pruning threshold for each trainable layer       
        #threshold = np.array()
        self.mask = []
        for i,num in enumerate(weights):
            self.mask.append(np.ones(weights[i].shape))   #build list that will contain masking for each trainable layer 
      
        # lists for log file.        
        total_weights = []
        zero_elements = []
        sparsity_precent = []
        layers_std = []
        layers_mean = []
        
        GraphSubeDir  = "before_pruning\\"
          
        #plot and save the images 
        self.plot_all_layers(self.GraphBaseDir + GraphSubeDir, "before Pruning", 50)
         
        #first pruning
        for i,num in enumerate(self.trainable_layers):
            layer = weights[num]
            #total weights for each layer
            total_weights.append(layer.size)
            #update threshold
            threshold.append(self.model.threshold[i]*layer.std()*0.3)
            #set std & mean
            layers_std.append(np.std(layer))
            layers_mean.append(np.mean(layer))
            #prune the layer
            layer[np.abs(layer)< threshold[i]] = 0.0 
            #update mask
            tmp_mask = self.mask[num]
            tmp_mask[np.abs(layer) == 0] = 0.0
            #compute information about the first iteration
            zero_elements.append(total_weights[i]-np.count_nonzero(layer))
            spasity = zero_elements[i]/total_weights[i]
            sparsity_precent.append(str(int(spasity*100))+"%")

        
        #iteration conection remove - first iteration its the number of zeros    
        iter_rem = zero_elements[:]
        #set the origin std of the layers - this factor is mult with the delta parameter.
        origin_std = np.copy(layers_std[:])
        origin_std = origin_std*self.model.threshold
        
        DC_LOG(self.name, "#############################################")
        DC_LOG(self.name, "                 PRUNING - " )
        DC_LOG(self.name, "#############################################")
        
        DC_LOG(self.name, "====================================")
        DC_LOG(self.name, "Prune number - " + str(prune_number))
        DC_LOG(self.name, "====================================")
        #convert the threshold to numpy array type
        threshold = np.array(threshold)
        
        DC_LOG(self.name, "Delta = " + str(self.delta))   
        DC_LOG(self.name, "Layers thrsholds:       " + str(threshold))
        DC_LOG(self.name, "Layers std:             " + str(layers_std))
        DC_LOG(self.name, "Layers mean:            " + str(layers_mean))
        DC_LOG(self.name, "Total weights:          " + str(total_weights))
        DC_LOG(self.name, "Total zeros    " + str(sum(zero_elements)))
        DC_LOG(self.name, "Zeros in each layer:    " + str(zero_elements))
        DC_LOG(self.name, "Removed this iteration: " + str(iter_rem))
        DC_LOG(self.name, "Sparsity precentage:    " + str(sparsity_precent))
        
        
        self.model.set_weights(weights)
        prune_drop = self.model.get_accuracy()
        DC_LOG(self.name, "accuracy without retraining: " + str(prune_drop))
        
        #retraining the model to fine-tune the remaining weights.
        DC_LOG(self.name, "Retraining the network.")
        self.model.retrain(self.epochs, self.mask)

        new_accuracy = self.model.get_accuracy()
        DC_LOG(self.name, "new accuracy = " + str(new_accuracy))
        
        accuracy_loss = accuracy - new_accuracy
        DC_LOG(self.name, "accuracy_loss = " + str(accuracy_loss))
        
        #graphs parameters
        GraphAccuracyLossY = np.array([accuracy_loss])
        sparsity = (sum(zero_elements)/sum(total_weights))*100
        GraphSparsityX = np.array([sparsity])
                         
        #save valid copy of the masking and the weights before prune again.
        last_weights = np.copy(self.model.get_weights())
        last_mask = self.mask[:]
        last_zero_elements = zero_elements[:]
        #prune more if there's no significant loss of accuracy
        if(accuracy_loss < self.accuracy_thr):
            #update threshold
            threshold = threshold + self.delta*origin_std

            
            #---------------------- Pruning Loop ----------------------#
            
            while(self.delta > 1e-5):
                prune_number = prune_number + 1
                self.model.set_weights(orig_weights)
                for z in range(0,len(self.mask)):
                    self.mask[z].fill(1)
                weights = self.model.get_weights()
                DC_LOG(self.name, "accuracy_loss was less than the loss_threshold - update threshold and prune again.\n")
                DC_LOG(self.name, "====================================")
                DC_LOG(self.name, "Prune number - " + str(prune_number))
                DC_LOG(self.name, "====================================")
                DC_LOG(self.name, "Delta = " + str(self.delta))
                DC_LOG(self.name, "Layers thrsholds: " + str(threshold))
                #pruning
                for i,num in enumerate(self.trainable_layers):
                    layer = weights[num]
                    #DC_LOG(self.name, "DEBUG: layer " + str(num)+" zeros before " + str(total_weights[i]-np.count_nonzero(layer)))
                    layers_std[i] = np.std(layer)
                    layers_mean[i] = np.mean(layer)
                    #prune the layer
                    layer[np.abs(layer)< threshold[i]] = 0.0 
                    #DC_LOG(self.name, "DEBUG: layer " + str(num)+" zeros after " + str(total_weights[i]-np.count_nonzero(layer)))
                    #update mask
                    tmp_mask = self.mask[num]
                    tmp_mask[np.abs(layer) == 0] = 0.0
                    #compute information about the first iteration
                    iter_rem[i] = last_zero_elements[i]
                    zero_elements[i] = total_weights[i]-np.count_nonzero(layer)
                    iter_rem[i] = zero_elements[i] - iter_rem[i] 
                    spasity = zero_elements[i]/total_weights[i]
                    sparsity_precent[i] = str(int(spasity*100)) + "%"

                
                # print pruning data to log file 
                DC_LOG(self.name, "Layers std:             " + str(layers_std))
                DC_LOG(self.name, "Layers mean:            " + str(layers_mean))                              
                DC_LOG(self.name, "Total weights:          " + str(total_weights))
                DC_LOG(self.name, "Total CNN weights:          " + str(sum(total_weights)))
                DC_LOG(self.name, "Total zeros    " + str(sum(zero_elements)))
                DC_LOG(self.name, "Zeros in each layer:    " + str(zero_elements))
                DC_LOG(self.name, "Removed this iteration: " + str(iter_rem))
                DC_LOG(self.name, "Sparsity precentage:    " + str(sparsity_precent))
                
                
                self.model.set_weights(weights)
                prune_drop = self.model.get_accuracy()
                DC_LOG(self.name, "accuracy without retraining: " + str(prune_drop))
                #retraining  
                self.model.retrain(self.epochs, self.mask)
                #update the accuracy
                new_accuracy = self.model.get_accuracy()
                DC_LOG(self.name, "new accuracy = " + str(new_accuracy))
                accuracy_loss = accuracy - new_accuracy
                DC_LOG(self.name, "accuracy_loss = " + str(accuracy_loss))
                
                #update threshold
                if (accuracy_loss < self.accuracy_thr):
                    DC_LOG(self.name, "accuracy_loss was less than the loss_threshold - update threshold and prune again.\n")
                    #save the last weights
                    last_weights = np.copy(self.model.get_weights())
                    #save the mask
                    last_mask = self.mask[:]
                    last_zero_elements = zero_elements[:]
                    #update graphs parameters
                    GraphAccuracyLossY = np.append(GraphAccuracyLossY, accuracy_loss)
                    sparsity = (sum(zero_elements)/sum(total_weights))*100
                    GraphSparsityX = np.append(GraphSparsityX, sparsity)
                    
                    DC_LOG(self.name, "thr = thr + " + str(self.delta*origin_std))
                    threshold = threshold + self.delta*origin_std
                else:
                    DC_LOG(self.name, "accuracy_loss has exceeded the loss_threshold - update delta and prune again.\n")
                    self.mask =  last_mask[:]
                    self.delta = self.delta/2;
                    DC_LOG(self.name, "thr = thr - " + str(self.delta*origin_std))
                    threshold = threshold - self.delta*origin_std
                    
            #-------------------------------------------------------------------#
        
    
        
        #save the masking to a file - set it just for using save_weights method.        
        self.mask =  last_mask[:]
        self.model.set_weights(self.mask)
        self.model.save_weights('masking.h5')
        DC_LOG(self.name, "saving weights masking to masking.h5 file...")
        
        #set the last valid weights and save them to file.
        self.model.set_weights(last_weights)
        self.model.save_weights('pruned_weights.h5')
        DC_LOG(self.name, "saving weights to pruned_weights.h5 file...")
        
        np.save("sparsity",GraphSparsityX)
        np.save("loss",GraphAccuracyLossY)
        
        DC_LOG(self.name, "Done pruning  \n")
        DC_LOG(self.name, "sparsity vetor:  \n")
        DC_LOG(self.name, str(GraphSparsityX))
        DC_LOG(self.name, "accuracy loss vetor:  \n")
        DC_LOG(self.name, str(GraphAccuracyLossY))
        
        # plot accuracy loss - sprsity graph
        plt.plot(GraphSparsityX,GraphAccuracyLossY , marker ='^')
        plt.xlabel("model size ratio after pruning")
        plt.ylabel("Accuracy loss")
        plt.title("Pruning\n")
        plt.grid( axis='x')
        plt.grid( axis='y',linestyle='--')
        plt.savefig(self.GraphBaseDir + "accuracy_loss.png", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
        plt.show() 
        
        
        #plot and save the images 
        GraphSubeDir  = "after_pruning\\"
        self.plot_all_layers(self.GraphBaseDir + GraphSubeDir, "after Pruning", 50)
        
        DC_LOG(self.name, "Pruning stage finshed!\n\n\n")
        

    #===============================================#
    #       Quantiztion & Weights Sharing           #
    #===============================================#  
           
    def quantiztion(self):
        
        DC_LOG(self.name, "Quantiztion stage starts!\n")
        self.SimBaseDir = "sim\\"
        if not self.P:
            #load maskong list
            self.model.load_weights('masking.h5')
            self.mask = self.model.get_weights()
            #load weights
            self.model.load_weights('pruned_weights.h5')
            weights = self.model.get_weights()
            
            
            
        if(self.save_pruned):
            for i,num in enumerate(self.trainable_layers):                      
                # ------- csr format with RLE --------#
                prun_layer = np.ndarray.flatten(weights[num])
                
                if(self.RLE):
                    for rle_func in self.rle_function:
                        DC_LOG(self.name, "output layer " + str(i) + " in csr format using rle " + str(rle_func) + " compression method")
                        bits_per_value = int(math.log(self.clusters + 1,2)) # calculate bits per non zero value
        
                        rle_method = getattr(rle,str(rle_func))
                        
                        data = ""
                        diff = 0
                        for element in range(0,prun_layer.size):
                            #counnt the number of zeros between non zero elements
                            if (prun_layer[element] == 0):
                                diff = diff + 1
                                continue
                            else:
                                #write the number of zeros with rle 
                                #data = data + globals()['rle_method'](diff)
                                data = data + rle_method(diff)
                                data = data + dec2bin(int(prun_layer[element]),32)
                                diff = 0
                        layer_weights_file = self.SimBaseDir + "pruned_layer" +str(i) + rle_func + ".txt"
                        DC_LOG(self.name, "writing layer " + str(i) + " csr format using " + rle_func + " to file: " + layer_weights_file +"...\n")
                        with open(layer_weights_file, 'w') as file:
                            file.write(data)        
                        
                        #convert txt file to .dat file
                        txt2binfile(str(layer_weights_file))
                        
                        #hufmman coding     
                        if self.H:
                            DC_LOG(self.name, "Huffman coding on " + layer_weights_file +"file")
                            huffman_file = self.huffman_coding(layer_weights_file)
                            DC_LOG(self.name, "converting " + str(huffman_file) + " to .dat file") 
                            txt2binfile(str(huffman_file))
                        
                        
                # ------- csr format -------- #
                if(self.csr):
                    DC_LOG(self.name, "output layer " + str(i) + "in csr format")
                    # compute parameters
                    max_diff = pow(2,self.index_bits) - 1 #max non zero diff we can represent in index_bits parameter
                    bits_per_value = int(math.log(self.clusters + 1,2)) #calculate bits per non zero value
                    
                    #optput to file comress weights - csr format 
                    data = ""
                    diff = 0
                    for element in range(0,prun_layer.size):
                        if (prun_layer[element] == 0):
                            diff = diff + 1
                            continue
                        else:
                            #write diff - if the diff distance between non zero elements is
                            #greater than 32 (31 zeros in between) write 31 diff with value 0000 - 
                            #if we see 31 diff with 0000 its means its a filler
                            while( diff > max_diff ):
                                #dec2bin(decimal number, number of bits to present)
                                data = data + dec2bin(max_diff,self.index_bits) #31 zeros between
                                data = data + dec2bin(0,32)  #value 0000 
                                diff = diff - max_diff
                            data = data + dec2bin(diff,self.index_bits)
                            data = data + dec2bin(int(prun_layer[element]),32)
                            diff = 0
                    layer_weights_file = self.SimBaseDir + "pruned_layer" +str(i) + "csr.txt"
                    DC_LOG(self.name, "writing layer " + str(i) + " csr format to file" + layer_weights_file +"...\n")
                    with open(layer_weights_file, 'w') as file:
                        file.write(data)
                    #convert txt file to .dat file
                    txt2binfile(str(layer_weights_file))  
                                     
                    #hufmman coding     
                    if self.H:
                        DC_LOG(self.name, "Huffman coding on " + layer_weights_file +"file")
                        huffman_file = self.huffman_coding(layer_weights_file)
                        txt2binfile(str(huffman_file))        
        
        

        #============ end of saving the pruned weights ============       
        

        #------declare lists-----#
        labels = []
        layers_hash_table = [] #list with hash tables for each layers that contain: {label:centroid} 
        hash_table = []
        layers_quantize = []
        
        weights = self.model.get_weights()
        
        DC_LOG(self.name, "accuracy before quantization & weight sharing: " +str(self.model.get_accuracy()) + "\n")
        
        for i,num in enumerate(self.trainable_layers):
            DC_LOG(self.name, "====================================")
            DC_LOG(self.name, "Quantize layer number:  " + str(i))
            DC_LOG(self.name, "====================================")
            #span the nD matix to 1D vector
            flat_layer = np.ndarray.flatten(weights[num])
            #weight sharing fot the non zero values
            values = flat_layer[flat_layer != 0]
            kmeans = KMeans(n_clusters=self.clusters, random_state=0).fit(values.reshape(-1,1))
            centroids = kmeans.cluster_centers_ #the values of the quantized weights
            cen_file =self.SimBaseDir + "centroids" + str(i) + ".txt"
            with open(cen_file, "wb") as fp:   #Pickling
                pickle.dump(centroids, fp)
            labels.append(kmeans.labels_) #label for each centroid
            
            DC_LOG(self.name, "layer centroids: \n " + str(centroids))
            #build a copy of the layer with it quantization values and zeros.
            quantize = np.copy(flat_layer)
            value_idx=0
            for j in range(0,quantize.size):
                if quantize[j] == 0:
                    continue
                else:
                    quantize[j] = labels[i][value_idx]+1 #label 0 -> 1 and so on...
                    value_idx = value_idx + 1
             #save the quantize layer       
            layers_quantize.append(np.array(quantize).reshape(weights[num].shape))
             
            # --- build hash table ---
            hash_table = np.ndarray.flatten(centroids)
            hash_table = np.insert(hash_table,0,0)
            layer_hash_file = self.SimBaseDir + "layer" +str(i) + "_hash.txt"
            DC_LOG(self.name, "writing centroids hash table to file" + layer_hash_file +"...\n")
            with open(layer_hash_file, 'w') as file:
                for label,centroid in enumerate(hash_table):
                    line = str(label)+" : " + str(centroid) + "\n"
                    file.write(str(line))
            layers_hash_table.append(hash_table)    
             
            # update weights with centroids
            for k in range(1,self.clusters+1):
                weights[num][layers_quantize[i] == k] = hash_table[k]
                
        
        self.model.set_weights(weights)
        DC_LOG(self.name, "accuracy after quantization & weight sharing: " + str(self.model.get_accuracy()) + "\n")   
        
        #saving the weights
        self.model.save_weights('quantized_weights.h5')
        
# =============================================================================
#         #plot and save the images 
#         GraphSubeDir  = "after_quantization\\"
#         self.plot_all_layers(self.GraphBaseDir + GraphSubeDir, "after Quantization", 16)
# =============================================================================
        
        #plotting with index
        #for i in range(0,len(layers_quantize)):
        #   self.plot_layer(layers_quantize[i], self.GraphBaseDir + GraphSubeDir, "layer " + str(i)+ "after Quantization", False, 15)
        
            
        #---------------------------------------------------------------#             
        #           Retraining and fine-tune centroids
        #---------------------------------------------------------------# 
        
        DC_LOG(self.name, "weights fine - tuning agter quantization\n")
    
        self.set_retraining_config()
        # build accumulator for the weights gradients.
        total_grad = []
        for num in range(0,len(self.model.model.layers)):     
            name =type(self.model.model.layers[num]).__name__
            if name == "Dense" or name == "Conv2D":
                total_grad.append(np.zeros(self.model.model.layers[num].get_weights()[0].shape))
        
        temp_weights = np.copy(weights)
        for z in range(0, 513,self.batch_size): #TODO need to change it to xtrain size
            self.model.set_weights(temp_weights) 
            end_idx = min(z + self.batch_size - 1, 50000 -1)
            idx = np.arange(z,end_idx)
            batch_samples = np.take(self.model.x_train, idx, axis=0, out=None, mode='raise')
            batch_labels = np.take(self.model.y_train, idx, axis=0, out=None, mode='raise') 
            
            loss = self.model.model.train_on_batch(batch_samples, batch_labels) 
    
            weight_grads = self.get_weight_grad(self.model.model, batch_samples, batch_labels)
            
            #output_grad = self.get_layer_output_grad(model.model, batch_samples, batch_labels)
            
            temp_weights = self.model.get_weights()
            #sum gradients
            for k in range(0,len(self.trainable_layers)):
                total_grad[k] = total_grad[k] + weight_grads[4*k] #TODO
            
            DC_LOG(self.name, "accuracy in iterate " + str(z) + ": " + str(self.model.get_accuracy()) + "\n") 
                   
#======================DEBUG PART===================================            
            #loop over all the layers
            for i in range(0,len(self.trainable_layers)): 
                #loop over each cluster from the codebook.   
                for j in range(1,self.clusters+1): 
                        group_grad = 0
                        group_grad = np.sum(total_grad[i][layers_quantize[i]==j])
                        layers_hash_table[i][j] = layers_hash_table[i][j] - group_grad*0.00001 #TODO
            DC_LOG(self.name, "layer " + str(i) + "centroids: \n" + str(layers_hash_table[i]))
            
            # update weights with centroids            
            for i,num in enumerate(self.trainable_layers):
                for k in range(0,self.clusters+1):
                    weights[num][layers_quantize[i] == k] = layers_hash_table[i][k]
                
            self.model.set_weights(weights)
            DC_LOG(self.name, "accuracy in iterate " + str(z) + " with weight sharing: " + str(self.model.get_accuracy()) + "\n")
 

       
# =============================================================================
#         update hash tables
# =============================================================================
     
        #loop over all the layers
        for i in range(0,len(self.trainable_layers)): 
            #loop over each cluster from the codebook.   
            for j in range(1,self.clusters+1): 
                        group_grad = 0
                        group_grad = np.sum(total_grad[i][layers_quantize[i]==j])
                        layers_hash_table[i][j] = layers_hash_table[i][j] - group_grad*0.00001 #TODO
            DC_LOG(self.name, "layer " + str(i) + "centroids: \n" + str(layers_hash_table[i]))

        #save updated hash tables to files
        for i in range(0,len(self.trainable_layers)):
            layer_hash_file = self.SimBaseDir + "layer" +str(i) + "_hash_retrained.txt"
            DC_LOG(self.name, "writing retrained centroids hash table to file" + layer_hash_file +"...\n")
            with open(layer_hash_file, 'w') as file:
                for label,centroid in enumerate(layers_hash_table[i]):
                    line = str(label)+" : " + str(centroid) + "\n"
                    file.write(str(line))
                    
        # update weights with centroids            
        for i,num in enumerate(self.trainable_layers):
            for k in range(0,self.clusters+1):
                weights[num][layers_quantize[i] == k] = layers_hash_table[i][k]
                
        self.model.set_weights(weights)
        DC_LOG(self.name, "accuracy after fine - tune centroids: " + str(self.model.get_accuracy()) + "\n") 
            
        self.model.save_weights('quantized_retrained_weights.h5')             
        
 

          
# =============================================================================
#                    Saving the sparse matrix & Huffman coding
# =============================================================================
           
            
        for i,num in enumerate(self.trainable_layers):                      
            # ------- csr format with RLE --------#
            quantize = np.ndarray.flatten(layers_quantize[i])
            
            if(self.RLE):
                for rle_func in self.rle_function:
                    DC_LOG(self.name, "output layer " + str(i) + " in csr format using rle " + str(rle_func) + " compression method")
                    bits_per_value = int(math.log(self.clusters + 1,2)) # calculate bits per non zero value
    
                    rle_method = getattr(rle,str(rle_func))
                    
                    data = ""
                    diff = 0
                    for element in range(0,quantize.size):
                        #counnt the number of zeros between non zero elements
                        if (quantize[element] == 0):
                            diff = diff + 1
                            continue
                        else:
                            #write the number of zeros with rle 
                            #data = data + globals()['rle_method'](diff)
                            data = data + rle_method(diff)
                            data = data + dec2bin(int(quantize[element]),bits_per_value)
                            diff = 0
                    layer_weights_file = self.SimBaseDir + "layer" +str(i) + rle_func + ".txt"
                    DC_LOG(self.name, "writing layer " + str(i) + " csr format using " + rle_func + " to file: " + layer_weights_file +"...\n")
                    with open(layer_weights_file, 'w') as file:
                        file.write(data)        
                    
                    #convert txt file to .dat file
                    txt2binfile(str(layer_weights_file))
                    
                    #hufmman coding     
                    if self.H:
                        DC_LOG(self.name, "Huffman coding on " + layer_weights_file +"file")
                        huffman_file = self.huffman_coding(layer_weights_file)
                        DC_LOG(self.name, "converting " + str(huffman_file) + " to .dat file") 
                        txt2binfile(str(huffman_file))
                    
                    
                    
            
            # ------- csr format -------- #
            if(self.csr):
                DC_LOG(self.name, "output layer " + str(i) + "in csr format")
                # compute parameters
                max_diff = pow(2,self.index_bits) - 1 #max non zero diff we can represent in index_bits parameter
                bits_per_value = int(math.log(self.clusters + 1,2)) #calculate bits per non zero value
                
                #optput to file comress weights - csr format 
                data = ""
                diff = 0
                for element in range(0,quantize.size):
                    if (quantize[element] == 0):
                        diff = diff + 1
                        continue
                    else:
                        #write diff - if the diff distance between non zero elements is
                        #greater than 32 (31 zeros in between) write 31 diff with value 0000 - 
                        #if we see 31 diff with 0000 its means its a filler
                        while( diff > max_diff ):
                            #dec2bin(decimal number, number of bits to present)
                            data = data + dec2bin(max_diff,self.index_bits) #31 zeros between
                            data = data + dec2bin(0,bits_per_value)  #value 0000 
                            diff = diff - max_diff
                        data = data + dec2bin(diff,self.index_bits)
                        data = data + dec2bin(int(quantize[element]),bits_per_value)
                        diff = 0
                layer_weights_file = self.SimBaseDir + "layer" +str(i) + "csr.txt"
                DC_LOG(self.name, "writing layer " + str(i) + " csr format to file" + layer_weights_file +"...\n")
                with open(layer_weights_file, 'w') as file:
                    file.write(data)
                #convert txt file to .dat file
                txt2binfile(str(layer_weights_file))  
                                 
                #hufmman coding     
                if self.H:
                    DC_LOG(self.name, "Huffman coding on " + layer_weights_file +"file")
                    huffman_file = self.huffman_coding(layer_weights_file)
                    txt2binfile(str(huffman_file))
              
   
    #----------------------------------#
    #              Huffman             #
    #----------------------------------#       
    
    def huffman_coding(self,file_name): 
        return use_huffman(file_name)
       
    def print_config(self):
        DC_LOG(self.name, "====================================")
        DC_LOG(self.name, "         configuration:  ")
        DC_LOG(self.name, "====================================")
        DC_LOG(self.name, "trainable_layers = " + str(self.trainable_layers))
        DC_LOG(self.name, "pruning_delta = " + str(self.delta))
        DC_LOG(self.name, "pruning_accuracy_thr = " + str(self.accuracy_thr))
        DC_LOG(self.name, "quantization_index_bits = " + str(self.index_bits))
        DC_LOG(self.name, "quantization_clusters = " + str(self.clusters))
        DC_LOG(self.name, "rle_function = " + str(self.rle_function))
        DC_LOG(self.name, "P = " + str(self.P))
        DC_LOG(self.name, "Q = " + str(self.Q))
        DC_LOG(self.name, "RLE = " + str(self.RLE))
        DC_LOG(self.name, "csr = " + str(self.csr))
        
    def plot_all_layers(self, direct , postfix, plot_zero = True, bins_num = 'auto'):
        '''
        plot histogram distribution of all the trainable weights.
        
        direct   - directory for saving the graphs.
        postfix  - title
        bins_num - number of bins for the histogram ( default - 'auto').
        '''
        weights = self.model.get_weights()
        for num in self.trainable_layers:
            title = "Histogram of \nlayer "+ str(num) + " distribution " + postfix
            data = np.ndarray.flatten(weights[num])
            if not plot_zero:
                data = data[data!=0]
            fig, ax = plt.subplots()
            #edgecolor='white'
            ax.hist(data, bins=bins_num)  
            ax.set_title(title)
            ax.set_xlabel("Weights value")
            ax.set_ylabel("Density")
            file_name = direct + "layer " + str(num) + postfix + ".png"
            #plt.savefig(centroids_file_name)
            plt.savefig(file_name, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
            plt.show()     
            
    def plot_layer(self, data, direct , postfix, plot_zero = True, bins_num = 'auto', width = None, grid = False):
        '''
        plot histogram distribution of the data vector.
        
        direct   - directory for saving the graphs.
        postfix  - title
        bins_num - number of bins for the histogram ( default - 'auto').
        '''
        title = "Histogram of \n" + postfix
        if not plot_zero:
            data = data[data!=0]
        fig, ax = plt.subplots()
        #edgecolor='white'
        if grid:
            ax.grid( axis='y',linestyle='--')

        ax.hist(data, bins=bins_num, rwidth = width)
        ax.set_title(title)
        ax.set_xlabel("Weights value")
        ax.set_ylabel("Density")
        file_name = direct + postfix + ".png"
        #plt.savefig(centroids_file_name)
        plt.savefig(file_name, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
        plt.show()  
            
            
    def get_weight_grad(self, model, inputs, outputs):
        """ Gets gradient of model for given inputs and outputs for all weights"""
        grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
        symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        f = K.function(symb_inputs, grads)
        x, y, sample_weight = model._standardize_user_data(inputs, outputs)
        output_grad = f(x + y + sample_weight)
        return output_grad


    def get_layer_output_grad(self, model, inputs, outputs, layer=-1):
        """ Gets gradient a layer output for given inputs and outputs"""
        grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
        symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        f = K.function(symb_inputs, grads)
        x, y, sample_weight = model._standardize_user_data(inputs, outputs)
        output_grad = f(x + y + sample_weight)
        return output_grad
    
    def set_retraining_config(self):
        learning_rate = 0.0001
        lr_decay = 1e-6

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(self.model.x_train)
        
        #optimization details
        learning_rate = 0.0001
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.model.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    
    
#----------main------------#

if __name__ == '__main__':
    
    #build configuration class using config.ini file
    ini_file = ".\\config.ini"
    config = configuration(ini_file) 
   
    #buikd VGG16 model
    model = cifar10vggD(config, train = False)
    #create instance of DeepCompression class
    DC_inst = DeepCompression(model,config)

    weights = model.get_weights()
    data = weights[36]
    #plot and save the images 
    GraphSubeDir  = "before_pruning\\"
    #DC_inst.plot_all_layers(DC_inst.GraphBaseDir + GraphSubeDir, "before Pruning",plot_zero = False)
    DC_inst.plot_layer(data,DC_inst.GraphBaseDir, "Conv layer before Pruning",plot_zero = False,grid = True)
    
    DC_inst.model.load_weights('pruned_weights.h5')
    weights = model.get_weights()
    data = weights[36]
    GraphSubeDir  = "after_pruning\\"
    #DC_inst.plot_all_layers(DC_inst.GraphBaseDir + GraphSubeDir, "after Pruning",plot_zero = False ,bins_num = 1000)
    DC_inst.plot_layer(data,DC_inst.GraphBaseDir, "Conv layer after Pruning & retraining",plot_zero = False,grid = True)
    
    DC_inst.model.load_weights('quantized_weights.h5')
    weights = model.get_weights()
    data = weights[36]
    GraphSubeDir  = "after_quantization\\"
    #DC_inst.plot_all_layers(DC_inst.GraphBaseDir + GraphSubeDir, "after Pruning",plot_zero = False ,bins_num = 1000)
    DC_inst.plot_layer(data,DC_inst.GraphBaseDir, "Conv layer after quatization (3 bits)",plot_zero = False,width = 0.9,grid = True )    
    
  
    DC_inst.model.load_weights('quantized_retrained_weights.h5')
    weights = model.get_weights()
    data = weights[36]
    GraphSubeDir  = "after_quantization\\"
    #DC_inst.plot_all_layers(DC_inst.GraphBaseDir + GraphSubeDir, "after Pruning",plot_zero = False ,bins_num = 1000)
    DC_inst.plot_layer(data,DC_inst.GraphBaseDir, "Conv layer after retraining & fine-tune centoids",plot_zero = False,width = 0.5,grid = True ) 
    

    
    
    