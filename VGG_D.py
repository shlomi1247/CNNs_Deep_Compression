# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:41:52 2019

@author: shlom
"""


import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from DC_logger import DC_LOG

class cifar10vggD:
    #constructor
    def __init__(self,config, train=True, clusters = 16):
        self.name = "cifar10vggD"
        DC_LOG(self.name, "cifar10vggD constructor!")
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.epochs_num = 15
        self.model = self.build_model()
        self.model.summary()
        self.threshold = self.set_threshold()
        self.trainable_layers =config.trainable_layers      #list with the number of trainable layers
        
        #build model dataset
        DC_LOG(self.name, "build cifar10 dataset")
        (self.x_train, self.y_train), (self.x_test,self. y_test) = self.build_dataset()
        
        #train the model or load weights
        #if train:
        #    self.model = self.train(self.model, max_epochs = self.epochs_num)
        #else:
        #    self.model.load_weights('cifar10vgg.h5')
        self.model = self.train(self.model, train, max_epochs = self.epochs_num) 
        
    def build_dataset(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        return (x_train, y_train), (x_test, y_test)
    
    def set_threshold(self):
        threshold = np.array([0.58,0.22,0.34,0.36,0.53,0.24,0.42,0.32,0.27,0.34,0.35,0.29,0.36,0.04,0.5]) #missing one layer and from some reason missing weights
        threshold = 1 - threshold #missing one layer and from some reason missing weights
        return threshold
        
    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model
           
        
    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model,train, max_epochs):
        #training parameters
        batch_size = 128
        maxepoches = max_epochs
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
           
        # The data, shuffled and split between train and test sets:
        #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #x_train = x_train.astype('float32')
        #x_test = x_test.astype('float32')
        #x_train, x_test = self.normalize(x_train, x_test)

        #y_train = keras.utils.to_categorical(y_train, self.num_classes)
        #y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

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
        datagen.fit(self.x_train)
        
        if train:
            #optimization details
            sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    
    
            # training process in a for loop with learning rate drop every 25 epoches.
    
            
            historytemp = model.fit_generator(datagen.flow(self.x_train, self.y_train,
                                             batch_size=batch_size),
                                steps_per_epoch=self.x_train.shape[0] // batch_size,
                                epochs=maxepoches,
                                validation_data=(self.x_test, self.y_test),callbacks=[reduce_lr],verbose=3)
            model.save_weights('cifar10vggD.h5')
            scores = self.model.evaluate(self.x_test, self.y_test, verbose=3)
            accuracy = scores[1]*10
        else:
            DC_LOG(self.name, "train = False -> compiling the model.")
            self.model.load_weights('cifar10vggD.h5')
            #optimization details
            sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
            scores = self.model.evaluate(self.x_test, self.y_test, verbose=3)
            accuracy = scores[1]*100
            DC_LOG(self.name, "model compiled!")
            DC_LOG(self.name, "accuracy = " + str(accuracy))
        return model
    
    
    def retrain(self, max_epochs,mask):
        #training parameters
        batch_size = 128
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
           
        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

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
        datagen.fit(self.x_train)
        
        #optimization details
        learning_rate = 0.0001
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

        # training process in a for loop with learning rate drop every 25 epoches.
        
        #masking
        #weights = self.model.get_weights()
        #for i,num in enumerate(weights):
        #   weights[i] = np.multiply(weights[i], mask[i])  
        #self.model.set_weights(weights)  
        
        for i in range(0,max_epochs):
            self.model.fit_generator(datagen.flow(self.x_train, self.y_train,
                                             batch_size=batch_size),
                                steps_per_epoch=self.x_train.shape[0] // batch_size,
                                epochs=1,
                                validation_data=(self.x_test, self.y_test),callbacks=[reduce_lr],verbose=3)      
            weights = self.model.get_weights()
            #masking
            for i,num in enumerate(weights):
               weights[i] = np.multiply(weights[i], mask[i])                          
        
        #weights = np.multiply(weights, mask)
        self.model.set_weights(weights)        
        return
    
    def summary(self):
        self.model.summary()
        
    def get_weights(self):
         return self.model.get_weights()
     
    def save_weights(self,filename):
        self.model.save_weights(filename)
    
    def load_weights(self,filename):
        self.model.load_weights(filename)
    
    def set_weights(self,weights):
        self.model.set_weights(weights)
    
    def get_accuracy(self):
        #accuracy = self.model.evaluate(self.x_test, self.y_test)
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=3)
        accuracy = scores[1]*100
        return accuracy
    
    