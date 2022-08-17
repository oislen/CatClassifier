# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:22:57 2021
@author: oislen
"""

# load in relevant libraries
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, AveragePooling2D

def LeNet5(input_shape = (28, 28, 1), 
           n_classes = 10,
           output_activation = 'softmax',
           name = 'LeNet5'
           ):
    
    """
        
    LeNet5 Documentation
    
    Function Overview
    
    This function generates a LeNet5 Model architecture:
        
         1. Conv2D 
             - filters: 32  
             - kernal: 5 x 5 
             - activation: relu 
             - padding: same
             
            MaxPooling2D 
             - pool: 2 x 2 
             
         2. Conv2D 
             - filters: 32  
             - kernal: 5 x 5 
             - activation: relu 
             - padding: same
             
            MaxPooling2D 
             - pool: 2 x 2 
             
            Flatten
         
         3. Dense 
             - units: 128  
             - activation: relu 
             
            Dropout 
             - rate: 0.25 
             
         4. Dense 
             - units: 64  
             - activation: relu 
             
            Dropout 
             - 0.25 rate
             
         5. Dense 
            - units: n classes  
            - activation: output_activation 
        
    Defaults
    
    LeNet5(input_shape = (28, 28, 1), 
           n_classes = 10,
           output_activation = 'softmax',
           name = 'LeNet5'
           )
    
    Parameters
    
    input_shape - the input image shape / dimensions
    n_classes - the number of target classes
    
    Returns
    
    model - keras.Model, the LeNet model
    
    Example
    
    LeNet5(input_shape = (28, 28, 1), 
           n_classes = 10, 
           output_activation = 'softmax', 
           name = 'LeNet5'
           )
    
    """
    
    # set input shapes
    inputs = Input(shape = input_shape)
    
    # first convulation and pooling layer
    conv_layer_1 = Conv2D(filters = 6, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', padding = 'valid')(inputs)
    pool_layer_1 = AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(conv_layer_1)
    
    # second convolution and pooling layer, with flattening layer
    conv_layer_2 = Conv2D(filters = 16, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', padding = 'valid')(pool_layer_1)
    pool_layer_2 = AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(conv_layer_2)
    flat_layer = Flatten()(pool_layer_2)
    
    # first dense layer with dropout regularization
    dense_layer_1 = Dense(units = 120, activation = 'relu')(flat_layer)
    drop_layer_1 = Dropout(0.25)(dense_layer_1)
    
    # second dense layer with dropout regularization
    dense_layer_2 = Dense(units = 84, activation = 'relu')(drop_layer_1)
    drop_layer_2 = Dropout(0.25)(dense_layer_2)
    
    # third dense layer for prediction
    dense_layer_3 = Dense(units = n_classes, activation = output_activation)(drop_layer_2)
    
    # wrap architecture within keras Model
    model = Model(inputs = inputs, outputs = dense_layer_3, name = name)
    
    return model