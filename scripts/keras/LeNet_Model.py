# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:22:57 2021
@author: oislen
"""

# load in relevant libraries
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, MaxPooling2D

def LeNet_Model(image_shape, 
                n_targets
                ):
    
    """
        
    LeNet Model Documentation
    
    Function Overview
    
    This function generates a LeNet Model architecture:
        
         1. Conv2D 
             - filters: 32  
             - kernal: 5 x 5 
             - activation: relu 
             - padding: same
             
         2. MaxPooling2D 
             - pool: 2 x 2 
             
         3. Conv2D 
             - filters: 32  
             - kernal: 5 x 5 
             - activation: relu 
             - padding: same
             
         4. MaxPooling2D 
             - pool: 2 x 2 
             
         5. Flatten
         
         6. Dense 
             - units: 128  
             - activation: relu 
             
         7. Dropout 
             - rate: 0.25 
             
         8. Dense 
             - units: 64  
             - activation: relu 
             
         9. Dropout 
             - 0.25 rate
             
        10. Dense 
            - units: n target  
            - activation:softmax 
        
    Defaults
    
    LeNet_Model(image_shape, 
                n_targets
                )
    
    Parameters
    
    image_shape - the input image shape / dimensions
    n_targets - the number of target classes
    
    Returns
    
    model - keras.Model, the LeNet model
    
    Example
    
    LeNet_Model(image_shape = (28, 28, 1), 
                n_targets = 2
                )
    
    
    """
    
    # set function constants
    act_func = 'relu'
    kernel_size = (5, 5)
    pad = 'same'
    pool_size = (2, 2)
    drop_rate = 0.25
    
    # Classification model
    # You can start from LeNet architecture
    inputs = Input(shape = image_shape)
    
    # first convulation layer
    conv_layer_1 = Conv2D(filters = 32, 
                          kernel_size = kernel_size, 
                          activation = act_func, 
                          padding = pad
                          )(inputs)
    
    # pooling layer
    pool_layer_1 = MaxPooling2D(pool_size)(conv_layer_1)
    
    # second convulation later
    conv_layer_2 = Conv2D(filters = 32, 
                          kernel_size = kernel_size, 
                          activation = act_func,
                          padding = pad
                          )(pool_layer_1)
    
    # pooling layer
    pool_layer_2 = MaxPooling2D(pool_size)(conv_layer_2)
    
    # flatten inputs
    flat_layer = Flatten()(pool_layer_2)
    
    # first dense layer
    dense_layer_1 = Dense(units = 128, 
                          activation = act_func
                          )(flat_layer)
    
    # drop out layer
    drop_layer_1= Dropout(drop_rate)(dense_layer_1)
    
    # second dense layer
    dense_layer_2 = Dense(units = 64, 
                          activation = act_func
                     )(drop_layer_1)
    
    # drop out layer
    drop_layer_2 = Dropout(drop_rate)(dense_layer_2)

    # set prediction layer
    dense_layer_3 = Dense(units = n_targets, 
                          activation = 'softmax'
                          )(drop_layer_2)
    
    # create level
    model = Model(inputs = inputs, 
                  outputs = dense_layer_3,
                  name = "LeNet"
                 )
    
    return model