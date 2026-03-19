from beartype import beartype
from typing import Union
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, AveragePooling2D

@beartype
def LeNet5(
    input_shape:Union[list,tuple]=(28,28,1), 
    n_classes:int=10,
    output_activation:str='softmax'
    ) -> Model:
    """
    LeNet5 keras model
    
    Parameters
    ----------
    input_shape : list,tuple
        the input image shape / dimensions, default is (28,28,1)
    n_classes : int
        The number of target classes, default is 10
    output_activation : str
        The type of activation function to use, default is softmax
    
    Returns
    -------
    Model
        Keras.Model, the LeNet model
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
    model = Model(inputs = inputs, outputs = dense_layer_3, name = 'LeNet5')
    
    return model