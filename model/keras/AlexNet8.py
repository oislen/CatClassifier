# load in relevant libraries
from beartype import beartype
from typing import Union
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, MaxPooling2D

@beartype
def AlexNet8(
        input_shape:Union[list,tuple]=(227,227,3), 
        n_classes:int=1000,
        output_activation:str='softmax'
        ) -> Model:
    """
    AlexNet8 Keras model
    
    Parameters
    ----------
    input_shape : list,tuple
        The dimensions of the input image arrays, default is (227.227,3)
    n_classes : int
        The number of output classes to classify for, default is 1000
    output_activation : str
        The type of activation function to use, default is softmax
    
    Returns
    -------
    Model
        The keras AlexNet8 model
    """

    # set input shapes
    inputs = Input(shape = input_shape)
    
    # first convulation and pooling layer
    con_layer_1 = Conv2D(filters = 96, kernel_size = (11, 11), strides = (4, 4), activation = 'relu', padding = 'valid')(inputs)
    pool_layer_1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(con_layer_1)

    # second convulation and pooling layer
    con_layer_2 = Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', padding = 'same')(pool_layer_1)
    pool_layer_2 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(con_layer_2)

    # third convulation and pooling layer
    con_layer_3 = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(pool_layer_2)
    con_layer_3 = Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(con_layer_3)
    con_layer_3 = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(con_layer_3)
    pool_layer_3 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(con_layer_3)
    flat_layer_3 = Flatten()(pool_layer_3)

    # forst dense layer with dropout regularization
    dense_layer_1 = Dense(units = 4096, activation = 'relu')(flat_layer_3)
    drop_layer_1 = Dropout(0.25)(dense_layer_1)
    
    # second dense layer with dropout regularization
    dense_layer_2 = Dense(units = 4096, activation = 'relu')(drop_layer_1)
    drop_layer_2 = Dropout(0.25)(dense_layer_2)
    
    # third dense layer for prediction
    dense_layer_3 = Dense(units = n_classes, activation = output_activation)(drop_layer_2)
    
    # wrap architecture within keras Model
    model = Model(inputs = inputs, outputs = dense_layer_3, name = 'AlexNet8')
    
    return model
    