from beartype import beartype
from types import UnionType
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Conv2D, MaxPooling2D

@beartype
def VGG16(input_shape:UnionType[list,tuple]=(224,224,3), 
         n_classes:int=1000,
         output_activation:str='softmax'
         ) -> Model:
    """
    VGG16 keras model
    
    Parameters
    ----------
    input_shape : list,tuple
        the input image shape / dimensions, default is (224,224,3)
    n_classes : int
        The number of target classes, default is 1000
    output_activation : str
        The type of activation function to use, default is softmax
    
    Returns
    -------
    Model
        Keras.Model, the VGG16 model
    """

    # set input shapes
    inputs = Input(shape = input_shape)
    
    # first convulation and pooling layer
    con_layer_1 = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(inputs)
    con_layer_1 = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(inputs)
    pool_layer_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(con_layer_1)

    # second convulation and pooling layer
    con_layer_2 = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(pool_layer_1)
    con_layer_2 = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(con_layer_2)
    pool_layer_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(con_layer_2)

    # third convulation and pooling layer
    con_layer_3 = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(pool_layer_2)
    con_layer_3 = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(con_layer_3)
    con_layer_3 = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(con_layer_3)
    pool_layer_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(con_layer_3)

    # fourth convulation and pooling layer
    con_layer_4 = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(pool_layer_3)
    con_layer_4 = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(con_layer_4)
    con_layer_4 = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(con_layer_4)
    pool_layer_4 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(con_layer_4)

    # fifth convulation and pooling layer
    con_layer_5 = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(pool_layer_4)
    con_layer_5 = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(con_layer_5)
    con_layer_5 = Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(con_layer_5)
    pool_layer_5 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(con_layer_5)
    flat_layer_3 = Flatten()(pool_layer_5)

    # first dense layer with dropout regularization
    dense_layer_1 = Dense(units = 4096, activation = 'relu')(flat_layer_3)
    drop_layer_1 = Dropout(0.25)(dense_layer_1)
    
    # second dense layer with dropout regularization
    dense_layer_2 = Dense(units = 4096, activation = 'relu')(drop_layer_1)
    drop_layer_2 = Dropout(0.25)(dense_layer_2)
    
    # third dense layer for prediction
    dense_layer_3 = Dense(units = n_classes, activation = output_activation)(drop_layer_2)
    
    # wrap architecture within keras Model
    model = Model(inputs = inputs, outputs = dense_layer_3, name = 'VGG16')
    
    return model
    