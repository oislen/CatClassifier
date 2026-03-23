from beartype import beartype
from typing import Union
from keras.layers import Dropout, Dense, GlobalMaxPooling2D
from keras.applications import VGG16
from keras.models import Model

@beartype
def VGG16_pretrained(
    input_shape:Union[list,tuple]=(224,224,3), 
    n_classes:int=1,
    output_activation:str='sigmoid'
    ) -> Model:
    """
    Pretrained VGG16 keras model
    
    Parameters
    ----------
    input_shape : list,tuple
        the input image shape / dimensions, default is (224,224,3)
    n_classes : int
        The number of target classes, default is 1
    output_activation : str
        The type of activation function to use, default is sigmoid
    
    Returns
    -------
    Model
        Keras.Model, the pretrained VGG16 model
    
    Example
    -------
    model = VGG16_pretrained(input_shape = (224, 224, 3), 
                             n_classes = 1,
                             output_activation = 'sigmoid',
                             name = 'VGG16_pretrained'
                             )

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy']
                  )

    model.summary()
    """

    VGG16_pretrained = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
        
    for layer in VGG16_pretrained.layers[:15]:
        layer.trainable = False

    for layer in VGG16_pretrained.layers[15:]:
        layer.trainable = True
        
    last_layer = VGG16_pretrained.get_layer('block5_pool')
    last_output = last_layer.output
        
    # Flatten the output layer to 1 dimension
    x = GlobalMaxPooling2D()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(512, activation='relu')(x)
    # Add a dropout rate of 0.5
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer for classification
    x = Dense(n_classes, activation=output_activation)(x)

    model = Model(inputs = VGG16_pretrained.input, outputs = x, name = 'VGG16_pretrained')

    return model
