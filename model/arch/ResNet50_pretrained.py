from keras.layers import Dropout, Dense, Flatten
from keras.applications import ResNet50
from keras.models import Model

def ResNet50_pretrained(input_shape = (224, 224, 3), 
                        n_classes = 2,
                        output_activation = 'softmax'
                        ):

    """
    model = ResNet50_pretrained(input_shape = (224, 224, 3), 
                                n_classes = 2,
                                output_activation = 'softmax',
                                name = 'ResNet50_pretrained'
                                )

    model.compile(loss='categorical_crossentropy',
                  optimizer = optimizers.Adam(1e-5),
                  metrics=['accuracy']
                  )

    model.summary()
    """

    #loading resent 
    ResNet50_pretrained = ResNet50(input_shape = input_shape, include_top = False, weights = 'imagenet')

    # Freeze layers
    ResNet50_pretrained.trainable = False 

    last_output = ResNet50_pretrained.output

    # Flatten the output layer to 1 dimension
    x = Flatten()(last_output)
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.4
    x = Dropout(0.4)(x)
    # Add a final sigmoid layer for classification
    x = Dense(n_classes, activation=output_activation)(x)

    model = Model(inputs = ResNet50_pretrained.input, outputs = x, name = 'ResNet50_pretrained')

    return model



