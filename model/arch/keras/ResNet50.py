#!/usr/bin/env python
# coding: utf-8

from keras.models import Model
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.initializers import random_uniform, glorot_uniform

def identity_block(X, f, filters, training=True, initializer=random_uniform):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value, will later add back to the main path. 
    shortcut = X
    
    # First component of main path
    conv_comp_1 = Conv2D(filters = F1, kernel_size = 1, strides = (1, 1), padding = 'valid', kernel_initializer = initializer(seed = 0))(X)
    norm_comp_1 = BatchNormalization(axis = 3)(conv_comp_1, training = training)
    act_comp_1 = Activation('relu')(norm_comp_1)
    
    # Second component of main path
    conv_comp_2 = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding = 'same', kernel_initializer = initializer(seed = 0))(act_comp_1)
    norm_comp_2 = BatchNormalization(axis = 3)(conv_comp_2, training = training)
    act_comp_2 = Activation('relu')(norm_comp_2)

    # Third component of main path
    conv_comp_3 = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding = 'valid', kernel_initializer = initializer(seed = 0))(act_comp_2)
    nrom_comp_3 = BatchNormalization(axis = 3)(conv_comp_3, training = training)
    short_comp_3 = Add()([shortcut, nrom_comp_3])
    act_comp_3 = Activation('relu')(short_comp_3)

    return act_comp_3


def convolutional_block(X, f, filters, s = 2, training=True, initializer = glorot_uniform):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                   also called Xavier uniform initializer.
    
    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    shortcut = X

    # First component of main path glorot_uniform(seed=0)
    conv_comp_1 = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    norm_comp_1 = BatchNormalization(axis = 3)(conv_comp_1, training=training)
    act_comp_1 = Activation('relu')(norm_comp_1)

    # Second component of main path 
    conv_comp_2 = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding = 'same', kernel_initializer = initializer(seed = 0))(act_comp_1)
    norm_comp_2 = BatchNormalization(axis = 3)(conv_comp_2, training = training)
    act_comp_2 = Activation('relu')(norm_comp_2)

    # Third component of main path
    conv_comp_3 = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer(seed = 0))(act_comp_2)
    norm_comp_3 = BatchNormalization(axis = 3)(conv_comp_3, training = training)
    shortcut_conv_comp_3 = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed = 0))(shortcut)
    shortcut_norm_comp_3 = BatchNormalization(axis = 3)(shortcut_conv_comp_3, training = training)
    shortcut_comp_3 = Add()([norm_comp_3, shortcut_norm_comp_3])
    act_comp_3 = Activation('relu')(shortcut_comp_3)
    
    return act_comp_3


def ResNet50(input_shape = (64, 64, 3), 
             n_classes = 6,
             output_activation = 'softmax'
             ):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    n_classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras

    Example:
    model = ResNet50(input_shape = (64, 64, 3), n_classes = 6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train = X_train_orig / 255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    model.fit(X_train, Y_train, epochs = 10, batch_size = 32)
    
    """
    
    # Define the input as a tensor with shape input_shape, and add zero padding
    inputs = Input(input_shape)
    input_pad = ZeroPadding2D((3, 3))(inputs)
    
    # Stage 1 - layer 1
    conv_stage_1 = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(input_pad)
    norm_stage_1= BatchNormalization(axis = 3)(conv_stage_1)
    act_stage_1 = Activation('relu')(norm_stage_1)
    pool_stage_1 = MaxPooling2D((3, 3), strides=(2, 2))(act_stage_1)

    # Stage 2 - layers 2 to 10
    convblock_stage_2 = convolutional_block(pool_stage_1, f = 3, filters = [64, 64, 256], s = 1)
    identblock_stage_2 = identity_block(convblock_stage_2, 3, [64, 64, 256])
    identblock_stage_2 = identity_block(identblock_stage_2, 3, [64, 64, 256])

    # Stage 3 - layers 13 to 22
    convblock_stage_3 = convolutional_block(identblock_stage_2, f = 3, filters = [128,128,512], s = 2)
    identblock_stage_3 = identity_block(convblock_stage_3, 3,  [128,128,512])
    identblock_stage_3 = identity_block(identblock_stage_3, 3,  [128,128,512])
    identblock_stage_3 = identity_block(identblock_stage_3, 3,  [128,128,512])
    
    # Stage 4 - layers 23 to 40
    convblock_stage_4 = convolutional_block(identblock_stage_3, f = 3, filters = [256, 256, 1024], s = 2)
    identblock_stage_4 = identity_block(convblock_stage_4, 3, [256, 256, 1024])
    identblock_stage_4 = identity_block(identblock_stage_4, 3, [256, 256, 1024])
    identblock_stage_4 = identity_block(identblock_stage_4, 3, [256, 256, 1024])
    identblock_stage_4 = identity_block(identblock_stage_4, 3, [256, 256, 1024])
    identblock_stage_4 = identity_block(identblock_stage_4, 3, [256, 256, 1024])

    # Stage 5 - layers 41 to 49
    convblock_stage_5 = convolutional_block(identblock_stage_4, f = 3, filters = [512, 512, 2048], s = 2)
    identblock_stage_5 = identity_block(convblock_stage_5, 3, [512, 512, 2048])
    identblock_stage_5 = identity_block(identblock_stage_5, 3, [512, 512, 2048])
    identblock_stage_5 = AveragePooling2D((2, 2))(identblock_stage_5)
    
    # Output - layer 50
    flat_outputs = Flatten()(identblock_stage_5)
    outputs = Dense(units = n_classes, activation = output_activation, kernel_initializer = glorot_uniform(seed = 0))(flat_outputs)
    
    # Create model
    model = Model(inputs = inputs, outputs = outputs, name = 'ResNet50')

    return model
