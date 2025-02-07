# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:31:42 2021
@author: oislen
"""

# load relevant libraries
import os
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def fit_model(model,
              X_train, 
              Y_train,
              X_val, 
              Y_val, 
              optimizer,
              batch_size = 32,
              valid_batch_size = 8,
              datagen = None, 
              lrate_red = None,
              output_dir = "checkpoints",
              class_weight = None,
              shuffle = True,
              loss = 'categorical_crossentropy',
              metric = 'accuracy',
              epochs = 60, 
              starting_epoch = None,
              verbose = True
              ):
    """
    
    Fit model.
    
    You can edit this function anyhow.
    
    """
    
    # check that the output directory exists
    if os.path.exists(output_dir) == False:
        raise OSError('Input Error: Output directory does not exist {}.'.format(output_dir))
    
    # create verbose_int
    verbose_int = int(verbose)
    
    # extract out the model name
    model_name = model.name
    
    # if printing updates
    if verbose:
        model.summary()
    
    # if reloading previous model
    if starting_epoch != None:
        check_in_filename = model_name +'{0:02d}.keras'.format(starting_epoch)
        check_in_fpath = os.path.join(output_dir, check_in_filename)
        model = load_model(check_in_fpath)
    # otherwise compile
    else:
        # compile model (can use another optimizer)
        starting_epoch = 0
        model.compile(optimizer = optimizer,
                      loss = loss,
                      metrics = [metric]
                     )
        
    # calculate steps per epoch
    if (batch_size is not None) and (valid_batch_size is not None):
        steps_per_epoch = len(X_train) // batch_size
        validation_steps = len(X_val) // valid_batch_size
    else:
        steps_per_epoch = None
        validation_steps = None
    
    # if not applying data augmentation
    if datagen == None:
        # define validation, x and y data accordingly
        validation_data = (X_val, Y_val)
        x = X_train 
        y = Y_train
    # others
    else:
        # define valudation, x and y data accordingly
        validation_data = (datagen.standardize(X_val), Y_val)
        x = datagen.flow(X_train, Y_train, batch_size = batch_size)
        y = None
        batch_size = None
    
    # if not apply learning rate reduction
    if lrate_red == None:
        # assign empty list
        lrate_red = []
    # otherwise
    else:
        # wrap learning rate reduction in a list
        lrate_red = [lrate_red]
        
    # create the output path
    check_out_fname = model_name + "{epoch:02d}.hdf5"
    check_out_fpath = os.path.join(output_dir, check_out_fname)
    
    # generate checkpoints
    check_points = [ModelCheckpoint(check_out_fpath, save_best_only = False, verbose = verbose_int)]
    
    # set model call backs
    callbacks = check_points + lrate_red
    
    # starts training
    model_fit = model.fit(x = x,
                          y = y,
                          batch_size = batch_size,
                          epochs = epochs, 
                          steps_per_epoch = steps_per_epoch,
                          callbacks = callbacks,
                          class_weight = class_weight,
                          shuffle = shuffle,
                          verbose = verbose_int,
                          validation_data = validation_data,
                          initial_epoch = starting_epoch,
                          validation_steps = validation_steps
                          )  
    
    return model_fit
