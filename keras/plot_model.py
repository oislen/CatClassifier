# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:37:53 2021
@author: oislen
"""

# load relevant libraries
import os
from matplotlib import pyplot as plt
 
def plot_model_fit(model_fit, output_fdir = None):
    
    """
    
    Plot Model Fit Documentation
    
    Function Overview
    
    This function plots the model's fit during training in relation to a validation set.
    
    Defaults
    
    plot_model_fit(model_fit)
    
    Parameters
    
    model_fit - model.predict(), the Keras model predict object
    
    Returns
    
    0 for successful execution
    
    Exmaple
    
    plot_model_fit(model_fit = model_fit)
    
    Source
    
    https://github.com/jiadaizhao/Advanced-Machine-Learning-Specialization
    
    """
    
    #-- Accuracy Plot --#
    
    # plot training and validation accuracy
    plt.plot(model_fit.history['accuracy'])
    plt.plot(model_fit.history['val_accuracy'])
    
    # assign plot title
    plt.title('model accuracy')
    
    # assign x and y labels
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    
    # assign plot legend
    plt.legend(['train', 'val'], loc='upper left')
    
    #  show save, plot and close
    if output_fdir != None:
        accuracy_output_fpath = os.path.join(output_fdir, 'model_accuracy.png')
        plt.savefig(accuracy_output_fpath)
    plt.show()
    plt.close()
    
    #-- Loss Plot --#
    
    # plot training validation loss
    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    
    # assign plot title
    plt.title('model loss')
    
    # assign x and y labels
    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    # assign plot legend
    plt.legend(['train', 'val'], loc='upper left')
    
    #  show save, plot and close
    if output_fdir != None:
        loss_output_fpath = os.path.join(output_fdir, 'model_loss.png')
        plt.savefig(loss_output_fpath)
    plt.show()
    plt.close()
    


    return 0