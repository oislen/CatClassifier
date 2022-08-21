# load relevant libraries
import os
import sys
import numpy as np
import pandas as pd
import pickle
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau


# load custom modules
sys.path.append(os.getcwd())
import cons
from utilities.plot_image import plot_image
from arch.LeNet5 import LeNet5
from fit_model import fit_model
from plot_model import plot_model_fit

print('Loading data ...')

# set data file directory
data_fdir = cons.data_fdir

# load in processed data
data = pd.read_pickle(os.path.join(data_fdir, 'model_data.pickle'))

# randomly shuffle training data
train = train.sample(frac = 1.0, replace = False)

print('Training Keras model ...')

print('Training Keras model ...')

# initiate lenet model
keras_model = LeNet5(input_shape = X_train[0].shape, 
                     n_classes = 1, 
                     output_activation = 'sigmoid'
                     )

# intiate rms prop optimiser
optimizer = RMSprop(learning_rate = 0.001, 
                    rho = 0.9, 
                    epsilon = 1e-08, 
                    decay = 0.0
                    )

# set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor = 'accuracy', 
                                            patience = 3, 
                                            verbose = 1, 
                                            factor = 0.5, 
                                            min_lr = 0.00001
                                            )

# Attention: Windows implementation may cause an error here. In that case use model_name=None.
model_fit = fit_model(model = keras_model, 
                     epochs = 30,
                     loss = 'binary_crossentropy',
                     starting_epoch = None,
                     batch_size = None,
                     valid_batch_size = None,
                     optimizer = optimizer,
                     lrate_red = learning_rate_reduction,
                     datagen = None, 
                     X_train = X_train,
                     X_val = X_valid, 
                     Y_train = y_train, 
                     Y_val = y_valid,
                     output_dir = cons.checkpoints_fdir
                     )

# save trained keras model
with open(cons.model_fit_pickle_fpath, 'wb') as handle:
    pickle.dump(model_fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save model fit
keras_model.save(cons.keras_model_pickle_fpath, save_format = "h5")

# plot model fits
plot_model_fit(model_fit = model_fit, output_fdir = cons.report_fdir)

if cons.test_sample_size > 0:
    # making test set predictions
    print('Making test predictions ...')
    # make model predictions
    test.loc[:, 'preds'] = pd.Series(keras_model.predict(X_test).reshape(cons.test_sample_size,), index = test.index).round()
    # save predictions to disk
    test.to_pickel(cons.test_preds_pickle_fpath)