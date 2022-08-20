import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

import cons
from utilities.plot_image import plot_image
from LeNet_Model import LeNet_Model
from fit_model import fit_model
from plot_model import plot_model_fit

# set data file directory
data_fdir = cons.data_fdir

# load in processed data
data = pd.read_pickle(os.path.join(data_fdir, 'model_data.pickle'))

# randomly shuffle data
data = data.sample(frac = 1.0, replace = False)

# extract out X dataframe and y series
y_train = data['target'].values
X_train = np.stack(data['pad_image_array'].values)

# standardise values between 0 and 1
X_train = X_train / 255

# split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1)

# initiate lenet model
lenet_model = LeNet_Model(image_shape = X_train[0].shape, n_targets = 1)

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
model_fit = fit_model(model = lenet_model, 
                     epochs = 20,
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
                     Y_val = y_valid
                     )

# plot model fits
plot_model_fit(model_fit = model_fit)