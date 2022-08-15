import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop

import cons
from utilities.plot_image import plot_image
from LeNet_Model import LeNet_Model
from fit_model import fit_model


data_fdir = cons.data_fdir

# load in processed data
data = pd.read_pickle(os.path.join(data_fdir, 'model_data.pickle'))

# randomly shuffle data
data = data.sample(frac = 1.0, replace = False)

data.head()

y_train = data['target']
X_train = data['pad_image_array']

X_train = X_train / 255

# reshape image into 3 dimensions: height = 28px, width = 28px, canal = 1 (greyscale)
X_train[0][:, : np.newaxis].shape
X_train = X_train.values.reshape(512, 256, 1)

a = X_train[0]
a.shape
a_new = a[..., np.newaxis]
a_new[:, :, 0]
a[..., np.newaxis].shape
a[:, : 1] = [1]
X_train[0][:, : 1].shape

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1)

X_train.head()
y_train.head()

plot_image(X_train.loc[0, 'pad_image_array'])

lenet_model = LeNet_Model(image_shape = X_train[0].shape, n_targets = 2)

optimizer = RMSprop(learning_rate = 0.001, 
                    rho = 0.9, 
                    epsilon = 1e-08, 
                    decay = 0.0
                    )