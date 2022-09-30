import os
import pickle
import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model

# load custom scripts
import cons
from utilities.plot_image import plot_image
from utilities.plot_generator import plot_generator
from utilities.plot_preds import plot_preds
from model.arch.AlexNet8 import AlexNet8
from model.plot_model import plot_model_fit

# create a dataframe of filenames and categories
filenames = os.listdir(cons.train_fdir)
categories = [1 if filename.split('.')[0] == 'dog' else 0 for filename in filenames]
df = pd.DataFrame({'filename': filenames, 'category': categories})
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
df['source'] = df['filename'].str.contains(pat = '[cat|dog].[0-9]+\.jpg', regex = True).map({True:'kaggle', False:'webscraper'})

# random image plot
sample = random.choice(filenames)
image = load_img(os.path.join(cons.train_fdir, sample))
plot_image(image, output_fpath = cons.random_image_fpath)

# prepare data
validate_df = df[df['source'] == 'kaggle'].sample(n = 5000, random_state = 42)
train_df = df[~df.index.isin(validate_df.index)]
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# set data constants
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

# set train datagen
train_datagen = ImageDataGenerator(rotation_range=cons.rotation_range, rescale=cons.rescale, shear_range=cons.shear_range, zoom_range=cons.zoom_range, horizontal_flip=cons.horizontal_flip, width_shift_range=cons.width_shift_range, height_shift_range=cons.height_shift_range)
train_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = cons.train_fdir, x_col='filename',y_col='category', target_size=cons.IMAGE_SIZE, class_mode='categorical', batch_size=cons.batch_size)

# set validation datagen
validation_datagen = ImageDataGenerator(rescale=cons.rescale)
validation_generator = validation_datagen.flow_from_dataframe(dataframe = validate_df, directory = cons.train_fdir, x_col='filename', y_col='category', target_size=cons.IMAGE_SIZE, class_mode='categorical', batch_size=cons.batch_size)

# datagen example
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(dataframe = example_df, directory = cons.train_fdir, x_col='filename', y_col='category', target_size=cons.IMAGE_SIZE, class_mode='categorical')
plot_generator(generator = example_generator, output_fpath = cons.generator_plot_fpath)

# initiate LeNet5 architecture
keras_model = AlexNet8(input_shape = cons.input_shape, n_classes = 2, output_activation = 'softmax')
keras_model.summary()

# set gradient decent compiler
keras_model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

# set early stopping
earlystop = EarlyStopping(patience=10)

# set learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

# set model check points
model_name = keras_model.name
check_out_fname = model_name + "-{epoch:02d}.hdf5"
check_out_fpath = os.path.join(cons.checkpoints_fdir, check_out_fname)
check_points = ModelCheckpoint(check_out_fpath, save_best_only = False, verbose = 1)

# combine model callbacks
callbacks = [earlystop, learning_rate_reduction, check_points]

# fit model
epochs = 3 if cons.FAST_RUN else 25
model_fit = keras_model.fit(train_generator, epochs=epochs, validation_data=validation_generator, validation_steps=total_validate//cons.batch_size, steps_per_epoch=total_train//cons.batch_size, callbacks=callbacks)

# save trained keras model
with open(cons.model_fit_pickle_fpath, 'wb') as handle:
    pickle.dump(model_fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plot model fits
plot_model_fit(model_fit = model_fit, output_fdir = cons.report_fdir)

# save model fit
keras_model.save(cons.keras_model_pickle_fpath, save_format = "h5")
keras_model = load_model(cons.keras_model_pickle_fpath)

# prepare test data
test_filenames = os.listdir(cons.test_fdir)
test_df = pd.DataFrame({'filename': test_filenames})
nb_samples = test_df.shape[0]

# set validation datagen
test_gen = ImageDataGenerator(rescale=cons.rescale)
test_generator = test_gen.flow_from_dataframe(dataframe = test_df, directory = cons.test_fdir, x_col='filename', y_col=None, class_mode=None, target_size=cons.IMAGE_SIZE, batch_size=cons.batch_size, shuffle=cons.shuffle)

# make test set predictions
predict = keras_model.predict(test_generator, steps=np.ceil(nb_samples/cons.batch_size))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

# plot random sample predictions
plot_preds(data = test_df, cons = cons, output_fpath = cons.pred_images_fpath)

# make submission
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category'].replace({ 'dog': 1, 'cat': 0 })
submission_df.to_csv(cons.submission_csv_fpath, index=False)