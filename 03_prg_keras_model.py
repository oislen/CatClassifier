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
from keras.utils import to_categorical

# load custom scripts
import cons
from arch.LeNet5 import LeNet5
from plot_model import plot_model_fit
from utilities.plot_image import plot_image

# create a dataframe of filenames and categories
filenames = os.listdir(cons.train_fdir)
categories = [1 if filename.split('.')[0] == 'dog' else 0 for filename in filenames]
df = pd.DataFrame({'filename': filenames, 'category': categories})

# count plot og categories
df['category'].value_counts()

# random image plot
sample = random.choice(filenames)
image = load_img(os.path.join(cons.train_fdir, sample))
plot_image(image)

# initiate LeNet5 architecture
keras_model = LeNet5(input_shape = cons.input_shape, n_classes = 2, output_activation = 'softmax', name = 'LeNet5')

# set gradient decent compiler
keras_model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
keras_model.summary()

# set early stopping
earlystop = EarlyStopping(patience=10)
# set learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
# set model check points
model_name = keras_model.name
check_out_fname = model_name + "{epoch:02d}.hdf5"
check_out_fpath = os.path.join(cons.checkpoints_fdir, check_out_fname)
check_points = ModelCheckpoint(check_out_fpath, save_best_only = False, verbose = 1)
# combine model callbacks
callbacks = [earlystop, learning_rate_reduction, check_points]

# prepare data
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
# plot category split of training data
train_df['category'].value_counts()
# plot category split of validation data
validate_df['category'].value_counts()

# set data constants
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

# set train datagen
train_datagen = ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
train_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = cons.train_fdir, x_col='filename',y_col='category', target_size=cons.IMAGE_SIZE, class_mode='categorical', batch_size=cons.batch_size)

# set validation datagen
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(dataframe = validate_df, directory = cons.train_fdir, x_col='filename', y_col='category', target_size=cons.IMAGE_SIZE, class_mode='categorical', batch_size=cons.batch_size)

# datagen example
#example_df = train_df.sample(n=1).reset_index(drop=True)
#example_generator = train_datagen.flow_from_dataframe(dataframe = example_df, directory = cons.train_fdir, x_col='filename', y_col='category', target_size=cons.IMAGE_SIZE, class_mode='categorical')

# plot example
#plt.figure(figsize=(12, 12))
#for i in range(0, 15):
#    plt.subplot(5, 3, i+1)
#    for X_batch, Y_batch in example_generator:
#        image = X_batch[0]
#        plt.imshow(image)
#        break
#plt.tight_layout()
#plt.show()

# fit model
epochs=3 if cons.FAST_RUN else 50
model_fit = keras_model.fit(train_generator, epochs=epochs, validation_data=validation_generator, validation_steps=total_validate//cons.batch_size, steps_per_epoch=total_train//cons.batch_size, callbacks=callbacks)

# save trained keras model
with open(cons.model_fit_pickle_fpath, 'wb') as handle:
    pickle.dump(model_fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save model fit
keras_model.save(cons.keras_model_pickle_fpath, save_format = "h5")

# plot model fits
plot_model_fit(model_fit = model_fit, output_fdir = cons.report_fdir)

# prepare test data
test_filenames = os.listdir(cons.test_fdir)
test_df = pd.DataFrame({'filename': test_filenames})
nb_samples = test_df.shape[0]

# set validation datagen
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(dataframe = test_df, directory = cons.test_fdir, x_col='filename', y_col=None, class_mode=None, target_size=cons.IMAGE_SIZE, batch_size=cons.batch_size, shuffle=False)

# make test set predictions
predict = keras_model.predict(test_generator, steps=np.ceil(nb_samples/cons.batch_size))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

# plot prediction results
test_df['category'].value_counts()

# plot random sample predictions
#sample_test = test_df.head(18)
#sample_test.head()
#plt.figure(figsize=(12, 24))
#for index, row in sample_test.iterrows():
#    filename = row['filename']
#    category = row['category']
#    img = load_img(os.path.join(cons.test_fdir, filename), target_size=cons.IMAGE_SIZE)
#    plt.subplot(6, 3, index+1)
#    plt.imshow(img)
#    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
#plt.tight_layout()
#plt.show()

# make submission
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
#submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv(os.path.join(cons.report_fdir, 'submission.csv'), index=False)