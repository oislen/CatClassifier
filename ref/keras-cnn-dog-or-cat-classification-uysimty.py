# https://www.kaggle.com/code/uysimty/keras-cnn-dog-or-cat-classification

import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# load custom scripts
import cons
from LeNet5 import LeNet5

FAST_RUN = True
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

# create a dataframe of filenames and categories
filenames = os.listdir(cons.train_fdir)
categories = [1 if filename.split('.')[0] == 'dog' else 0 for filename in filenames]
df = pd.DataFrame({'filename': filenames, 'category': categories})
df.head()
df.tail()

# count plot og categories
df['category'].value_counts().plot.bar()
plt.show()

# random image plot
sample = random.choice(filenames)
image = load_img(os.path.join(cons.train_fdir, sample))
plt.imshow(image)
plt.show()

# initiate LeNet5 architecture
lenet5_arch = LeNet5(input_shape = input_shape, n_classes = 2, output_activation = 'softmax', name = 'LeNet5')

# set gradient decent compiler
lenet5_arch.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
lenet5_arch.summary()

# set early stopping
earlystop = EarlyStopping(patience=10)
# set learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
# combine model callbacks
callbacks = [earlystop, learning_rate_reduction]

# prepare data
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
# plot category split of training data
train_df['category'].value_counts().plot.bar()
plt.show()
# plot category split of validation data
validate_df['category'].value_counts().plot.bar()
plt.show()

# set data constants
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15

# set train datagen
train_datagen = ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
train_generator = train_datagen.flow_from_dataframe(dataframe = train_df, directory = cons.train_fdir, x_col='filename',y_col='category', target_size=IMAGE_SIZE, class_mode='categorical', batch_size=batch_size)

# set validation datagen
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(dataframe = validate_df, directory = cons.train_fdir, x_col='filename', y_col='category', target_size=IMAGE_SIZE, class_mode='categorical', batch_size=batch_size)

# datagen example
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(dataframe = example_df, directory = cons.train_fdir, x_col='filename', y_col='category', target_size=IMAGE_SIZE, class_mode='categorical')

# plot example
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

# fit model
epochs=3 if FAST_RUN else 50
history = lenet5_arch.fit_generator(train_generator, epochs=epochs, validation_data=validation_generator, validation_steps=total_validate//batch_size, steps_per_epoch=total_train//batch_size, callbacks=callbacks)

# save trained model
#lenet5_arch.save_weights("model.h5")

# plot model training
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))
ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))
legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

# prepare test data
test_filenames = os.listdir(cons.test_fdir)
test_df = pd.DataFrame({'filename': test_filenames})
nb_samples = test_df.shape[0]

# set validation datagen
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(dataframe = test_df, directory = cons.test_fdir, x_col='filename', y_col=None, class_mode=None, target_size=IMAGE_SIZE, batch_size=batch_size, shuffle=False)

# make test set predictions
predict = lenet5_arch.predict(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

# plot prediction results
test_df['category'].value_counts().plot.bar()
plt.show()

# plot random sample predictions
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(os.path.join(cons.test_fdir, filename), target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()

# make submission
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
#submission_df.to_csv('submission.csv', index=False)