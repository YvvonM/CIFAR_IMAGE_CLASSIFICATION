# -*- coding: utf-8 -*-
"""CIFAR10 _CLASSIFICATION_USING_TENSORFLOW.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rgqf9RmxqUvqSa9QNISMpaFpKYJ5ODFS

### Importing libraries
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import pandas as pd

"""### Importing the datasets"""

cifar10 = keras.datasets.cifar10

"""- The cifar10 dataset is the one that will be used in the model"""

#loading the train and test data
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

"""- The train and test data has been loaded"""

# getting the shape of the train data and test data
print('*'*30)
print(train_data.shape)
print('*'*30)
print(test_data.shape)
print('*'*30)

"""- The train dataset contains 50000 images while the test data contains 10000 images
- The images are RBG images
"""

# chaning the images to range 0 to 1
train_img = train_data/255.0
test_img = test_data/255.0

"""- Here, the image data is normalized.
- It is changed to decimal numbers from 0 to 1
"""

#getting the classnames of cifar10
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print(class_names)

"""- The cifar10 dataset has 10 classes. The above are the names of the classes

### Model building
"""

# Create a Sequential model
model = keras.Sequential()

# Add Convolutional layers with MaxPooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output and add Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Print the model summary
model.summary()

"""- The conv net has 184138 parameters to train.
- It has 0 non-trainable parameters
- Here same padding was used to ensure the same image size after convolution

---

### Training the model
"""

# setting the loss, optimizer, batchsize and epochs
batch_size = 64
epochs = 50
loss = 'sparse_categorical_crossentropy'
optimizera = keras.optimizers.Adam(learning_rate = 0.001)
metrics = ['accuracy']

"""- Here we have set the loss we will use wich will be *`CategoricalCrossentropy`*, the number of epochs which will be 50, the batch size which is 64 and the optimizer which is *`Adam`* optimizer
- The metrics used here is accuracy
"""

#setting the earlystopping of the model
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,          # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore the best model weights when stopping
)

"""- This will stop the training of the model when the validation loss has no improvemets after 5 epochs"""

#defining the model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',  # Monitor validation accuracy
    save_best_only=True,  # Save only the best model
    mode='max'
)

"""- Here, the best model is saved.
- The validation accuracy is used to monitor the model
"""

#compiling the model
model.compile(loss = loss, optimizer = optimizera, metrics = metrics)

"""- Here, we compile the model"""

#training the model
history = model.fit(train_img, train_labels, epochs = epochs, validation_data=(test_img, test_labels), callbacks=[early_stopping, checkpoint], verbose = 1)

"""- The model is fit on the training data and used the evaluation data to evaluate it.
- The model has used 13 epochs.
"""

best_model = keras.models.load_model('best_model.h5')
print(best_model)

"""- The best model has been saved as a h5 file.

### fitting the model
"""

hist1 = best_model.fit(train_img, train_labels, epochs = epochs, validation_data=(test_img, test_labels), callbacks=[early_stopping, checkpoint], verbose = 1)

"""- The best model has a validation accuracy of 74 percent"""

#checking the history of the model
hist1.history

#changing it to a dataframe
hist_df = pd.DataFrame(hist1.history)
print(hist_df)