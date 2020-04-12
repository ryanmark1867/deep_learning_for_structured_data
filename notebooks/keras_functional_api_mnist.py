# minimal Kera functional API model for MNIST
# adapted from https://www.tensorflow.org/guide/keras/functional

#import libraries

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# define model layers

inputs = keras.Input(shape=(784,)) 
flatten = layers.Flatten(input_shape=(28, 28))
flattened = flatten(inputs)
dense = layers.Dense(128, activation='relu')(flattened)
dropout = layers.Dropout(0.2) (dense)
outputs = layers.Dense(10) (dropout) 

# define model inputs and outputs (taken from layer definition)

model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255 
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# compile model, including specifying the loss function, optimizer, and metrics

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy']) 

# train model

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2) 

# assess model performance

test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
