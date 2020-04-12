# minimal Kera sequential API model for MNIST
# adapted from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb

#import libraries

import tensorflow as tf
import pydotplus
from tensorflow.keras.utils import plot_model

mnist = tf.keras.datasets.mnist

# define model inputs

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# define model layers

model = tf.keras.models.Sequential([ 
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  tf.keras.layers.Dense(128, activation='relu'), 
  tf.keras.layers.Dropout(0.2), 
  tf.keras.layers.Dense(10) 
])

# compile model, including specifying the loss function, optimizer, and metrics

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) 

# train model

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)

# assess model performance
                    
test_scores = model.evaluate(x_test,  y_test, verbose=2) 
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
