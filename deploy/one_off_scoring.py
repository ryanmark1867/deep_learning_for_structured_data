# stub to do one-off scoring using the trained streetcar delay model

# imports

import pandas as pd
import numpy as np
import time
# import datetime, timedelta
import datetime
from datetime import datetime, timedelta
from datetime import date
from dateutil import relativedelta
from io import StringIO
import pandas as pd
import pickle
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from io import StringIO
import requests
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import math
from subprocess import check_output
#model libraries
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pickle import load
from custom_classes import encode_categorical
from custom_classes import prep_for_keras_input
from custom_classes import fill_empty
from custom_classes import encode_text
from datetime import date
from datetime import datetime
import logging

logging.getLogger().setLevel(logging.WARNING)
logging.warning("logging check")

# define key values:

model_path = 'C:\personal\chatbot_july_2019\streetcar_1\keras_models\scmodeldec27b_5.h5'
BATCH_SIZE = 1000

'''
Output of X_test structure (used to test as part of the training process) to
use to get a single record to exercise the trained model

X_test  {'hour': array([18,  4, 11, ...,  2, 23, 17]), 'Route': array([ 0, 12,  2, ..., 10, 12,  2]), 'daym': array([21, 16, 10, ..., 12, 26,  6]),
    'month': array([0, 1, 0, ..., 6, 2, 1]), 'year': array([5, 2, 3, ..., 1, 4, 3]), 'Direction': array([1, 1, 4, ..., 2, 3, 0]),
    'day': array([1, 2, 2, ..., 0, 1, 1])}


'''


# define dictionary containing hand-crafted input for the trained model to score
# take the first value from each of the numpy arrays in X_test above
score_sample = {}
score_sample['hour'] = np.array([18])
score_sample['Route'] = np.array([0])
score_sample['daym'] = np.array([21])
score_sample['month'] = np.array([0])
score_sample['year'] = np.array([5])
score_sample['Direction'] = np.array([1])
score_sample['day'] = np.array([1])

# load model from file, print summary and make prediction for the hand-crafted score-sample

loaded_model = load_model(model_path)
loaded_model.summary()

preds = loaded_model.predict(score_sample, batch_size=BATCH_SIZE)

print("prediction is "+str(preds[0][0]))

