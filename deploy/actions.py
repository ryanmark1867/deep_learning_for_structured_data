# This files contains the custom actions related to the Rasa model for the streetcar delay prediction project

# common imports
from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet



import zipfile
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
# DSX code to import uploaded documents
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
from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
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
import yaml

# load config gile
current_path = os.getcwd()
print("current directory is: "+current_path)

path_to_yaml = os.path.join(current_path, 'deploy_config.yml')
print("path_to_yaml "+path_to_yaml)
try:
    with open (path_to_yaml, 'r') as c_file:
        config = yaml.safe_load(c_file)
except Exception as e:
    print('Error reading the config file')


# paths for model and pipeline files
pipeline1_filename = config['file_names']['pipeline1_filename']
pipeline2_filename =  config['file_names']['pipeline2_filename']
model_filename =  config['file_names']['model_filename']

# other parms
debug_on = config['general']['debug_on']
logging_level = config['general']['logging_level']
BATCH_SIZE = config['general']['BATCH_SIZE']

# set logging level
logging_level_set = logging.WARNING
if logging_level == 'WARNING':
    logging_level_set = logging.WARNING
if logging_level == 'ERROR':
    logging_level_set = logging.ERROR
if logging_level == 'DEBUG':
    logging_level_set = logging.DEBUG
if logging_level == 'INFO':
    logging_level_set = logging.INFO   
logging.getLogger().setLevel(logging_level_set)
logging.warning("logging check - beginning of logging")

def get_path(subpath):
    rawpath = os.getcwd()
    # data is in a directory called "data" that is a sibling to the directory containing the notebook
    path = os.path.abspath(os.path.join(rawpath, '..', subpath))
    return(path)

# get complete paths for pipelines and Keras models
pipeline_path = get_path('pipelines')

pipeline1_path = os.path.join(pipeline_path,pipeline1_filename)
pipeline2_path = os.path.join(pipeline_path,pipeline2_filename)
model_path = os.path.join(get_path('models'),model_filename)


# load the Keras model
loaded_model = load_model(model_path)
loaded_model.summary()
# moved pipeline definitions to custom action class
#pipeline1 = load(open(pipeline1_path, 'rb'))
#pipeline2 = load(open(pipeline2_path, 'rb'))


# brute force a scoring sample, bagged from test set
score_sample = {}
score_sample['hour'] = np.array([18])
score_sample['Route'] = np.array([0])
score_sample['daym'] = np.array([21])
score_sample['month'] = np.array([0])
score_sample['year'] = np.array([5])
score_sample['Direction'] = np.array([1])
score_sample['day'] = np.array([1])

 #  'day': array([1, 2, 2, ..., 0, 1, 1])}


# dictionary of default values
score_default = {}
score_default['hour'] = 12
score_default['Route'] = '501'
score_default['daym'] = 1
score_default['month'] = 1
score_default['year'] = '2019'
score_default['Direction'] = 'e'
score_default['day'] = 2

score_cols = ['Route','Direction','hour','year','month','daym','day']
logging.warning("score_cols after define is "+str(score_cols))



preds = loaded_model.predict(score_sample, batch_size=BATCH_SIZE)
logging.warning("pred is "+str(preds))

logging.warning("preds[0] is "+str(preds[0]))
logging.warning("preds[0][0] is "+str(preds[0][0]))

# example using pipeline on prepped data
# routedirection_frame = pd.read_csv(path+"routedirection.csv")


class ActionPredictDelay(Action):

    def name(self) -> Text:
        return "action_predict_delay"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("in action_predict_delay")
        if preds[0][0] >= 0.5:
            predict_string = "yes"
        else:
            predict_string = "no"
        dispatcher.utter_message("Delay prediction is:"+predict_string)

        return []


class ActionPredictDelayComplete(Action):
    ''' predict delay when the user has provided sufficient content to make a prediction'''
    def name(self) -> Text:
        return "action_predict_delay_complete"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.warning("in action_predict_delay_complete")
        # TODO brute force way to ensure pipeline doesn't get clogged
        pipeline1 = load(open(pipeline1_path, 'rb'))
        pipeline2 = load(open(pipeline2_path, 'rb'))
        # init dictionary to hold scoring values
        score_current = {}
        now = datetime.now()
        score_default['daym'] = now.day
        score_default['month'] = now.month
        score_default['year'] = now.year
        # TODO - remove once the model is trained with post 2019 data
        if int(score_default['year']) > 2019:
            score_default['year'] = 2019        
        score_default['hour'] = now.hour
        score_default['day'] = now.weekday()
        logging.warning("score_default is "+str(score_default))
        logging.warning("score_cols is "+str(score_cols))
        score_df = pd.DataFrame(columns=score_cols)
        
        logging.warning("score_df before load is "+str(score_df))
        # load the score_df
        for col in score_cols:
            if tracker.get_slot(col) != None:
                # if the slot has been set in Rasa, use that value
                if tracker.get_slot(col) == "today":
                    logging.warning("GOT a TOday")
                    score_df.at[0,col] = score_default[col]
                score_df.at[0,col] = tracker.get_slot(col)
            else:
                # if the slot has not been set in Rasa, use the default value
                score_df.at[0,col] = score_default[col]
        logging.warning("score_df after load is "+str(score_df))
        if debug_on:
            dispatcher.utter_message("input is: "+str(score_df))
        prepped_xform1 = pipeline1.transform(score_df)
        prepped_xform2 = pipeline2.transform(prepped_xform1)
        print("prepped_xform2 is ",prepped_xform2)
        pred = loaded_model.predict(prepped_xform2, batch_size=BATCH_SIZE)
        logging.warning("pred is "+str(pred))
        if pred[0][0] >= 0.5:
            predict_string = "yes, delay predicted"
        else:
            predict_string = "no delay predicted"
        
        dispatcher.utter_message("Delay prediction is: "+predict_string)

        return [SlotSet("hour",None),SlotSet("day",None),SlotSet("daym",None),SlotSet("month",None)]
