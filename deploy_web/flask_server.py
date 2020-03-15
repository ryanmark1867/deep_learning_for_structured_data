# tester app to exercised Flask

from flask import Flask, render_template, request
from string import Template
from OpenSSL import SSL
import pickle
from rasa.nlu.model import Interpreter
import requests
import json
import pandas as pd
import numpy as np
import time
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
import os
import logging
import yaml

# load config gile
current_path = os.getcwd()
print("current directory is: "+current_path)
path_to_yaml = os.path.join(current_path, 'deploy_web_config.yml')
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

'''
LOADING THE MODEL HERE CAUSES AN ERROR - LOADING IT WHEN THE PAGE IS LOADED IS FINE

# load the Keras model
loaded_model = load_model(model_path)
loaded_model._make_predict_function()

loaded_model.summary()
'''

# brute force a scoring sample, bagged from test set
score_sample = {}
score_sample['hour'] = np.array([18])
score_sample['Route'] = np.array([0])
score_sample['daym'] = np.array([21])
score_sample['month'] = np.array([0])
score_sample['year'] = np.array([5])
score_sample['Direction'] = np.array([1])
score_sample['day'] = np.array([1])




app = Flask(__name__)

HTML_TEMPLATE = Template("""
<h1>Hello ${file_name}!</h1>

<img src="https://image.tmdb.org/t/p/w342/${file_name}" alt="poster for ${file_name}">

""")



@app.route('/')
def homepage():   
    graphic_example = "https://raw.githubusercontent.com/ryanmark1867/webview_rasa_example/master/media/Flag_of_Ontario.svg"
    '''
    image_URL = request.args.get('image')
    description_text = request.args.get('description')
    province = request.args.get('province')
    print("image_URL is "+str(image_URL))
    print("province is "+str(province))
    title_text = "Flag of "+province
    title = {'titlename':title_text}
    image = {'graphicname':image_URL}
    description = {'descriptionname':description_text}
    return render_template('home.html',title=title,image = image,description=description)
    '''
    title_text = "Test title"
    title = {'titlename':title_text}
    return render_template('home.html',title=title)
    #return """<h1>Test of web page Feb 9 night</h1>"""
    
@app.route('/test-link/')
def about():
    print("I got tested")
    loaded_model = load_model(model_path)
    loaded_model._make_predict_function()
    preds = loaded_model.predict(score_sample, batch_size=BATCH_SIZE)
    logging.warning("pred is "+str(preds))
    logging.warning("preds[0] is "+str(preds[0]))
    logging.warning("preds[0][0] is "+str(preds[0][0]))
    print("here is a prediction "+str(preds[0][0]))
    return "here is a prediction "+str(preds[0][0])



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    
