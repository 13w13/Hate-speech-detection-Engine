#import libraries 
'''
import sys
import fastai
from fastai.callbacks import TrainingPhase, GeneralScheduler
import fastprogress # a remettre
from fastprogress import master_bar, progress_bar #a remettre 
import numpy as np
import pandas as pd
import os
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import torch.utils.data
from tqdm import tqdms
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig
from nltk.tokenize.treebank import TreebankWordTokenizer
from gensim.models import KeyedVectors
'''

#import libraries
import logging
import numpy as np
import pandas as pd 
from flask import Flask, request, jsonify, render_template, Response, make_response
import pickle

#Initialize the flask App
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb')) #changer par le bin

handler = logging.FileHandler("test.log")  # Create the file logger
app.logger.addHandler(handler)             # Add it to the built-in logger
app.logger.setLevel(logging.DEBUG) 

#default page of our web-app
@app.route('/')
def home():
    #print('Hello world!')
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
#def button_clicked():
#    print('Hello world!')
#    return redirect('/')
def predict():
    '''
    For rendering results on HTML GUI
    '''
    tweet = [str(x) for x in request.form.values()]

    #app.logger.info(tweet)

    ##faire les memes operations de test que dans le notebook 

    #return(int_features)

    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)

    #app.logger.info(type(tweet))
    output = "tata"
    
    return render_template('predict.html', prediction_text='Is it an hate message? :{}'.format(output))

@app.route('/predict_csv',methods=['POST'])
def predict_csv():
    return render_template('predict.html', prediction_text='Is it an hate message? :{}'.format('tutu'))

@app.route('/predict_csv/file',methods=['POST'])
def predict_csv_file():
    '''
    For rendering results on HTML GUI
    '''
    # get the uploaded file
    uploaded_file = request.files['file']
    csvData = pd.read_csv(uploaded_file, header=None)
    csvData = csvData.rename(columns={0:'tweet'})
    csvData['pred'] = None
    
    for i in range(len(csvData)): 
        csvData['pred'][i] = 0.5
        #csvData['pred'][i] = predict_words(csvData.loc[i].values[0])
        # #print(tweet_df.loc[i].values[0] + " " + str(predict_words(tweet_df.loc[i].values[0])))

    #app.logger.info(csvData)

    #output = round(prediction[0], 2)
    #output = "toto"

    resp = make_response(csvData.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"

    return resp

if __name__ == "__main__":
    app.run(debug=True)