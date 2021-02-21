#import libraries
#import logging
from flask import Flask, request, render_template
#import sys
#import fastai
#from fastai.callbacks import TrainingPhase, GeneralScheduler
#import fastprogress # a remettre
#from fastprogress import master_bar, progress_bar #a remettre 
import numpy as np
import pandas as pd
import os
#import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
#from keras.preprocessing import text, sequence
import torch
from torch import nn
#from torch.utils import data
#from torch.nn import functional as F
#import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig
#from nltk.tokenize.treebank import TreebankWordTokenizer
#from gensim.models import KeyedVectors

#Initialize the flask App
app = Flask(__name__)
#handler = logging.FileHandler("test.log")  # Create the file logger
#app.logger.addHandler(handler)             # Add it to the built-in logger
#app.logger.setLevel(logging.DEBUG) 

device=torch.device('cpu')
MAX_SEQUENCE_LENGTH = 300 ## 220 in training
SEED = 1234
BATCH_SIZE = 512
#BERT_MODEL_PATH = "C:\Users\antoi\Desktop\TBS\UE 6\AI and BIG DATA Management\Hate-speech-detection-Engine\input\bert-pretrained-models\uncased_l-12_h-768_a-12\uncased_L-12_H-768_A-12"
#BERT_MODEL_PATH = "C:\Users\antoi\Desktop\TBS\UE6\AIandBIGDATAManagement\Hate-speech-detection-Engine\input\bert-pretrained-models\uncased_l-12_h-768_a-12\uncased_L-12_H-768_A-12"
#BERT_MODEL_PATH = "input\\bert-pretrained-models\\uncased_l-12_h-768_a-12\\uncased_L-12_H-768_A-12"
BERT_MODEL_PATH = "input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12"
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
bert_config = BertConfig("input/arti-bert-inference/bert/bert_config.json")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)

#tqdm.pandas()
#CRAWL_EMBEDDING_PATH = CRAWL_EMBEDDING_PATH = "input\\gensim-embeddings-dataset\\crawl-300d-2M.gensim\\crawl-300d-2M.gensim"
#GLOVE_EMBEDDING_PATH = GLOVE_EMBEDDING_PATH = "input\\gensim-embeddings-dataset\\glove.840B.300d.gensim\\glove.840B.300d.gensim"
NUM_MODELS = 2
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_LEN = 220

def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_words(tweet): 

    test_input_df = pd.DataFrame(np.array([[1,tweet]]),
                    columns=['id', 'comment_text'])
    
    test_input_df['comment_text'] = test_input_df['comment_text'].astype(str)
 
    X_test = convert_lines(test_input_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)

    test_preds = np.zeros((len(X_test)))

    #test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
    test = TensorDataset(torch.tensor(X_test, dtype=torch.long))

    #test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    test_loader = DataLoader(test, batch_size=512, shuffle=False)
    tk0 = tqdm(test_loader)

    return(model)

    for i, (x_batch,) in enumerate(tk0):
        pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        test_preds[i * 512:(i + 1) * 512] = pred[:, 0].detach().cpu().squeeze().numpy()

    test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()  

    return("tata")

    submission_bert = pd.DataFrame.from_dict({
        'id': test_input_df['id'],
        'prediction': test_pred
    }) 

    return float(submission_bert['prediction'].values)

seed_everything()

model = BertForSequenceClassification(bert_config, num_labels=1)
model.load_state_dict(torch.hub.load_state_dict_from_url("https://www.googleapis.com/drive/v3/files/1RYFMsASHW7a92qa7zW296zgnToRQFeb5?alt=media&key=AIzaSyA0OHTKp3e0TvdIyua79c8jH_v6WBmGEKI", map_location=torch.device('cpu')))

model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

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

    ############

    # Input 

    ##faire les memes operations de test que dans le notebook 

    #return(int_features)

    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)

    prediction = "bonjour"

    prob_prediction=predict_words(tweet[0])

    """
    if prob_prediction >= 0.6: 
        prediction = "Insult "
    elif prob_prediction >= 0.4 and prob_prediction < 0.6: 
        prediction = "Neutral "
    else:
        prediction = "Non toxic "

    """
    return render_template('index.html', prediction_text='Prediction is :{}'.format(prob_prediction))

if __name__ == '__main__':
    #app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)

    