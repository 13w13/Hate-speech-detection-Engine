#import libraries
import logging
from flask import Flask, request, jsonify, render_template
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
#from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import torch.utils.data
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig
from nltk.tokenize.treebank import TreebankWordTokenizer
from gensim.models import KeyedVectors

#Initialize the flask App
app = Flask(__name__)

handler = logging.FileHandler("test.log")  # Create the file logger
app.logger.addHandler(handler)             # Add it to the built-in logger
app.logger.setLevel(logging.DEBUG) 

device=torch.device('cpu')
MAX_SEQUENCE_LENGTH = 300 ## 220 in training
SEED = 1234
BATCH_SIZE = 512
#BERT_MODEL_PATH = "C:\Users\antoi\Desktop\TBS\UE 6\AI and BIG DATA Management\Hate-speech-detection-Engine\input\bert-pretrained-models\uncased_l-12_h-768_a-12\uncased_L-12_H-768_A-12"
#BERT_MODEL_PATH = "C:\Users\antoi\Desktop\TBS\UE6\AIandBIGDATAManagement\Hate-speech-detection-Engine\input\bert-pretrained-models\uncased_l-12_h-768_a-12\uncased_L-12_H-768_A-12"
BERT_MODEL_PATH = "input\\bert-pretrained-models\\uncased_l-12_h-768_a-12\\uncased_L-12_H-768_A-12"
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
bert_config = BertConfig("input\\arti-bert-inference\\bert\\bert_config.json")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)

#tqdm.pandas()
CRAWL_EMBEDDING_PATH = CRAWL_EMBEDDING_PATH = "input\\gensim-embeddings-dataset\\crawl-300d-2M.gensim\\crawl-300d-2M.gensim"
GLOVE_EMBEDDING_PATH = GLOVE_EMBEDDING_PATH = "input\\gensim-embeddings-dataset\\glove.840B.300d.gensim\\glove.840B.300d.gensim"
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

def is_interactive():
    return 'SHLVL' not in os.environ

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    #with open(path,'rb') as f:
    emb_arr = KeyedVectors.load(path)
    return emb_arr

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

def handle_contractions(x):
    x = tokenizer.tokenize(x)
    return x

def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def predict_words(tweet): 

  test_input_df = pd.DataFrame(np.array([[1,tweet]]),
                   columns=['id', 'comment_text'])
  
  test_input_df['comment_text'] = test_input_df['comment_text'].astype(str) 

  X_test = convert_lines(test_input_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)
  test_preds = np.zeros((len(X_test)))

  test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
  test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
  tk0 = tqdm(test_loader)
  for i, (x_batch,) in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
    test_preds[i * 512:(i + 1) * 512] = pred[:, 0].detach().cpu().squeeze().numpy()

  test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()  

  submission_bert = pd.DataFrame.from_dict({
    'id': test_input_df['id'],
    'prediction': test_pred
  }) 

  return float(submission_bert['prediction'].values)

if not is_interactive():
    def nop(it, *a, **k):
        return it

    tqdm = nop

    fastprogress.fastprogress.NO_BAR = True
    #master_bar, progress_bar = force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar

seed_everything()

model = BertForSequenceClassification(bert_config, num_labels=1)
model.load_state_dict(torch.load("input\\arti-bert-inference\\bert\\bert_pytorch.bin", map_location=torch.device('cpu')))

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
    app.logger.info(tweet)

    ############

    # Input 

    ##faire les memes operations de test que dans le notebook 

    #return(int_features)

    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)

    prob_prediction=predict_words(tweet[0])

    if prob_prediction >= 0.6: 
        prediction = "Injuries "
    elif prob_prediction >= 0.4 and prob_prediction < 0.6: 
        prediction = "Neutral "
    else:
        prediction = "Non toxic "


    return render_template('index.html', prediction_text='Prediction is :{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)