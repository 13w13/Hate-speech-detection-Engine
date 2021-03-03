from flask import Flask, request, jsonify, render_template, Response, make_response
import numpy as np
import pandas as pd
import os
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig
import logging
#import langid
from textblob import TextBlob

#Initialize the flask App
#app = Flask(__name__)
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
BERT_MODEL_PATH = os.getcwd()
np.random.seed(SEED)
torch.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True
bert_config = BertConfig("arti-bert-inference-bert_config.json")
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
    #tk0 = tqdm(test_loader)

    for i, (x_batch,) in enumerate(test_loader):
        pred = model(x_batch, attention_mask=(x_batch > 0), labels=None)
        test_preds[i * 512:(i + 1) * 512] = pred[:, 0].detach().cpu().squeeze().numpy()

    test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()  

    submission_bert = pd.DataFrame.from_dict({
        'id': test_input_df['id'],
        'prediction': test_pred
    }) 

    return float(submission_bert['prediction'].values)

seed_everything()

model = BertForSequenceClassification(bert_config, num_labels=1)
#torch.hub.load_state_dict_from_url("https://www.dropbox.com/s/4320po4qph1lrx6/bert_pytorch.bin?dl=1", model_dir="./",map_location=torch.device('cpu'))
model.load_state_dict(torch.hub.load_state_dict_from_url("https://www.googleapis.com/drive/v3/files/1RYFMsASHW7a92qa7zW296zgnToRQFeb5?alt=media&key=AIzaSyA0OHTKp3e0TvdIyua79c8jH_v6WBmGEKI", model_dir="input/arti-bert-inference/bert",map_location=torch.device('cpu')))
#model.load_state_dict()
for param in model.parameters():
    param.requires_grad = False
model.eval()

#Initialize the flask App
app = Flask(__name__)

#model = pickle.load(open('model.pkl', 'rb')) #changer par le bin

handler = logging.FileHandler("test.log")  # Create the file logger
app.logger.addHandler(handler)             # Add it to the built-in logger
app.logger.setLevel(logging.DEBUG) 

#default page of our web-app
@app.route('/')
def home():
    #print('Hello world!')
    return render_template('index.html', prediction_text=None, status_prediction=False)

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    tweet = [str(x) for x in request.form.values()]
    #app.logger.info(tweet)

    # Input 
    # ##faire les memes operations de test que dans le notebook 


    prediction = "indetermined"
    #prob_prediction = predict_words(tweet[0])
    
    if((len(tweet[0]) > 2) and (TextBlob(tweet[0]).detect_language() == 'en')):
        prob_prediction = predict_words(tweet[0])
    elif(len(tweet[0]) <= 2):
        prob_prediction = predict_words(tweet[0])
    else:
        return render_template('index.html', tweet_text=tweet[0], prediction_text=TextBlob(tweet[0]).detect_language(), status_prediction=True) 
       
    if prob_prediction > 0.7: 
        prediction = "Hate message"
    elif prob_prediction > 0.5 and prob_prediction <= 0.7: 
        prediction = "Offensive message"
    else:
        prediction = "Neutral"

    return render_template('index.html', tweet_text=tweet[0], prediction_text=prediction, status_prediction=True)

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
    csvData['classif'] = None
    
    for i in range(len(csvData)): 
        #csvData['pred'][i] = 0.5
        csvData['pred'][i] = predict_words(csvData.loc[i].values[0])

        if csvData['pred'][i] > 0.7: 
            csvData['classif'][i] = "Hate message"
        elif csvData['pred'][i] > 0.5 and csvData['pred'][i] <= 0.7: 
            csvData['classif'][i] = "Offensive message"
        else:
            csvData['classif'][i] = "Neutral"


        # #print(tweet_df.loc[i].values[0] + " " + str(predict_words(tweet_df.loc[i].values[0])))

    #app.logger.info(csvData)

    #output = round(prediction[0], 2)
    #output = "toto"

    resp = make_response(csvData.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=your_prediction.csv"
    resp.headers["Content-Type"] = "text/csv"

    return resp

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='127.0.0.1', port=port, debug=True)
    #port = int(os.environ.get("PORT", 5000))
    #app.run(host='0.0.0.0', port=port, debug=True)
