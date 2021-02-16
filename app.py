#import libraries
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template
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
    int_features = [str(x) for x in request.form.values()]
    app.logger.info(int_features)

    #return(int_features)

    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    output = "toto"
    return render_template('index.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)