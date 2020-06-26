    # -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:44:16 2020

@author: krnsa
"""

import numpy as np
import pickle
from flask import Flask , render_template, jsonify , request

#initializing the flask application
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

# Routing the application to route folder 
@app.route('/')
def home():
    return render_template('index.htm')

# Routing to the prediction outcome
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_data = [np.array(int_features)]
    prediction = model.predict_proba(final_data)
    Out_var = prediction[0][1]
    Output = "{:.0%}".format(Out_var)
    return render_template('index.htm',prediction_text = ' Probability of purchase {}'.format(Output))

if __name__ == '__main__':
    app.run(debug = True)