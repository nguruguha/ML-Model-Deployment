# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:50:12 2023

@author: ggnar
"""

import numpy as np
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained machine learning model
model = pickle.load(open("penguinsRFCModel.pkl", "rb"))

@app.route('/')
def index():
    return render_template('PredictSpecies.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the HTML form
    island = request.form['island']
    if island == 'Torgersen':
        island_t = 1
        island_d = 0
    elif island == 'Dream':
        island_t = 0
        island_d = 1
    else:
        island_t = 0
        island_d = 0
    culmen_length = float(request.form['culmen_length'])
    culmen_depth = float(request.form['culmen_depth'])
    flipper_length = float(request.form['flipper_length'])
    body_mass = float(request.form['body_mass'])
    sex = request.form['sex']
    if sex == 'Male':
        sex = 1
    else:
        sex = 0
    
    # Create a Pandas DataFrame with the user input
    data = {
            'Culmen Length (mm)': [culmen_length],
            'Culmen Depth (mm)': [culmen_depth],
            'Flipper Length (mm)': [flipper_length],
            'Body Mass (g)': [body_mass],
            'Island_Dream': [island_d],
            'Island_Torgersen': [island_t],
            'Sex_Male': [sex]}
    df = pd.DataFrame(data)
    df = pd.get_dummies(df)
    print(df)
    
    # Use the pre-trained model to predict the species of penguin
    prediction_text = model.predict(df)[0]
    
        
    return render_template('PredictSpecies.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
