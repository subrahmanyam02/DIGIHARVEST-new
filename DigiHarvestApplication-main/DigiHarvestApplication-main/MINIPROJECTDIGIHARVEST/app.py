
from flask import Flask, render_template, request, Markup,redirect, url_for, session
import pandas as pd
from utils.fertilizer import fertilizer_dict
import os
import numpy as np
import pickle
import sys
import datetime
from bs4 import BeautifulSoup
"""from flask_mysqldb import MySQL
import MySQLdb.cursors"""
import re


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath ='Trained_model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")
def pred_pest(pest):
    
        test_image = load_img(pest, target_size = (64, 64)) # load image 
        print("@@ Got Image for prediction")
  
        test_image = img_to_array(test_image)# convert image to np array and normalize
        test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
        result = model.predict(test_image) # predict diseased palnt or not
        print('@@ Raw result = ', result)
  
        pred = np.argmax(result, axis=1)
        return pred
    

crop_recommendation_model = pickle.load(open("RFmodel.pkl", "rb"))

yield_prediction_model = pickle.load(open("RFYield.pkl", "rb"))

fpath = 'model_weight_Adam.hdf5'
model2 = load_model(fpath)
print(model2)

print("Model2 Loaded Successfully")

def pred_weed(weed):
    
        test_image = load_img(weed, target_size = (51, 51)) # load image 
        print("@@ Got Image for prediction")
  
        test_image = img_to_array(test_image)# convert image to np array and normalize
        test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
        result = model2.predict(test_image) 
        print('@@ Raw result = ', result)
  
        find = np.argmax(result, axis=1)
        return find

app = Flask(__name__)
"""
# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '11113333'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '12345678'
app.config['MYSQL_DB'] = 'harvest'

# Intialize MySQL
mysql = MySQL(app)
"""
@ app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():

    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv('Data/Crop_NPK.csv')

    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = N_desired- N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,
                           recommendation2=response2, recommendation3=response3,
                           diff_n = abs_n, diff_p = abs_p, diff_k = abs_k)



@app.route("/")

@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")


@app.route("/Costofcultivation.html")
def cultivation():
    return render_template("Costofcultivation.html")


@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")

@app.route("/herbicides.html")
def herbicide():
    return render_template("herbicides.html")   

@app.route("/weed.html")
def weed():
    return render_template("weed.html")


@app.route("/Yieldprediction.html")
def yieldpred():
    return render_template("Yieldprediction.html")


@app.route("/Yieldresult")
def yieldresult():
    return render_template("Yieldresult.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename

        file_path = os.path.join('C:/Users/Administrator/Desktop/flaskdemo (2)/flaskdemo/static/user uploaded/', filename)
        file.save(file_path)

        pred = pred_pest(pest=file_path)
    
        if pred[0] == 0:
            pest_identified = 'aphids'
        elif pred[0] == 1:
            pest_identified = 'armyworm'
        elif pred[0] == 2:
            pest_identified = 'beetle'
        elif pred[0] == 3:
            pest_identified = 'bollworm'
        elif pred[0] == 4:
            pest_identified = 'earthworm'
        elif pred[0] == 5:
            pest_identified = 'grasshopper'
        elif pred[0] == 6:
            pest_identified = 'mites'
        elif pred[0] == 7:
            pest_identified = 'mosquito'
        elif pred[0] == 8:
            pest_identified = 'sawfly'
        elif pred[0] == 9:
            pest_identified = 'stem borer'

        return render_template(pest_identified + ".html",pred=pest_identified)

@ app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/'+final_prediction+'.jpg')


@ app.route('/yield_prediction', methods=['POST'])
def  yield_prediction():
    if request.method == 'POST':
        crop = int(request.form['Crop'])
        state = int(request.form['State'])
        coca = int(request.form['Cost of Cultivation (`/Hectare) A2+FL'])
        cocb = float(request.form['Cost of Cultivation (`/Hectare) C2'])
        cop = float(request.form['Cost of Production (`/Quintal) C2'])
        ypq = float(request.form['Yield (Quintal/ Hectare)'])
        phcp = float(request.form['Per Hectare Cost Price'])
        data = np.array([[crop, state, coca, cocb, cop, ypq, phcp]])
        my_pred = yield_prediction_model.predict(data)
        newarr = np.array_split(my_pred, 1)
        return render_template('Yieldresult.html', costofcultivation=newarr[0][0][0],totalyield=newarr[0][0][1])
       

@app.route("/predictweed", methods=['GET', 'POST'])

def predictweed():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename

        file_path = os.path.join('C:/Users/Administrator/Desktop/flaskdemo (2)/flaskdemo/static/user uploaded/', filename)
        file.save(file_path)

        find = pred_weed(weed=file_path)
        
        if  find[0] == 0:
            weed_identified = 'Black-grass'
        elif find[0] == 1:
            weed_identified = 'Charlock'
        elif find[0] == 2:
            weed_identified = 'Cleavers'
        elif find[0] == 3:
            weed_identified = 'Common Chickweed'
        elif find[0] == 4:
            weed_identified = 'Common wheat'
        elif find[0] == 5:
            weed_identified = 'Fat Hen'
        elif find[0] == 6:
            weed_identified = 'Loose Silky-bent'
        elif find[0] == 7:
            weed_identified = 'Maize'
        elif find[0] == 8:
            weed_identified = 'Scentless Mayweed'
        elif find[0] == 9:
            weed_identified = 'Shepherds Purse'
        elif find[0] == 10:
            weed_identified = 'Small-flowered Cranesbill'
        elif find[0] == 11:
            weed_identified = 'Sugar beet'

        return render_template("result.html",weed_identified=weed_identified)
   

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=80)