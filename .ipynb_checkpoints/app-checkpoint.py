import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__)

#LOAD MODEL
regModel = pickle.load(open("gradient_boosting_model.pkl", "rb"))

#LOAD SCALER
scaler = pickle.load(open("scaler.pkl", "rb"))

#CREATE APP ROUTE
@app.route('/')
def home():
	return render_template("home.html")

#ROUTE FOR PREDICTION API
@app.route('/predict_api', methods = ['POST'])
def predict_api():
	data = request.json['data']

	print(data)
	print(np.array(list(data.values())).reshape(1, -1))

	new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))

	output = regModel.predict(new_data)
	#print(output[0])

	return jsonify(output[0])

if(__name__ == "__main__"):
	app.run(debug = True)