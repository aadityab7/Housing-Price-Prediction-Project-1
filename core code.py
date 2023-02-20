#IMPORTING LIBRARIES
import pandas as pd
import numpy as np

#PICKLE TO STORE THE FITTED (TRAINED) OBJECTS AS BYTE FILES TO BE USED LATER 
import pickle

#IMPORTING DATA
from sklearn.datasets import fetch_california_housing  
cal_housing_dataset = fetch_california_housing(as_frame = True)

#STORING THE HOUSING DATA IN A SEPERATE PANDAS DATAFRAME VARIABLE
dataset = cal_housing_dataset.frame

#Seperating Target and Features
y_target = dataset['MedHouseVal']
X_features = dataset.drop(['MedHouseVal'], axis = 1)

#IMPORTING LIBRARIES TO PERFORM NORMALIZATION 

y_target_log = np.log1p(y_target)

# TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size = 0.2, random_state = 1)

#STANDARDIZE THE DATASET

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#SCALING THE TRAIN DATA - FIT AND TRANSFORM

X_train = scaler.fit_transform(X_train)

#TRANSFORM (SCALE) TEST DATA USING SAME SCALER FITTED USED TRAIN SET   

X_test = scaler.transform(X_test)

pickle.dump(scaler, open("scaler.pkl", 'wb')) #here wb = write byte

#FUNCTION TO CALCULATE THE EVALUATION SCORE USING CROSS VALIDATION AND R SQAURE SCORING
from sklearn.model_selection import cross_val_score

def calculate_eval_metric(model, X, y, cv = 3):
    scores = cross_val_score(model, X, y, cv = cv, scoring = 'r2')
    print("Evaluation score on 3 cross-validation sets : ", scores)
    print("Average R squared score : ", scores.mean())
    return scores.mean()

#LIB
from sklearn.ensemble import GradientBoostingRegressor

#BOOSTING MODEL
gradient_boosting_model = GradientBoostingRegressor(random_state = 0, n_estimators = 100, learning_rate = 0.1, max_depth = 8)

#FIT THE MODEL USING THE COMPLETE TRAINING DATA FOR BETTER PERFORMANCE
gradient_boosting_model.fit(X_train, y_train)

#EVALUATION SCORE (R SQUARED) ON TRAIN DATA - HOW WELL OUR MODEL PERFORMS ON TRAINING DATA
gradient_boosting_model.score(X_train, y_train)

#EVALUATION SCORE ON TEST DATA TO SEE HOW WELL OUR MODEL GENERALIZE TO NEW DATA
gradient_boosting_model.score(X_test, y_test)

#SAVE THE MODEL FOR LATER
pickle.dump(gradient_boosting_model, open("gradient_boosting_model.pkl", 'wb') ) #here wb = write byte