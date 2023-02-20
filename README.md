Welcome to my very first machine learning practice project! 🎉

The project was focused on predicting California housing prices, this project was an incredible learning experience where I had the opportunity to perform all the essential steps of a data science project, including data extraction, EDA, preprocessing, modeling, gaining insights from model prediction.

I also went a step further and created a front end flask webapp to make the trained model available on the web. This gave me an opportunity to showcase my skills in full-stack development and put my machine learning model into production.

Through this project, I was able to gain valuable insights into the data and build a model that accurately predicted housing prices in California. I used various machine learning techniques, including linear regression, decision trees, and random forests, Gradient Boosted Regression to build the most effective model.

This project marks an important milestone in my data science journey, and I am thrilled to finally be able to apply my theoretical knowledge in practice and learn so many new technologies and techniques along the way, and I am excited to continue learning and growing in this field.

I am grateful for this opportunity and excited to see what the future holds! 🌟 

Best regards,
Aaditya Bansal

I have shared the Complete Development process of this project using Blogs and you can see [The Development Process Blogs](https://adiaturb.editorx.io/aaditya-portfolio/post/california-housing-price-prediction-machine-learning-model-project)

## Overview

This Project covers all the necessary steps to complete the Machine Learning Task of Predicting the Housing Prices on California Housing Dataset available on scikit-learn.
I performed the following steps for successfully creating a model for house price prediction:

### 1. Data Extraction 
* Import libraries. 
* Import Dataset from scikit-learn. 
* Understanding the given Description of Data and the problem Statement 
* Take a look at different Inputs and details available with dataset. 
* Storing the obtained dataset into a Pandas Data Frame. 

### 2. EDA (Exploratory Data Analysis) and Visualization
* Getting a closer Look at obtained Data.
* Exploring different Statistics of the Data (Summary and Distributions)
* Looking at Correlations (between indiviual features and between Input features and Target)
* Geospatial Data / Coordinates - Longitude and Lattitude features

### 3. Preprocessing
* Dealing with Duplicate and Null (NaN) values
* Dealing with Categorical features (e.g. Dummy coding)
* Dealing with Outlier values
* Visualization (Box-Plots)
* Using IQR
* Using Z-Score
* Seperating Target and Input Features
* Target feature Normalization (Plots and Tests)
* Splitting Dataset into train and test sets
* Feature Scaling (Feature Transformation)

### 4. Modeling 
* Specifying Evaluation Metric R squared (using Cross-Validation) 
* Model Training - trying multiple models and hyperparameters: 
* Linear Regression 
* Polynomial Regression 
* Ridge Regression 
* Decision Trees Regressor 
* Random Forests Regressor 
* Gradient Boosted Regressor 
* eXtreme Gradient Boosting (XGBoost) Regressor 
* Support Vector Regressor 
* Model Selection (by comparing evaluation metrics) 
* Learn Feature Importance and Relations 
* Prediction 

### 5. Deployment 
* Exporting the trained model to be used for later predictions. (by storing model object as byte file - Pickling) 
* Creating a Flask App for deploying model on the web.
