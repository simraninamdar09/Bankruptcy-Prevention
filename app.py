#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Load the dataset
data=pd.read_excel("bankruptcy-prevention.xlsx")

# Encoding Data
label_encoder = preprocessing.LabelEncoder()
data[' class'] = label_encoder.fit_transform(data[' class'])


# Split the data into features (X) and target (y)
x = data.drop(' class', axis=1)
y = data[' class']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Create the AdaBoost classifier
kfold=KFold(n_splits=5,random_state=72,shuffle=True)
model_ab= AdaBoostClassifier(n_estimators=60, random_state=8)        
result_ab = cross_val_score(model_ab, x, y, cv=kfold)
#Accuracy
print(result_ab.mean())


filename = 'final_Adaboost_model.pkl'
pickle.dump(model_ab, open(filename,'wb'))
model_ab.fit(x,y)
pk=model_ab.predict(x_test)

st.title("Bankruptcy-Prevention")

data['industrial_risk']=data['industrial_risk'].replace({0.0:'Low',0.5:'Medium',1:'High'})
data[' management_risk']=data[' management_risk'].replace({0.0:'Low',0.5:'Medium',1:'High'})
data[' financial_flexibility']=data[' financial_flexibility'].replace({0.0:'Low',0.5:'Medium',1:'High'})
data[' credibility']=data[' credibility'].replace({0.0:'Low',0.5:'Medium',1:'High'})
data[' competitiveness']=data[' competitiveness'].replace({0.0:'Low',0.5:'Medium',1:'High'})
data[' operating_risk']=data[' operating_risk'].replace({0.0:'Low',0.5:'Medium',1:'High'})


Industrial_risk = st.selectbox('Industrial_risk', data['industrial_risk'].unique())
Management_risk = st.selectbox(' Management_risk', data[' management_risk'].unique())
Financial_flexibility = st.selectbox(' Financial_flexibility', data[' financial_flexibility'].unique())
Credibility = st.selectbox(' Credibility', data[' credibility'].unique())
Competitiveness = st.selectbox(' Competitiveness', data[' competitiveness'].unique())
Operating_risk = st.selectbox(' Operating_risk', data[' operating_risk'].unique())



if st.button('Prevention Type'):
    df = {
        'industrial_risk': Industrial_risk,
        ' management_risk': Management_risk,
        ' financial_flexibility': Financial_flexibility,
        ' credibility': Credibility,
        ' competitiveness': Competitiveness,
        ' operating_risk': Operating_risk
    }

    df1 = pd.DataFrame(df, index=[1])
    predictions = model_ab.predict(df1)

 #   if predictions.any() == 'High':
  #      prediction_value = 'Non-Bankruptcy'
  #  else:
     #   prediction_value = 'Bankruptcy'
    if prediction.any()==1:
        prediction = "Non-Bankruptcy'"
    else:
        prediction = "Bankruptcy'"
       
    st.title("Mushroom type is " + str(prediction))
    
 #   st.title("Business type is " + str(prediction_value))


    





