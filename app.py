#!/usr/bin/env python
# coding: utf-8



import pickle

#libraries
import pandas as pd # data processing
import numpy as np # linear algebra

# data transformation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import pickle as pickle
import streamlit as st

# Loading data
data=pd.read_excel('bankruptcy-prevention.xlsx')
st.title("Bankruptcy Prevention")
data.head()
Industrial_risk = st.selectbox('industrial_risk', [0,0.5,1])
Management_risk = st.selectbox('management_risk', [0,0.5,1])
Financial_flexibility = st.selectbox('financial_flexibility', [0,0.5,1])
Credibility = st.selectbox('credibility', [0,0.5,1])
Competitiveness =st.selectbox('competitiveness', [0,0.5,1])
Operating_risk = st.selectbox('operating_risk', [0,0.5,1])
  
data[' class'].unique()


# OHE on Features
data_F=pd.get_dummies(data.iloc[:,1:])

# Lebel encoding on target
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data[' class']=label_encoder.fit_transform(data[' class'])

# forming all encoded columns together
data=pd.concat([data[' class'],data_F],axis=1)

# Dividing data into Features(x) & Target(y)
x = data.iloc[:,1:]
y= data[' class']

# Train-Test Split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
#num_trees = 60
#AdaBoost Classification
kfold=KFold(n_splits=5,random_state=72,shuffle=True)
model = [AdaBoostClassifier(n_estimators=60,random_state=8)]
#model.fit(x_train, y_train)
#model_knn.predict(x_test)
result_ab = cross_val_score(model, x_train,y_train, cv=kfold)
result_ab.mean()
#Accuracy
print(result_ab.mean())

#Pickel file
ffilename = 'final_Adaboost_model.pkl'
pickled_model=pickle.load(open('final_Adaboost_model.pkl','rb'))
pickled_model.fit(x_train,y_train)
pk=pickled_model.predict(x_test)


if st.button('prevention type'):
   # prediction=pickled_model.predict(data.drop(' class',axis=1))
   if pk.any()==0:
        prediction = "Bankruptcy"
   else:
         prediction = "Non-Bankruptcy"
 
   st.title("business type is "+str(prediction))
    





    





