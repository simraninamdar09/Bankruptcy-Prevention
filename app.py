#!/usr/bin/env python
# coding: utf-8

# In[31]:

import pickle

#libraries
import pandas as pd # data processing
import numpy as np # linear algebra

#ploting libraries
#import seaborn as sns
#import matplotlib.pyplot as plt 

#feature engineering
#from sklearn import preprocessing

# data transformation
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier

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

#st.title("Bankruptcy Prevention")


# OHE on Features
data_F=pd.get_dummies(data.iloc[:,1:])

# Lebel encoding on target
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data[' class']=label_encoder.fit_transform(data[' class'])

# forming all encoded columns together
data=pd.concat([data[' class'],data_F],axis=1)



# Dividing data into Features(X) & Target(y)
x = data.iloc[:,1:]
y= data[' class']
# Train-Test Split 
#Train test split will be a 70:30 ratio respectively.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
#Adaboost Classification
model_knn = KNeighborsClassifier(n_neighbors=2)
model_knn.fit(x_train,y_train)
result_knn = model_knn.score(x_test,y_test)

#Accuracy
print(np.round(result_knn, 4))


ffilename = 'final_KNN_model1.pkl'
pickled_model=pickle.load(open('final_KNN_model1.pkl','rb'))
pickled_model.fit(x_train,y_train)
pk=pickled_model.predict(x_test)


if st.button('prevention type'):
  
   prediction=pickled_model.predict(data.drop(' class',axis=1))
    
   # if pk.any() == 0:
       # prediction = "Bankruptcy"
    #else:
      #  prediction = "Non-Bankruptcy"
      if prediction == 0:
        prediction = 'Bankruptcy'
      else:
          prediction = 'Non-bankruptcy'
        
       
    st.title("business type is " + str(prediction))
    





    





