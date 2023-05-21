#!/usr/bin/env python
# coding: utf-8

# In[31]:

#import pickle
pip install openpyxl
#libraries
import pandas as pd # data processing
import numpy as np # linear algebra

#ploting libraries
#import seaborn as sns
#import matplotlib.pyplot as plt 

#feature engineering
from sklearn import preprocessing

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



data.head()

data['industrial_risk']=np.where(data['industrial_risk'].isin(['0','0.5','1']),
                           data['industrial_risk'].str.title(),
                           'Other_risk')

data['management_risk']=np.where(data['management_risk'].isin(['0','0.5','1']),
                           data['management_risk'].str.title(),
                           'Other_risk')

data['financial_flexibility']=np.where(data['financial_flexibility'].isin(['0','0.5','1']),
                           data['financial_flexibility'].str.title(),
                           'Other_risk')

data['credibility']=np.where(data['credibility'].isin(['0','0.5','1']),
                           data['credibility'].str.title(),
                           'Other_risk')

data['competitiveness']=np.where(data['competitiveness'].isin(['0','0.5','1']),
                           data['competitiveness'].str.title(),
                           'Other_risk')

data['operating_risk']=np.where(data['operating_risk'].isin(['0','0.5','1']),
                           data['operating_risk'].str.title(),
                           'Other_risk')






data['industrial_risk']=data['industrial_risk'].replace({'0':'low','0.5':'medium','1':'high'})

data['management_risk']=data['management_risk'].replace({'0':'low','0.5':'medium','1':'high'})

data['financial_flexibility']=data['financial_flexibility'].replace({'0':'low','0.5':'medium','1':'high'})

data['credibility']=data['credibility'].replace({'0':'low','0.5':'medium','1':'high'})

data['competitiveness']=data['competitiveness'].replace({'0':'low','0.5':'medium','1':'high'})

data['operating_risk']=data['operating_risk'].replace({'0':'low','0.5':'medium','1':'high'})




# In[48]:
data['class'].unique()

st.title("Bankruptcy Prevention")

industrial_risk= st.selectbox('industrial_risk', data['industrial_risk'].unique())
management_risk= st.selectbox('management_risk', data['management_risk'].unique())
financial_flexibility= st.selectbox('financial_flexibility', data['financial_flexibility'].unique())
credibility= st.selectbox('credibility', data['credibility'].unique())
competitiveness= st.selectbox('competitiveness', data['competitiveness'].unique())
operating_risk= st.selectbox('operating_risk', data['operating_risk'].unique())




# OHE on Features
data_F=pd.get_dummies(data.iloc[:,1:])

# Lebel encoding on target
label_encoder=preprocessing.LabelEncoder()
data['class']=label_encoder.fit_transform(data['class'])

# forming all encoded columns together
data=pd.concat([data['class'],data_F],axis=1)



# Dividing data into Features(X) & Target(y)
X = data.iloc[:,1:]
y=data['class']
# Train-Test Split 
#Train test split will be a 70:30 ratio respectively.
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
#Adaboost Classification
model_knn = KNeighborsClassifier(n_neighbors=2)
model_knn.fit(x_train,y_train)
       
result_knn= model_knn.score(x_test,y_test)
#Accuracy
print(accuracy_score(x_test,y_test))


ffilename = 'final_KNN_model.pkl'
pickled_model=pickle.load(open('final_KNN_model.pkl','rb'))
pickled_model.fit(x_train,y_train)
pk=pickled_model.predict(x_test)

if st.button('Mushroom type'):
   # query = np.array([cap_shape,cap_surface,cap_color,bruises,odor,
                      #gill_spacing,gill_size,gill_color,stalk_shape,stalk_root,stalk_surface_above_ring,
                      #stalk_color_above_ring,ring_type,spore_print_color,population,habitat])

    #query = query.reshape(1, 16)

    #prediction=pickled_model.predict(data.drop('class',axis=1))
    
    if pk.any()==1:
        prediction = "edible Mushroom"
    else:
        prediction = "Poidn Mushroom"
       
    st.title("Mushroom type is " + str(prediction))
    





    





