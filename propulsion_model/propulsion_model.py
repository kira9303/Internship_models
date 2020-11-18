#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[206]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression
import math
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[207]:


data = pd.read_csv('propulsion.csv')


# In[208]:


data.head()
#data.dtypes


# In[209]:


#Not considering two features with constant value

cat_feature = ['Lever position (lp) [ ]', 'Ship speed (v) [knots]', 'Gas Turbine shaft torque (GTT) [kN m]','Gas Turbine rate of revolutions (GTn) [rpm]','Gas Generator rate of revolutions (GGn) [rpm]','Starboard Propeller Torque (Ts) [kN]','Port Propeller Torque (Tp) [kN]','HP Turbine exit temperature (T48) [C]','GT Compressor outlet air temperature (T2) [C]','HP Turbine exit pressure (P48) [bar]','GT Compressor outlet air pressure (P2) [bar]','Gas Turbine exhaust gas pressure (Pexh) [bar]','Turbine Injecton Control (TIC) [%]','Fuel flow (mf) [kg/s]']
print(len(cat_feature))



# In[210]:


#logic for appending features 

X = [] #features
new_sample = data[cat_feature[2]]
print(len(new_sample)) #length of rows

for i in range(0, len(new_sample)):   #length for rows
    sample_arr = []
    for j in range(0, len(cat_feature)):  #For columns
        
        sample = data[cat_feature[j]]
        #print(sample.shape)
        sample_arr.append(sample[i])
    
    X.append(sample_arr)
    
    
X = np.array(X)
print(X.shape)
    
 


# In[211]:


target_1 = []  #GT Compressor decay state coefficient.
target_2 = []  #GT Turbine decay state coefficient.


sample_y = data['GT Compressor decay state coefficient.']
#print(sample_y[1])


for i in range(0, len(sample_y)):
    target_1.append(sample_y[i])
    

target_1 = np.array(target_1)


print(target_1)


# In[212]:


Xs = scale(X)
print(Xs)


# In[213]:


#splitting_data 
X_train, X_test, y_train, y_test = train_test_split(X, target_1, test_size = 0.2, random_state = 0)


#using sklearn's regression model
regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print("Accuracy using regression model is:   {}%".format(regressor.score(X_test, y_test) * 100))

y_pred = regressor.predict(X_test)


# In[214]:


#printing the actual and predicted_preds for better accuracy. (84% accuracy)
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
result.head()


# In[215]:



#Prediction using a nural net with (mean_squared_error as loss func)
model = Sequential()
model.add(Dense(20, input_dim = X_train.shape[1], activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))

model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))




model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

hist = model.fit(X_train, y_train, epochs = 20, validation_split = 0.2, shuffle=True)

model.save('propulsion_train.h5', hist)

y_new_pred = model.predict(X_test)


# In[216]:


#printing the actual and predicted_preds for better accuracy.
y_new_pred = y_new_pred.flatten()
new_result = pd.DataFrame({'Actual': y_test, 'Predicted': y_new_pred})
new_result.head()


# In[217]:



#Storing first 10 predictions (of target_1) for further use if needed
sample_pred = []
sample_original_pred = []

for i in range(0, 10):
    sample_pred.append(y_test[i])
    sample_original_pred.append(y_new_pred[i])
    
sample_pred = np.array(sample_pred)
sample_original_pred = np.array(sample_original_pred)
    
print(sample_pred.shape)
print(sample_original_pred.shape)

sample_X = []

for i in range(0, 10):
    sample_X.append(X_test[i])
    
sample_X = np.array(sample_X)


# In[218]:


train_hist = hist.history['loss']
print(train_hist)


# In[219]:


target_2 = []  #GT Turbine decay state coefficient.

sample_y_2 = data['GT Turbine decay state coefficient.']
print(sample_y_2)
    
for j in range(0, len(sample_y_2)):
    target_2.append(sample_y_2[j])
    
target_2 = np.array(target_2)

print(target_2)


# In[220]:


#NOTE:- time for predicting target_2 (GT Turbine decay state coefficient.)

X_train, X_test, y_train, y_test = train_test_split(X, target_2, test_size = 0.2, random_state = 0)

new_regressor = LinearRegression()

new_regressor.fit(X_train, y_train)

y2_pred = new_regressor.predict(X_test)

print("Accuracy using regression model is:   {}%".format(new_regressor.score(X_test, y_test) * 100))

y2_pred = regressor.predict(X_test)


# In[221]:


#Printing the actual and predicted values for better understanding (accuracy:- 91%)
new_result = pd.DataFrame({'Actual': y_test, 'Predicted': y2_pred})
new_result.head()


# In[ ]:




