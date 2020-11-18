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
from sklearn.ensemble import ExtraTreesClassifier


counter = 0

#NOTE:- cars_price_clean.csv is the csv file which doesn't include rows with missing features.

cars_price = pd.read_csv('cars_price_clean.csv')


all_cat = ['make', 'model', 'condition', 'fuel_type', 'transmission', 'year', 'mileage(kilometers)', 'volume(cm3)', 'color', 'drive_unit', 'segment']
all_cat = np.array(all_cat)
all_cat = all_cat.reshape(11, 1)

#print(cars_price.dtypes)
#cars_price.dropna().to_csv('cars_price_clean.csv')
y = cars_price['make']

#storing the 'make' classes in 'cat_make' variable
cat_make = np.unique(y)
#print(cat_make)
y = cars_price['model']

#storing the 'model' classes in 'cat_model' variable
cat_model = np.unique(y)
#print(cat_model)

y = cars_price['condition']

#storing the 'condition' classes in 'cat_condition' variable
cat_condition = np.unique(y)
#print(cat_condition)

y = cars_price['fuel_type']

#storing the 'fuel_type' classes in 'cat_fuel_type' variable
cat_fuel_type = np.unique(y)
#print(cat_fuel_type)

y = cars_price['transmission']

#storing the 'transmission' classes in 'cat_transmission' variable
cat_transmission = np.unique(y)
#print(cat_transmission)

#y = cars_price['segment']

#storing the 'segment' classes in 'cat_segment' variable
#cat_segment = np.unique(y)
#print(cat_segment)

y = cars_price['year']

cat_year = np.unique(y)

#storing classes for color
y = cars_price['color']

cat_color = np.unique(y)


#storing classes for drive_unit
y = cars_price['drive_unit']

cat_drive_unit = np.unique(y)

#storing classes for segment
y = cars_price['segment']

cat_segment = np.unique(y)


#print(cars_price.dtypes)
cars_price['priceUSD'] = cars_price['priceUSD'].astype('float64')
#print(cars_price['priceUSD'])



#to understand relation between 'mileage' and 'price'
        
#cars_price = cars_price.dropna(axis = 0, how = 'any', inplace = True) 

#print(cars_price)
        
new_col = cars_price['segment']
#print(new_col)
        
new_col = np.array(new_col)
print(new_col)
nan_counter = 0





#preparing the training data:-
number = LabelEncoder()

#encoding the str column data into numerical data
num_make = number.fit_transform(cars_price['make'])
#max_make = max(num_make)
#num_make = num_make / max_make
print("the shape of num_make is {}".format(num_make.shape))

num_model = number.fit_transform(cars_price['model'])
#max_model = max(num_model)
#num_model = num_model / max_model
print("the shape of num_model is {}".format(num_model.shape))

num_condition = number.fit_transform(cars_price['condition'])
print("the shape of num_condition is {}".format(num_condition.shape))

num_fuel_type = number.fit_transform(cars_price['fuel_type'])
print("the shape of num_fuel_type is {}".format(num_fuel_type.shape))



num_year = number.fit_transform(cars_price['year'])
print("the shape of num_year is {}".format(num_year.shape))

num_mil = cars_price['mileage(kilometers)']
print(num_mil.shape)
#num_mil = num_mil * 1000.0
#max_mil = max(num_mil)
#num_mil = num_mil / float(max_mil)


num_vol = cars_price['volume(cm3)']
print(num_vol.shape)
#max_vol = max(num_vol)
#num_vol = num_vol / float(max_vol)
#num_vol = num_vol / 1000000.0

num_col = number.fit_transform(cars_price['color'])
#max_col = max(num_col)
#num_col = num_col / max_col
print("The shape of num_col is {}".format(num_col.shape))

num_trans = number.fit_transform(cars_price['transmission'])
#max_trans = max(num_trans)
#num_trans = num_trans / max_trans
print("The shape of num_trans is {}".format(num_trans.shape))

num_drive_unit = number.fit_transform(cars_price['drive_unit'])
#max_drive = max(num_drive_unit)
#num_drive_unit = num_drive_unit / max_drive
print("The shape of num_drive_unit is {}".format(num_drive_unit.shape))

num_segment = number.fit_transform(cars_price['segment'])
#max_seg = max(num_segment)
#num_segment = num_segment / max_seg

print("The shape of num_segment is {}".format(num_segment.shape))



target_price = cars_price['priceUSD']
print("The size of The traget_price to predict is   {}".format(target_price.shape))


print(num_make[0])
print(num_model[0])
print(num_condition[0])
print(num_fuel_type[0])
print(num_year[0])






training_data = []
target = []

#Logic for appending the features using training data.
for i in range(0, len(num_year)):
    sample_data = []
    sample_data.append(num_make[i])
    sample_data.append(num_model[i])
    sample_data.append(num_condition[i])
    sample_data.append(num_fuel_type[i])
    sample_data.append(num_year[i])
    sample_data.append(num_mil[i])
    sample_data.append(num_vol[i])
    sample_data.append(num_col[i])
    sample_data.append(num_trans[i])
    sample_data.append(num_drive_unit[i])
    sample_data.append(num_segment[i])
    
    target_data = []
    target_data.append(target_price[i])
    
    training_data.append([sample_data, target_data])
    
#shuffling the training data
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label) 
    

X = np.array(X)
y = np.array(y)


#Scaling the feature data for better computation and increased accuracy
Xs = scale(X)
print(Xs)




y = y.flatten()


print(X.shape)

print(y.shape)



  #Using chi-squared to understand the impact of most important feature on our target_price
  #Infered from chi-squared:- Segment feature of the data is really impactful on the price followed by 'color'
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
print("here are scores: ")
print(dfscores.shape)
dfcolumns = pd.DataFrame(all_cat)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features




#Splitting the data into training and testing(20%)
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size = 0.2, random_state = 0)


#using sklearn's regression model
regressor = LinearRegression()

regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)

print("Accuracy using regression model is:   {}%".format(regressor.score(X_test, y_test) * 100))

print("Here are the actual and predicted values using regression:--------")

#displaying the actual and predicted values for better understanding
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(result.head())




y_test_new = []
y_test_orig = []


#Second model using neural networks
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

model.save('train_cars.h5', hist)

y_new_pred = model.predict(X_test)

print(X_test.shape)

print(y_new_pred.shape)

y_new_pred = y_new_pred.flatten()






#printing the actual and predicted values for comparison
print("Here are the actual and predicted values using neural nets:--------")
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_new_pred})
print(result.head())




#Logic for calculating accuracy on test_data
new_count = 0
for i in range(0, 7004):
    accuracy_pos = y_test[i] + 1000
    accuracy_neg = y_test[i] - 1000
    if(y_new_pred[i] < y_test[i] and y_new_pred[i] > accuracy_neg):
        new_count = new_count + 1
        
    if((y_new_pred[i] > y_test[i]) and (y_new_pred[i] < accuracy_pos)):
        new_count = new_count + 1
       
        
print("Right predictions are out of 7004 are:  {}".format(new_count))

percent = (float(new_count) * 100) / 7004

print("accuracy using neural nets is:   {}  %".format(percent))

print("Accuracy using regression model is:   {}%".format(regressor.score(X_test, y_test) * 100))


print("plot showing test and training loss:---")

plt.title('Loss / Mean Squared Error')
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
plt.show()



#Note:- Used two models for predictions. Both reached the max accuracy of prediction upto 55%. Accuracy can further be increased after monitoring the relationship between data or adding more relevant features
    





	




















