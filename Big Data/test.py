import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split    
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow 
import keras


ipl = pd.read_csv("/ipl dataset.csv")
str_cols = ipl.columns[ipl.dtypes==object]
ipl[str_cols] = ipl[str_cols].fillna('.')

a1 = ipl['venue'].unique()
a2 = ipl['batting_team'].unique()
a3 = ipl['bowling_team'].unique()
a4 = ipl['striker'].unique()
a5 = ipl['bowler'].unique()
  
def labelEncoding(data):
    dataset = pd.DataFrame(ipl)
    feature_dict ={}
      
    for feature in dataset:
        if dataset[feature].dtype==object:
            le = preprocessing.LabelEncoder()
            fs = dataset[feature].unique()
            le.fit(fs)
            dataset[feature] = le.transform(dataset[feature])
            feature_dict[feature] = le
              
    return dataset
  
labelEncoding(ipl)
ipl.drop(["Unnamed: 0"],axis=1)

ip_dataset = ipl[['venue','innings', 'batting_team', 
                      'bowling_team', 'striker', 'non_striker',
                      'bowler']]
  
b1 = ip_dataset['venue'].unique()
b2 = ip_dataset['batting_team'].unique()
b3 = ip_dataset['bowling_team'].unique()
b4 = ip_dataset['striker'].unique()
b5 = ip_dataset['bowler'].unique()
ipl.fillna(0,inplace=True)
  
features={}
  
for i in range(len(a1)):
    features[a1[i]]=b1[i]
for i in range(len(a2)):
    features[a2[i]]=b2[i]
for i in range(len(a3)):
    features[a3[i]]=b3[i]
for i in range(len(a4)):
    features[a4[i]]=b4[i]
for i in range(len(a5)):
    features[a5[i]]=b5[i]

X = ipl[['venue', 'innings','batting_team',
             'bowling_team', 'striker','bowler']].values
y = ipl['y'].values
  
from sklearn.model_selection import train_test_split
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=25,verbose=1, mode='min',)

model = Sequential()
  
model.add(Dense(43, activation='relu'))
model.add(Dropout(0.5))
  
model.add(Dense(22, activation='relu'))
model.add(Dropout(0.5))
  
model.add(Dense(11, activation='relu'))
model.add(Dropout(0.5))
  
model.add(Dense(1))
  
model.compile(optimizer='adam', loss='mse')

predictions = model.predict(X_test)
sample = pd.DataFrame(predictions)
sample['Actual']=y_test
print(sample.head(10))