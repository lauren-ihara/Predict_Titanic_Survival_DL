#%% md
# # Binary Classification
#%%
import pandas as pd
#%%
df = pd.read_csv('ship.csv')
#%%
df.head()
#%%
df = pd.read_csv("ship.csv")

from ship_helper import *

df.rename(columns={'Gender':"Sex"},inplace=True)

df_cleaned = cleanShipData(df)
#%%
#separate x and y  (we want to predict whether someone survived)
X = df_cleaned.drop(columns = 'Survived')
y = df_cleaned['Survived']

#split the data 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y,  random_state=42, stratify=y)
#%%
#standardize the data 
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(x_train)
x_train_std =std.transform(x_train)
x_test_std = std.transform(x_test)
#%%
#what i need when i define my model: 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.metrics import Precision, Recall
#%% md
# build the model:
#%%
x_train.shape[1]
#%% md
# Define model:
#%%
model = Sequential()
model.add(Input(shape=(x_train.shape[1],))) 
model.add(Dense(units= 12, activation = 'relu')) #96 weights
model.add(Dense(units=15, activation = 'relu')) #195 weights
model.add(Dense(units=20, activation = 'relu')) #320 weights
model.add(Dense(units=1, activation = 'sigmoid')) #21 weights, sigmoid because we want to do a binary classification (survived or not)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()]) #this only works for binary classification
#%%
model.summary()
#%%
pip install livelossplot==0.1.2
#%%
#fit the model: 
model.fit(x_train_std,y_train,epochs=50,batch_size=64,validation_data=(x_test_std,y_test), callbacks=[PlotLossesKerasTF()])
#%%
