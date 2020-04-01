import math
import pandas as pd
import pandas_datareader as web
import numpy as np
import datetime as dt
import time
import pickle
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import os
plt.style.use('fivethirtyeight')


##Initialize the parameters..
today = dt.date.today()
preDayDate = today - timedelta(1)

sDate = '2010-01-01'
eDate = preDayDate
retryCount = 3
datasource = 'yahoo'
stock = 'AAPL'

## hyperparameters..
batchsize=10
epochs=5
slots=60
LSTMNeurons=50
denseNeurons=25

## Read the data..
df = web.DataReader(stock, data_source=datasource, start=sDate, end=eDate, retry_count=retryCount)


plt.figure(figsize=(16,8))
plt.title('Price fluctuation')
plt.plot(df['Close'],color='skyblue', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()

## scale the data..
data = df.filter(['Close'])
dataset = data.values

train_data_len = math.ceil(len(dataset) * 0.8)

## Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(dataset)

train_data = scaler_data[0:train_data_len, :]


## Create x_train and y_train..
x_train = []
y_train = []
i=0
for i  in range(slots, len(train_data)):
    #print(i)
    x_train.append(train_data[i-slots:i,0])
    y_train.append(train_data[i,0])
      
## convert into numpy array..
x_train, y_train = np.array(x_train), np.array(y_train)


#reshape the array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))



def build_and_compile_model():
    ## Build the model
    model = Sequential()

    model.add(LSTM(LSTMNeurons, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(LSTMNeurons, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(denseNeurons))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    ## Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    ## Return the model
    return model


def prepare_test_data():
    ## create test dataset
    test_data = scaler_data[train_data_len - slots: ,:]

    x_test = []
    y_test = dataset[train_data_len: ,:]
    i=0
    for i  in range(slots, len(test_data)):
        #print(i)
        x_test.append(test_data[i - slots:i,0])
     
    ## convert to array and reshape..
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, y_test


def calculate_rmse(predictions, y_test):
    ## calculate RMSE
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    
    return rmse


def fit_model_and_make_prediction(model, x_test, b, e):
    ## fit the model
    t1 = time.time()
    model.fit(x_train, y_train, batch_size=b, epochs=e)
    t2 = time.time()
    total_time_taken = (t2-t1)
    
    ## Prediction..
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions, total_time_taken


def do_predictions(model, x_test):
    ## Prediction..
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions


def save_model(model, b, e):
    ## fit the model
    model.fit(x_train, y_train, batch_size=b, epochs=e)
    
    model_name = 'stockPicePrediction.sav'

    if os.path.exists(model_name):
        os.remove(model_name)
        
    pickle.dump(model, open(model_name, 'wb'))
    
    
list_rmse = []
ep = [20,25,30,35,40,45,50]
bs = [30,35,40,45,50,55,60]

#===========
model = build_and_compile_model()
save_model(model, 45, 20)
#============

def call_iterations():    
    for e in ep:
        for b in bs:
            ## get the model
            model = build_and_compile_model()
        
            ## prepare the test data..
            x_test, y_test = prepare_test_data()
        
            ## fit the model
            predictions, total_time_taken = fit_model_and_make_prediction(model, x_test, b, e)
        
            ## make predictions
            #predictions = do_predictions(fit_model, x_test)
        
            ## calculate rmse.
            rmse = calculate_rmse(predictions, y_test)
                
            r = []
            r.append(e)
            r.append(b)
            r.append(rmse)
            r.append(total_time_taken)
            list_rmse.append(r)

            

#call_iterations()