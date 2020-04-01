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
plt.style.use('fivethirtyeight')

class makePrediction():
    
    def __init__(self):
        self.today = dt.date.today()
        self.preDayDate = self.today - timedelta(1)

        self.sDate = '2010-01-01'
        self.eDate = self.preDayDate
        self.retryCount = 3
        self.datasource = 'yahoo'
        self.stock = 'AAPL'
        self.slots=60
        self.X_test = []
        self.model = None
        self.scaler = None

    def load_data(self):
        quote = web.DataReader(self.stock, data_source = self.datasource, start=self.sDate, end=self.eDate, retry_count=self.retryCount)
        new_df = quote.filter(['Close'])
        return new_df
    
        
    def scale_data(self, new_df):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        last_60_days = new_df[-self.slots:].values
        last_60_days_scaled = self.scaler.fit_transform(last_60_days)
        return last_60_days_scaled
        

    def get_test_data(self, last_60_days_scaled):
        self.X_test.append(last_60_days_scaled)                                       
        ## create array
        self.X_test = np.array(self.X_test)
        ## reshape
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        ##return self.X_test
        

    def load_model(self):
        ## Load the saved model..
        model_name = 'stockPicePrediction.sav'
        self.model = pickle.load(open(model_name, 'rb'))
               
                        
    def make_prediction(self):
        ## make predictions..
        predicted_price = self.model.predict(self.X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price)
                                       
        return predicted_price


