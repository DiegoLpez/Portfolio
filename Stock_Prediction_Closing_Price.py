#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout


# In[2]:


ts = TimeSeries(key='0IN175LIOUOBEICW',output_format = 'pandas')
# Get json object with the intraday data and another with  the call's metadata
df, meta_data = ts.get_daily_adjusted(symbol='MMM', outputsize='full')
df = df.rename(columns = {'1. open':'Open','2. high':'High', '3. low':'Low', 
                 '4. close':'Close', '5. adjusted close':'Price', '6. volume':'Volume',
                 '7. dividend amount':'Dividends', '8. split coefficient':'Stock Splits'})
df.dropna(inplace=True)
# top should be olders
df = df.sort_values(by='date')
df.head()


# In[3]:


#drop nulls 
import seaborn as sns
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
#ax=sns.lineplot(data=df, x='timestamp',y='close', color="blue");
plt.xlabel('timestamp',fontsize=18)
plt.ylabel('Close Price INR',fontsize=18)
plt.show()


# # Data Processing

# Convert to numpy array

# In[4]:


#get dates and closing prices
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset)*.8)
print(data.shape)


# Scaling data to 0 to 1

# In[5]:


sc = MinMaxScaler(feature_range=(0,1))
scaled_data = sc.fit_transform(dataset)
print(scaled_data.shape)


# Create x and y training data

# In[6]:


train_data = scaled_data[0:training_data_len  , : ]
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
print(type(x_train))
print(len(x_train))


# create x and y testing data

# In[7]:


test_data = scaled_data[training_data_len - 60: , : ]#Create the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


# Spliting for train and test

# In[8]:


#convert into arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape
#same for training
x_test = np.array(x_test)
print(x_train.shape)
print(y_train.shape)


# In[9]:


# Convert into 3 D
X_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

x_train.shape


# # Model

# Building the Model

# In[10]:


model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))


# Train with adam optimizer and mean squared error as loss function

# In[11]:


model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train,y_train,epochs=10,batch_size=32)


# # Prediction

# In[12]:


predictions = model.predict(x_test)
predictions = sc.inverse_transform(predictions)


# In[13]:


#### RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# Plotting Predictions

# In[14]:


train = data[:training_data_len]
display = data[training_data_len:]
display['Predictions'] = predictions #Visualize the data
plt.figure(figsize=(16,8))
plt.title('BOX')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.plot(train['Close'])
plt.plot(display['Close'])
plt.plot(display['Predictions'])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')
plt.show()


# In[15]:


display


# In[ ]:




