import pandas as pd
import numpy as np
import seaborn as sns
import keras
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import time

#reference: https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5

class NeuralNetworkClassifier:
    def classify(self):
            first = time.time() #assign first to be the start time
            
            dataset = pd.read_csv('african_crises.csv') #read the dataset

            #data_x is the attributes that the dataset has
            #data_y is the attributes that whether the country has crisis or not
            data_x = dataset[['case', 'year', 'systemic_crisis', 'exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence', 'currency_crises', 'inflation_crises']]
            data_y = dataset['banking_crisis']
            data_y = keras.utils.to_categorical(data_y, num_classes=None, dtype='float32')

            #preparing data for training 
            X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.8, random_state=0)

            #training the algorithm
            #creating model sequentially and the output of each layer we add is input to the next layer we specify
            model = Sequential()
            model.add(Dense(10, input_dim = 11, activation = 'relu'))
            model.add(Dense(5,activation='relu'))
            model.add(Dense(2, activation='softmax'))

            #specify the loss function and optimizer
            model.compile(loss='categorical_crossentropy', optimizer = 'adam',
                          metrics=['accuracy'])

            #training model
            history = model.fit(X_train, y_train, epochs = 10, batch_size = 64, verbose = 0)

            #check the accuracy
            y_pred = model.predict(X_test)

            pred = list()
            for i in range(len(y_pred)):
                pred.append(np.argmax(y_pred[i]))

                test = list()
            for i in range(len(y_test)):
                test.append(np.argmax(y_test[i]))

            a = accuracy_score(pred, test)
            print('time for nn', time.time() - first)
            return a



