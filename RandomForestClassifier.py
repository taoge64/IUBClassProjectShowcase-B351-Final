import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class RandomForest_Classifier:
    def classify(self):
        first=time.time()#assign first to be the start time 
        
        dataset = pd.read_csv('african_crises.csv') #read the dataset 

        #data_x is the attributes that the dataset has
        #data_y is the attributes that whether the country has crisis or not
        data_x = dataset[['case', 'year', 'systemic_crisis', 'exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence', 'currency_crises', 'inflation_crises']]
        data_y = dataset['banking_crisis']


        #preparing data for training
        X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.8, random_state=0)


        #training the algorithm
        clf = RandomForestClassifier(n_estimators = 10, n_jobs= 2, random_state = 0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_predict = clf.predict_proba(X_test)

        #evaluating the algorithm
        a=accuracy_score(y_test,y_pred)
        print("TIme is ", time.time()-first)
        return a

