import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#https://mubaris.com/posts/kmeans-clustering/
#https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d
from matplotlib import pyplot as plt
data =pd.read_csv('african_crises.csv')
print(data.shape)
# this function gives the idea about what our dataset and stuff look like
def datavisualization():
    item=[]
    for i in range(len(data)):
        item.append(i)
    dataset=['year','systemic_crisis','exch_usd','domestic_debt_in_default','sovereign_external_debt_default','gdp_weighted_default','inflation_annual_cpi','independence','currency_crises','inflation_crises']
    fend=data['banking_crisis'].values
    plt.scatter(fend,item,s=2)
    plt.show()
if __name__ == "__main__":
    datavisualization()

    plt.figure(figsize = (9, 6))
    name = ['1NN','3NN','Cluster','Perceptron','RandomForest','NeuralNetwork']
    value = [86.0, 86.0, 48.5, 88.0, 97.4, 91.0]

    plt.subplot(1,1,1)
    plt.bar(name, value)
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')

    plt.show()
    plt.figure(figsize = (9, 6))
    name = ['1NN','3NN','Cluster','Perceptron','RandomForest','NeuralNetwork']
    value = [3.62, 3.57, 33.27, 0.66, 0.34, 0.86]

    plt.subplot(1,1,1)
    plt.bar(name, value)
    plt.xlabel('Algorithm')
    plt.ylabel('Time')

    plt.show()


