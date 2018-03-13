'''
Created on 13 Mar 2018

@author: pw61
'''

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataFrameSelector import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

energy = pd.read_csv('energydata_complete.csv')

def calculateNsm(date):
    tomorrow = date+timedelta(1)
    midnight = datetime(year = tomorrow.year, month = tomorrow.month, 
                        day=tomorrow.day, hour=0, minute=0, second=0)
    return (midnight - date).seconds

nsmList = []
energy["nsm"] = ""
for index,row in energy.iterrows():
    dateStr = row["date"]
    dateObj = datetime.strptime(dateStr, "%Y-%m-%d %H:%M:%S")
    nsmList.append(calculateNsm(dateObj))

energy["nsm"] = nsmList
energy.drop(["rv1","rv2","date"],axis =1, inplace = True)

energy.hist(bins = 50,figsize = (20,15))
#plt.show()
#energy.info()
print energy.describe()

trainSet, testSet = train_test_split(energy,test_size=0.2, random_state=42)
xTrain = trainSet.drop("Appliances",axis =1)
yTrain = trainSet["Appliances"].copy()

fullPipeLine = Pipeline([
        ("selector",DataFrameSelector(list(xTrain))),
        ("stf_scale",StandardScaler())
    ])

xTrain  = fullPipeLine.fit_transform(xTrain)
np.savetxt("xTrain.txt", xTrain)
np.savetxt("yTrain.txt", yTrain)

