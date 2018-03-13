'''
Created on 13 Mar 2018

@author: pw61
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

trainingX = np.loadtxt('xTrain.txt')
trainingY = np.loadtxt('yTrain.txt')

trainingX = np.c_[np.ones_like(trainingX),trainingX]
linreg = LinearRegression()
linreg.fit(trainingX,trainingY)
y_hat = linreg.predict(trainingX)
'''Save into file'''
joblib.dump(linreg, "trainedLinearReg.pkl")
scores = cross_val_score(linreg,trainingX,trainingY, scoring = "neg_mean_squared_error",cv=10)
rmse = np.sqrt(-scores)

print("Scores", rmse)
print("Mean:", rmse.mean())
print("Standard deviation", rmse.std())

print('MSE = ', mean_squared_error(trainingY,y_hat))
