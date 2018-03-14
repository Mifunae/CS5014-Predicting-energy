'''
Created on 14 Mar 2018

@author: pw61
'''
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

''' ################### Training #####################'''
trainingX = np.loadtxt('xTrain.txt')
trainingY = np.loadtxt('yTrain.txt')

# Apply Polynomial feature
fullPipeLine = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("stf_scale", StandardScaler())
    ])

trainingX = fullPipeLine.fit_transform(trainingX)
ridgeReg = Ridge(alpha=1, solver="cholesky")
ridgeReg.fit(trainingX, trainingY)
y_hat = ridgeReg.predict(trainingX)

scores = cross_val_score(ridgeReg, trainingX, trainingY, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)
print("Scores", rmse)
print("Mean:", rmse.mean())
print("Standard deviation", rmse.std())
print('MSE = ', mean_squared_error(trainingY, y_hat))

''' ###################### Testing ##################'''
testingX = np.loadtxt('xTest.txt')
testingY = np.loadtxt('yTest.txt')

testingX = fullPipeLine.fit_transform(testingX)

y_hat = ridgeReg.predict(testingX)

scores = cross_val_score(ridgeReg, testingX, testingY, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)
print "\nTesting result"
print("Scores", rmse)
print("Mean:", rmse.mean())
print("Standard deviation", rmse.std())
print('MSE = ', mean_squared_error(testingY, y_hat))
