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

''' ################### Training #####################'''
trainingX = np.loadtxt('xTrain.txt')
trainingY = np.loadtxt('yTrain.txt')


#### Linear Reg
trainingX = np.c_[np.ones_like(trainingX),trainingX]
regModel = LinearRegression()

''' ####### Ridge
fullPipeLine = Pipeline([
        ("polynomial",PolynomialFeatures(degree=2,include_bias=False)),
        ("stf_scale",StandardScaler())
    ])

trainingX  = fullPipeLine.fit_transform(trainingX)
regModel = Ridge(alpha=1, solver = "cholesky")
'''

''' ####### Polynomial

fullPipeLine = Pipeline([
        ("polynomial",PolynomialFeatures(degree=2,include_bias=False)),
        ("stf_scale",StandardScaler())
    ])

trainingX  = fullPipeLine.fit_transform(trainingX)
regModel = LinearRegression()
'''


regModel.fit(trainingX,trainingY)

y_hat = regModel.predict(trainingX)
'''Save into file'''
joblib.dump(regModel, "trainedLinearReg.pkl")
scores = cross_val_score(regModel,trainingX,trainingY, scoring = "neg_mean_squared_error",cv=10)
rmse = np.sqrt(-scores)

print "Training result"
print("Scores", rmse)
print("Mean:", rmse.mean())
print("Standard deviation", rmse.std())

#trainingX = np.c_[np.ones_like(trainingX),trainingX]
print('MSE = ', mean_squared_error(trainingY,y_hat))

''' ###################### Testing ##################'''
testingX = np.loadtxt('xTest.txt')
testingY = np.loadtxt('yTest.txt')

testingX = np.c_[np.ones_like(testingX),testingX]
y_hat = regModel.predict(testingX)

scores = cross_val_score(regModel,testingX,testingY, scoring = "neg_mean_squared_error",cv=10)
rmse = np.sqrt(-scores)
print "\nTesting result"
print("Scores", rmse)
print("Mean:", rmse.mean())
print("Standard deviation", rmse.std())
print('MSE = ', mean_squared_error(testingY,y_hat))
