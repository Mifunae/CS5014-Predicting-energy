'''
Created on 13 Mar 2018

@author: pw61
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import argparse

parser = argparse.ArgumentParser(description='Training.')

parser.add_argument('-ln', action='store_true',
                    help='Use the use linear regression model')

parser.add_argument('-rid', action='store_true',
                    help='Use the use linear regression model')

parser.add_argument('-poly', action='store_true',
                    help='Use the use linear regression model')

parser.add_argument('-sgd', action='store_true',
                    help='Use the use linear regression model')

args = parser.parse_args()



''' ################### Training #####################'''
trainingX = np.loadtxt('xTrain.txt')
trainingY = np.loadtxt('yTrain.txt')

if args.ln is True:
    print ("Use linear regression model")
    regModel = LinearRegression()
    
elif args.rid is True:
    print ("Use Ridge regression model")
    fullPipeLine = Pipeline([
        ("polynomial",PolynomialFeatures(degree=2,include_bias=False)),
        ("stf_scale",StandardScaler())
    ])

    trainingX  = fullPipeLine.fit_transform(trainingX)
    regModel = Ridge()
    
elif args.poly is True:#
    print ("Use Polynomial regression model")
    fullPipeLine = Pipeline([
        ("polynomial",PolynomialFeatures(degree=2,include_bias=False)),
        ("stf_scale",StandardScaler())
    ])
    
    trainingX  = fullPipeLine.fit_transform(trainingX)
    regModel = LinearRegression()
elif args.sgd is True:
    print ("Use Stochastic Gradient Descent model")
    fullPipeLine = Pipeline([
        ("stf_scale",StandardScaler())
    ])
    trainingX  = fullPipeLine.fit_transform(trainingX)
    regModel = SGDRegressor(penalty="l2")
else:
    print "Please select the regression model"
    quit()

regModel.fit(trainingX,trainingY)

y_hat = regModel.predict(trainingX)
'''Save into file'''
joblib.dump(regModel, "trainedReg.pkl")
scores = cross_val_score(regModel,trainingX,trainingY, scoring = "neg_mean_squared_error",cv=10)
rmse = np.sqrt(-scores)

print "Training result"
print("Scores", rmse)
print("RMSE:", rmse.mean())
print("Standard deviation", rmse.std())

#trainingX = np.c_[np.ones_like(trainingX),trainingX]
print('MSE = ', mean_squared_error(trainingY,y_hat))

''' ###################### Testing ##################'''
testingX = np.loadtxt('xTest.txt')
testingY = np.loadtxt('yTest.txt')

#testingX = np.c_[np.ones_like(testingX),testingX]
if args.ln is False:
    testingX  = fullPipeLine.fit_transform(testingX)
    
y_hat = regModel.predict(testingX)

mse = mean_squared_error(testingY, y_hat)
rmse = np.sqrt(mse)
print "\nTesting result"
print("RMSE:", rmse.mean())
print('MSE = ', mse)
