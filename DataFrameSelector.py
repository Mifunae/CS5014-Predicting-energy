'''
Created on 13 Mar 2018

@author: pw61
'''
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector():
    def __init__(self, attributeNames):
        self.attributeNames = attributeNames
    
    def fit(self, X, y=None):
        return self
    
    def transform(self,X):
        return X[self.attributeNames].values