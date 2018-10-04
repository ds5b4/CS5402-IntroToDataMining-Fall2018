# -*- coding: utf-8 -*-
"""
@author: David Strickland
Course: CS5402 -Intro to Data Mining
Assignemnt: HW4
"""

import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pickle

irisDS = pd.read_csv('irisTrain.csv')

def main():
    trainX = irisDS[irisDS.columns[1:5]]
    trainY = irisDS['class']    
    print(trainX)
    train(trainX, trainY)
    
def train(x, y):
    dfTrain = pd.DataFrame(columns=irisDS.columns)
    print(x)
    for i in x:
        print(i)
        
        #dfTrain = dfTrain.append(irisDS.iloc[[(i+1)]])
    
    dfTest = pd.DataFrame(columns=irisDS.columns)
    #for j in y:
        #print(iriDS.iloc[[j+1]])
        #dfTest = dfTest.append(irisDS.iloc[[j+1]])
        #print(dfTest)
    dfTestX = dfTest[irisDS.columns[1:5]]
    dfTestY = dfTest['class']

    KNeighborsClassifier
    
if __name__ == '__main__':
    main()