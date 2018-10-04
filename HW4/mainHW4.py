# -*- coding: utf-8 -*-
"""
@author: David Strickland
Course: CS5402 -Intro to Data Mining
Assignemnt: HW4
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pickle

def main():
    irisDS = pd.read_csv('iris.csv')
    features = list(irisDS.columns[:4])

    irisDSX = irisDS[features]
    irisDSY = irisDS['class']

    xTrain, xTest, yTrain, yTest = train_test_split(irisDSX, irisDSY, random_state=1)
    
    yTrainDf = yTrain.to_frame()
    yTestDf = yTest.to_frame()
    print(xTrain)
    yTrainDf['class'] = yTrainDf['class'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
    yTestDf['class'] = yTestDf['class'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
    
    for i in range(2,11):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xTrain,yTrain)
        #A = knn.kneighbors_graph(xTrain)
        results = knn.predict(xTest)
        accuracyCount = 0
        for j in range(len(results)):
            if results[j] == yTest.iloc[j]:
                accuracyCount = accuracyCount + 1
        accuracy = accuracyCount/len(results)
        print("Accuracy for k = %s is %s" % (i, accuracy))
#        kf = KFold(n_splits=5)
#        for irisTrain, irisTest in kf.split(irisDS):
#            print("%s %s" % (irisTrain,irisTest))
#            train(irisTrain, irisTest)
    
if __name__ == '__main__':
    main()