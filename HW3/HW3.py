# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:24:58 2018

@author: David Strickland
Course: CS5402 -Intro to Data Mining
Assignemnt: HW3
"""
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import graphviz
from subprocess import call

def calcPrecision(real, predict):
    truePositive = 0
    falsePositive = 0
    
    for i in range(len(real)):
        if predict[i] == real[i]:
            truePositive += 1
        elif predict[i] == 'Win':
            falsePositive += 1
       
    precision = truePositive / (truePositive + falsePositive)    
    return precision
    
def calcRecall(real, predict):
    truePositive = 0
    falseNegative = 0
    for i in range(len(real)):
        if predict[i] == real[i]:
            truePositive += 1
        elif predict[i] == 'Lose':
            falseNegative += 1
    recall = truePositive / (truePositive + falseNegative)
    return recall

fbNBTestDF = pd.read_csv('..\HW2\Test.5.csv')
fbNBTrainDF = pd.read_csv('..\HW2\Train.5.csv')

fbNBFeatures = list(fbNBTrainDF.columns[3:6])

#Drop columns that are only for human identification purposes
fbNBdataOnlyTrain = fbNBTrainDF.drop('ID', axis=1)
fbNBdataOnlyTest = fbNBTestDF.drop('ID', axis=1)
fbNBdataOnlyTrain = fbNBdataOnlyTrain.drop('Date', axis=1)
fbNBdataOnlyTest = fbNBdataOnlyTest.drop('Date', axis=1)
fbNBdataOnlyTrain = fbNBdataOnlyTrain.drop('Opponent', axis=1)
fbNBdataOnlyTest = fbNBdataOnlyTest.drop('Opponent', axis=1)

#Map non-continuous data to numerical data
fbNBdataOnlyTrain['Media'] = fbNBdataOnlyTrain['Media'].map({'1-NBC':1, '2-ESPN':2,'3-FOX':3, '4-ABC':4, '5-CBS':5})
fbNBdataOnlyTest['Media'] = fbNBdataOnlyTest['Media'].map({'1-NBC':1, '2-ESPN':2,'3-FOX':3, '4-ABC':4, '5-CBS':5})

fbNBdataOnlyTrain['Is_Home_or_Away'] = fbNBdataOnlyTrain['Is_Home_or_Away'].map({'Away':1, 'Home':2})
fbNBdataOnlyTest['Is_Home_or_Away'] = fbNBdataOnlyTest['Is_Home_or_Away'].map({'Away':1, 'Home':2})

fbNBdataOnlyTrain['Is_Opponent_in_AP25_Preseason'] = fbNBdataOnlyTrain['Is_Opponent_in_AP25_Preseason'].map({'In':1, 'Out':2})
fbNBdataOnlyTest['Is_Opponent_in_AP25_Preseason'] = fbNBdataOnlyTest['Is_Opponent_in_AP25_Preseason'].map({'In':1, 'Out':2})

fbNBX = fbNBdataOnlyTrain[fbNBFeatures]
fbNBY = fbNBdataOnlyTrain['Label']

#Multinomial NaiveBayes was chosen as it uses laplase smoothing as discussed in class rather than more complicated smoothing methods
#additionally it also does not assume a gaussian distribution or that the features are binary
fbNBmodel = MultinomialNB()
fbNBmodel.fit(fbNBX,fbNBY)

fbNBPredict = fbNBmodel.predict(fbNBdataOnlyTest[fbNBFeatures])
print(fbNBPredict)
#print(fbNBmodel)
print(accuracy_score(fbNBdataOnlyTest['Label'],fbNBPredict))
recall = calcRecall(fbNBdataOnlyTest['Label'], fbNBPredict)
precision = calcPrecision(fbNBdataOnlyTest['Label'], fbNBPredict) 
F1 = 2*((precision * recall)/(precision + recall))
print("Precision:  {} \nRecall:  {} \nF1:  {}".format(str(precision), str(recall), str(F1)))
#export_graphviz(fbNBmodel, out_file='fbNBtree.dot', feature_names=fbNBFeatures)
#call(["dot", "fbNBtree.dot","-Tpng", "-o", "fbNBtree.png"])
