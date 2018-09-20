# -*- coding: utf-8 -*-
"""
@author: David Strickland
Course: CS5402 -Intro to Data Mining
Assignemnt: HW1 
"""

import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from subprocess import call
import graphviz
import numpy as np

from id3 import Id3Estimator
from id3 import export_graphviz

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

#
#Task 4
#Question 1
#
footballTestDF = pd.read_csv('fbTest.csv')
footballTrainDF = pd.read_csv('fbTrain.csv')

fbFeatures = list(footballTrainDF.columns[2:5])

#Drop columns that are only for human identification purposes
dataOnlyTrain = footballTrainDF.drop('University', axis=1)
dataOnlyTest = footballTestDF.drop('University', axis=1)
dataOnlyTrain = dataOnlyTrain.drop('Date', axis=1)
dataOnlyTest = dataOnlyTest.drop('Date', axis=1)

#Map non-continuous data to numerical data
dataOnlyTrain['Media'] = dataOnlyTrain['Media'].map({'1-NBC':1, '2-ESPN':2,'3-FOX':3, '4-ABC':4})
dataOnlyTest['Media'] = dataOnlyTest['Media'].map({'1-NBC':1, '2-ESPN':2,'3-FOX':3, '4-ABC':4})

dataOnlyTrain['Is Home/Away?'] = dataOnlyTrain['Is Home/Away?'].map({'Away':1, 'Home':2})
dataOnlyTest['Is Home/Away?'] = dataOnlyTest['Is Home/Away?'].map({'Away':1, 'Home':2})

dataOnlyTrain['Is Opponent in AP Top 25 at Preseason?'] = dataOnlyTrain['Is Opponent in AP Top 25 at Preseason?'].map({'In':1, 'Out':2})
dataOnlyTest['Is Opponent in AP Top 25 at Preseason?'] = dataOnlyTest['Is Opponent in AP Top 25 at Preseason?'].map({'In':1, 'Out':2})

fbX = dataOnlyTrain[fbFeatures]
fbY = dataOnlyTrain['Label: Win/Lose']

xTrain, xTest, yTrain, yTest = train_test_split(fbX, fbY, random_state=1)

#
#CART using scikit-learn
#

model = tree.DecisionTreeClassifier()
model.fit(fbX,fbY)

#print(xTest)
#print(yTest)

yPredict = model.predict(dataOnlyTest)
#print(accuracy_score(yTest,yPredict))

tree.export_graphviz(model, out_file='footballCARTtree.dot', feature_names=fbFeatures)
call(["dot", "footballCARTtree.dot","-Tpng", "-o", "footballCARTtree.png"])

#
#ID3
#

estimator = Id3Estimator()
estimator.fit(fbX, fbY)
export_graphviz(estimator.tree_, 'footballID3tree.dot', fbFeatures)
call(["dot", "footballID3tree.dot","-Tpng", "-o", "footballID3tree.png"])

#
#C4.5
#

"""
Cannot find a reliable source that implemented C4.5.
"""


#
#Task 4
#Question 2
#

#
#
weatherTestDF = pd.read_csv('weatherTest.csv')
weatherTrainDF = pd.read_csv('weatherTrain.csv')



weatherFeatures = list(weatherTrainDF.columns[2:6])

#Drop columns that are only for human identification purposes
weatherDataOnlyTrain = weatherTrainDF.drop('ID', axis=1)
weatherDataOnlyTest = weatherTestDF.drop('ID', axis=1)
weatherDataOnlyTrain = weatherDataOnlyTrain.drop('Date', axis=1)
weatherDataOnlyTest = weatherDataOnlyTest.drop('Date', axis=1)

#Map non-binary non-continuous data to numerical data with OneHotEncoder library

weatherDataOnlyTrain['Outlook'] = weatherDataOnlyTrain['Outlook'].map({'Sunny':1, 'Overcast':2,'Rainy':3})
weatherDataOnlyTest['Outlook'] = weatherDataOnlyTest['Outlook'].map({'Sunny':1, 'Overcast':2,'Rainy':3})

weatherDataOnlyTrain['Temperature'] = weatherDataOnlyTrain['Temperature'].map({'Hot':1, 'Mild':2, 'Cool':3})
weatherDataOnlyTest['Temperature'] = weatherDataOnlyTest['Temperature'].map({'Hot':1, 'Mild':2, 'Cool':3})

weatherDataOnlyTrain['Humidity'] = weatherDataOnlyTrain['Humidity'].map({'High':1, 'Normal':2})
weatherDataOnlyTest['Humidity'] = weatherDataOnlyTest['Humidity'].map({'High':1, 'Normal':2})

#since Windy is boolean already, it does not need to be mapped
#weatherDataOnlyTrain['Windy'] = weatherDataOnlyTrain['Windy'].map({'True':1, 'False':2})
#weatherDataOnlyTest['Windy'] = weatherDataOnlyTest['Windy'].map({'True':1, 'False':2})

weatherX = weatherDataOnlyTrain[weatherFeatures]
weatherY = weatherDataOnlyTrain['Label: Play?']

weatherEnc = preprocessing.OneHotEncoder()
weatherEnc.fit(weatherX)
weatherLabels = weatherEnc.transform(weatherX).toarray()
weatherLabels.shape
#print(weatherLabels)

#
#CART using scikit-learn
#

model = tree.DecisionTreeClassifier()
model.fit(weatherX, weatherY)

#print(xTest)
#print(yTest)

yPredict = model.predict(weatherDataOnlyTest)
print(yPredict)
#print(accuracy_score(yTest,yPredict))

tree.export_graphviz(model, out_file='weatherCARTTree.dot', feature_names= weatherFeatures)
call(["dot", "weatherCARTTree.dot","-Tpng", "-o", "weatherCARTTree.png"])

#
#ID3
#

estimator = Id3Estimator()
estimator.fit(weatherX, weatherY)
export_graphviz(estimator.tree_, 'weatherID3tree.dot', weatherFeatures)
call(["dot", "weatherID3tree.dot","-Tpng", "-o", "weatherID3tree.png"])






#
# Task 5
# Question 1
#

football2TestDF = pd.read_csv('Test.5.csv')
football2TrainDF = pd.read_csv('Train.5.csv')

fb2Features = list(football2TrainDF.columns[3:6])

#Drop columns that are only for human identification purposes
fb2dataOnlyTrain = football2TrainDF.drop('ID', axis=1)
fb2dataOnlyTest = football2TestDF.drop('ID', axis=1)
fb2dataOnlyTrain = fb2dataOnlyTrain.drop('Date', axis=1)
fb2dataOnlyTest = fb2dataOnlyTest.drop('Date', axis=1)
fb2dataOnlyTrain = fb2dataOnlyTrain.drop('Opponent', axis=1)
fb2dataOnlyTest = fb2dataOnlyTest.drop('Opponent', axis=1)

#Map non-continuous data to numerical data
fb2dataOnlyTrain['Media'] = fb2dataOnlyTrain['Media'].map({'1-NBC':1, '2-ESPN':2,'3-FOX':3, '4-ABC':4, '5-CBS':5})
fb2dataOnlyTest['Media'] = fb2dataOnlyTest['Media'].map({'1-NBC':1, '2-ESPN':2,'3-FOX':3, '4-ABC':4, '5-CBS':5})

fb2dataOnlyTrain['Is_Home_or_Away'] = fb2dataOnlyTrain['Is_Home_or_Away'].map({'Away':1, 'Home':2})
fb2dataOnlyTest['Is_Home_or_Away'] = fb2dataOnlyTest['Is_Home_or_Away'].map({'Away':1, 'Home':2})

fb2dataOnlyTrain['Is_Opponent_in_AP25_Preseason'] = fb2dataOnlyTrain['Is_Opponent_in_AP25_Preseason'].map({'In':1, 'Out':2})
fb2dataOnlyTest['Is_Opponent_in_AP25_Preseason'] = fb2dataOnlyTest['Is_Opponent_in_AP25_Preseason'].map({'In':1, 'Out':2})

fb2X = fb2dataOnlyTrain[fb2Features]
fb2Y = fb2dataOnlyTrain['Label']
#print(fb2X)
#print(fb2dataOnlyTest)

xFB2Train, xFB2Test, yFB2Train, yFB2Test = train_test_split(fb2X, fb2Y, random_state=1)

#
#ID3 using decision-tree-ID3
#

estimator = Id3Estimator()
estimator.fit(fb2X, fb2Y)
id3Predict = estimator.predict(fb2dataOnlyTest[fb2Features])
print(accuracy_score(fb2dataOnlyTest['Label'],id3Predict))
print(id3Predict)
export_graphviz(estimator.tree_, out_file='fb2ID3Tree.dot', feature_names= fb2Features)
call(["dot", "fb2ID3Tree.dot","-Tpng", "-o", "fb2ID3Tree.png"])

#calc recall, precision, and F1
recall = calcRecall(fb2dataOnlyTest['Label'], id3Predict)
precision = calcPrecision(fb2dataOnlyTest['Label'], id3Predict)
F1 = 2 * ((recall * precision)/(precision + recall))

print(str(recall) + "  " + str(precision) + "  " + str(F1))

#
#CART but with entropy so kinda ID3 (scikit-learn)
#Using because it allows the use of accuracy score
#

fb2model = tree.DecisionTreeClassifier(criterion='entropy')
fb2model.fit(fb2X,fb2Y)

fb2Predict = fb2model.predict(fb2dataOnlyTest[fb2Features])
#print(fb2Predict)

#print(accuracy_score(fb2dataOnlyTest['Label'],fb2Predict))
tree.export_graphviz(fb2model, out_file='football2CARTtree.dot', feature_names=fb2Features)
call(["dot", "football2CARTtree.dot","-Tpng", "-o", "football2CARTtree.png"])

