# -*- coding: utf-8 -*-
"""
@author: David Strickland
Course: CS5402 -Intro to Data Mining
Assignemnt: HW1 
"""

import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from subprocess import call
import graphviz
import numpy as np

from id3 import Id3Estimator
from id3 import export_graphviz

#
#Task 4
#Question 1
#
footballTestDF = pd.read_csv('fbTest.csv')
footballTrainDF = pd.read_csv('fbTrain.csv')

features = list(footballTrainDF.columns[2:5])

dataOnlyTrain = footballTrainDF.drop('University', axis=1)
dataOnlyTest = footballTestDF.drop('University', axis=1)
dataOnlyTrain = dataOnlyTrain.drop('Date', axis=1)
dataOnlyTest = dataOnlyTest.drop('Date', axis=1)

dataOnlyTrain['Media'] = dataOnlyTrain['Media'].map({'1-NBC':1, '2-ESPN':2,'3-FOX':3, '4-ABC':4})
dataOnlyTest['Media'] = dataOnlyTest['Media'].map({'1-NBC':1, '2-ESPN':2,'3-FOX':3, '4-ABC':4})

dataOnlyTrain['Is Home/Away?'] = dataOnlyTrain['Is Home/Away?'].map({'Away':1, 'Home':2})
dataOnlyTest['Is Home/Away?'] = dataOnlyTest['Is Home/Away?'].map({'Away':1, 'Home':2})

dataOnlyTrain['Is Opponent in AP Top 25 at Preseason?'] = dataOnlyTrain['Is Opponent in AP Top 25 at Preseason?'].map({'In':1, 'Out':2})
dataOnlyTest['Is Opponent in AP Top 25 at Preseason?'] = dataOnlyTest['Is Opponent in AP Top 25 at Preseason?'].map({'In':1, 'Out':2})

print(dataOnlyTest)
print(dataOnlyTrain)

x = dataOnlyTrain[features]
y = dataOnlyTrain['Label: Win/Lose']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=1)

#
#CART using scikit-learn
#

model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(x,y)

print(xTest)
print(yTest)

yPredict = model.predict(dataOnlyTest)
print(yPredict)
#print(accuracy_score(yTest,yPredict))

tree.export_graphviz(model, out_file='tree.dot', feature_names=features)
call(["dot", "tree.dot","-Tpng", "-o", "tree.png"])

#
#ID3
#

estimator = Id3Estimator()
estimator.fit(x, y)
export_graphviz(estimator.tree_, 'ID3tree.dot', features)
call(["dot", "ID3tree.dot","-Tpng", "-o", "ID3tree.png"])