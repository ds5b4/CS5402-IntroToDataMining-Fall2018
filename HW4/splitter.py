# -*- coding: utf-8 -*-
"""
@author: David Strickland
Course: CS5402 -Intro to Data Mining
Assignemnt: HW4
"""

import pandas as pd
from sklearn.model_selection import train_test_split
#import sys

irisDS = pd.read_csv('iris.csv')
features = list(irisDS.columns[1:4])

irisDSX = irisDS[features]
irisDSY = irisDS['class']

xTrain, xTest, yTrain, yTest = train_test_split(irisDSX, irisDSY, random_state=1)

train = pd.concat([xTrain, yTrain], axis=1, join='inner')
test = pd.concat([xTest, yTest], axis=1, join='inner')

train.to_csv('irisTrain.csv')
test.to_csv('irisTest.csv')