# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:33:41 2018

@author: David Strickland
Course: CS5402 -Intro to Data Mining
Assignemnt: HW1 
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, math, operator as op


def main(): 
    TrainDataSet = pd.read_csv('../train.csv')
    testDataSet = pd.read_csv('../test.csv')
    mergeDataSets = [TrainDataSet, testDataSet]
    completeDataSet = pd.concat(mergeDataSets, ignore_index=True, sort=False)
    
    report = "" #what will be written to the output file
    numFeature = [] #filled in Q3 used in Q7
    
    #Only printing the column names for the trainDataSet since they are the same for both
    #additionally they are 
    Question1 = "Q1: The features of the dataset are: " 
    for columns in completeDataSet.columns.values:
        Question1 += columns + ", "
    Question1 = Question1[:-2] +".\n\n" #Replace the , with a .
    report += Question1
    
    #since no categories were set up previously we need to do this manually
    Question2 = "Q2: The following features are categorical: Survived, Pclass, Sex, Cabin, Embarked.\n\n"
    catFeatures = ['Survived', 'Pclass', 'Sex', 'Cabin', 'Embarked']
    report += Question2    
    
    Question3 = "Q3: The follwing features are numerical: "
    iterator = 0
    for dataTypes in completeDataSet.dtypes:
        if dataTypes == 'int64' or dataTypes == 'float64':
            Question3 += completeDataSet.columns.values[iterator] + ", "
            numFeature.append(completeDataSet.columns.values[iterator])
        iterator += 1
    Question3 = Question3[:-2] + """.\n These fields are numeric because they all have a datatype of int or float.\n\n"""
    report += Question3
    
    
    Question4 = "Q4: None of the features are mixed data type, as even the Ticket is just a string.\n\n"
    report += Question4
    
    Question5 = "Q5: The following features contain blank, null or empyt values: "
    for column in completeDataSet.columns.values:
        for index, row in completeDataSet.iterrows():
            if isinstance(row[column], int) or isinstance(row[column], float):
                if math.isnan(row[column]):
                    Question5 += column + ", "
                    break
            elif isinstance(row[column], str):
                if row[column] == '':
                    Question5 += column + ", "
                    break
    Question5 = Question5[:-2] + ".\n\n"                
    report += Question5
    
    
    Question6 = "Q6: The data types of each feature are as follows: \n"
    for column in completeDataSet.columns.values:
        if issubclass(completeDataSet[column].dtype.type, np.integer):
            Question6 += column + ":\t\tinteger\n"
            
        elif isinstance(completeDataSet[column][0],float):
            Question6 += column + ":\t\tfloat\n"
        
        elif isinstance(completeDataSet[column][0], str):
            Question6 += column + ":\t\tstring\n"
          
        elif isinstance(completeDataSet[column][0], bool):
            Question6 += column + ":\t\tbool\n"
    
    report += Question6 + '\n'
    
    #unclear what count meant and am assuming it is just sum.
    Question7 = "Q7:\n\t\tstd\t\tmean\t\tcount\t\tmax\t\tmin\t\t25%\t\t50%\t\t75%\n"
    for column in numFeature:
        Question7 += column + "\t" + str(completeDataSet[column].std()) + "\t\t" + str(completeDataSet[column].mean()) + "\t\t" + str(completeDataSet[column].sum()) + "\t\t" + str(completeDataSet[column].max()) + "\t\t" + str(completeDataSet[column].min()) + "\t\t" + str(completeDataSet[column].quantile(.25)) + "\t\t" + str(completeDataSet[column].quantile(.5)) + "\t\t" + str(completeDataSet[column].quantile(.75)) + "\n"
    
    report += Question7 + "\n"
    
    
    Question8 = "Q8:\n\t\tcount\t\tunique\t\ttop\t\tfreq\n"
    frequency = {}
    count = 0
    for column in catFeatures:
        for index, row in completeDataSet.iterrows():
            if row[column] in frequency:
                frequency[row[column]] += 1 #increment the frequency
            else:
                frequency[row[column]] = 1 #found one so it is intialized to 1
            count += 1
        Question8 += column + "\t" + str(count) + "\t" + str(len(frequency)) + "\t" + str(max(frequency.keys(), key=(lambda k: frequency[k]))) + "\t" + str(frequency[max(frequency.keys(), key=(lambda k: frequency[k]))]) + "\n"
        frequency.clear()
    report += Question8 + "\n"
    
    
    Question9 = "Q9: The correlation between Pclass=1 and Survived is " + str((completeDataSet.loc[completeDataSet['Pclass'] != 1]).corr(method='pearson')['Survived']['Pclass']) + " for those that were not Pclass 1 and " + str((completeDataSet.loc[completeDataSet['Pclass'] != 1]).corr(method='pearson')['Survived']['Pclass']) + " with Pclass = 1. This shows that Pclass is a weak indicator of survival. However it is the best indicator of the data given so it should be included\n\n"
    report += Question9
    
    Question10 = "Q10: Women were "
    femaleDataFrame = (TrainDataSet.loc[TrainDataSet['Sex'] == 'female'])
    maleDataFrame = (TrainDataSet.loc[TrainDataSet['Sex'] == 'male'])
    fSurvivalRate = (femaleDataFrame.loc[femaleDataFrame['Survived'] == 1].count()/femaleDataFrame.count())['Sex']
    mSurvivalRate = (maleDataFrame.loc[maleDataFrame['Survived'] == 1].count()/maleDataFrame.count())['Sex']
    Question10 += "significanly more likely to survive than men as they had a survival rate of " + str(fSurvivalRate) + " where as me only had a survival rate of " + str(mSurvivalRate) + ".\n\n"
    report += Question10
    
    Question11 = "Q11:/n4 year olds and younger had a very good survival rate with more than half surviving the ship sinking.\n Older passengers also had a very high survival rate with seemingly no one 80 or older dying.\nHowever this does not apply to 15-25 year olds who have a greater chance of dying than surviving the crash.\n\n"
    bins = np.linspace(0, 100, num=25, dtype=int)
    
    plt.clf()
    TrainDataSet.hist(column='Age', by='Survived', bins=bins)
    plt.title("Survived")
    plt.xlabel("Age")
    plt.xlim([0,100])
    plt.ylabel("Frequency")
    
    locs, labels = plt.xticks()
    plt.setp(labels,rotation=90)
    plt.show()
    report += Question11
    
   
    
    Question12 = "Q12:\nPclass=3 does have teh most passengers but a vast majority died as seen in the graph.\nInfant in Pclass=2 and 3 largely survived with Pclass=2 having the fewest infant deaths and Pclass=3 being almost 50/50.\nMost passengers in Pclass=1 did survive.\nThe age distibution of Pclass seems to be that Pclass=1 has teh most even spread of ages with most being around 25-50. Pclass=2 is slightly younger, being mostly 15-35 year olds. Pclass=3 is by far the younges with a similar age range to Pclass=2 but without many older passengers and many more younger passengers making the average age close to 25.\nYes Pclass seems to factor heavily into the survival rate of the passengers.\n\n"
    plt.clf()
    TrainDataSet.hist(column='Age', by=['Pclass', 'Survived'], bins=bins, sharex=True, sharey=True)
    plt.setp(labels,rotation=90)
    plt.show()
    report += Question12
    
    Question13 = ""
    plt.clf()
    #TestDataSet = TrainDataSet[['Sex', 'Fare', 'Survived','Embarked']].groupby('Sex').agg('mean')
    TrainDataSet.hist(column='Sex', by=['Survived', 'Embarked'], sharex=True)
    #TestDataSet.hist(column='Sex', by=['Survived','Embarked'])
    plt.setp(labels,rotation=0)
    plt.show()
    
    Question14 = "Q14: Duplicate Tickets"
    
    
    
     #print(report)
    with open('HW1Report.txt', 'w') as f:
        f.write(report)
    
if __name__ == '__main__':
    main()