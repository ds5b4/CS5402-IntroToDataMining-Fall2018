# -*- coding: utf-8 -*-
"""
@author: David Strickland
Course: CS5402 - Intro to Data Mining
Assignemnt: HW6
"""

from surprise import KNNBasic, NMF, SVD, Dataset, Reader, evaluate, print_perf
import time
import os

#Questions 1-9
#suppress output depending on which question is desired

threeFolds = False
Q14 = False
Q15 = True

#load data3Folds from a file 
file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data3Folds = Dataset.load_from_file(file_path, reader=reader)

data3Folds.split(n_folds=3)

#
# 3-Folds Comparison
#
if threeFolds == True:
    print('SVD')
    algoSVD = SVD()
    start_time = time.time()
    perfSVD = evaluate(algoSVD,data3Folds,measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfSVD)
    print(end_time - start_time, '\n\n')
    
    
    
    #PMF
    algoPMF = SVD(biased=False)
    start_time = time.time()
    perfPMF = evaluate(algoPMF,data3Folds,measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfPMF)
    print(end_time - start_time, '\n\n')
    
    print('NMF')
    algoNMF = NMF()
    start_time = time.time()
    perfNMF = evaluate(algoNMF,data3Folds,measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfNMF)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasic = KNNBasic(sim_options={'user_based':True})
    start_time = time.time()
    perfKNNBasic = evaluate(algoKNNBasic, data3Folds, measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfKNNBasic)
    print(end_time - start_time, '\n\n')
    
    algoItem = KNNBasic(sim_options={'user_based':False})
    start_time = time.time()
    perfItem = evaluate(algoItem, data3Folds, measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfItem)
    print(end_time - start_time, '\n\n')

#Used to decrease run time 
if Q14 == True:
    algoKNNBasicM = KNNBasic(sim_options={'name':'MSD','user_based':True})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasicC = KNNBasic(sim_options={'name':'cosine','user_based':True})
    start_time = time.time()
    perfKNNBasicC = evaluate(algoKNNBasicC, data3Folds, measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfKNNBasicC)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasicP = KNNBasic(sim_options={'name':'pearson','user_based':True})
    start_time = time.time()
    perfKNNBasicP = evaluate(algoKNNBasicP, data3Folds, measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfKNNBasicP)
    print(end_time - start_time, '\n\n')
    
    algoItemM = KNNBasic(sim_options={'name':'MSD','user_based':False})
    start_time = time.time()
    perfItemM = evaluate(algoItemM, data3Folds, measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfItemM)
    print(end_time - start_time, '\n\n')
    
    algoItemC = KNNBasic(sim_options={'name':'cosine','user_based':False})
    start_time = time.time()
    perfItemC = evaluate(algoItemC, data3Folds, measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfItemC)
    print(end_time - start_time, '\n\n')
    
    algoItemP = KNNBasic(sim_options={'name':'pearson','user_based':False})
    start_time = time.time()
    perfItemP = evaluate(algoItemP, data3Folds, measures=['RMSE','MAE'])
    end_time = time.time()
    print_perf(perfItemP)
    print(end_time - start_time, '\n\n')
    

    
if Q15 == True:
    algoKNNBasicM = KNNBasic(k=1,sim_options={'user_based':True})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasicM = KNNBasic(k=2,sim_options={'user_based':True})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasicM = KNNBasic(k=3,sim_options={'user_based':True})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasicM = KNNBasic(k=4,sim_options={'user_based':True})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasicM = KNNBasic(k=5,sim_options={'user_based':True})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')
    
    print('\n------------------------------------------------------------\n------------------------------------------------------------\n----------------------------------------------------------\n\n\n')
    
    algoKNNBasicM = KNNBasic(k=1,sim_options={'user_based':False})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasicM = KNNBasic(k=2,sim_options={'user_based':False})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasicM = KNNBasic(k=3,sim_options={'user_based':False})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasicM = KNNBasic(k=4,sim_options={'user_based':False})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')
    
    algoKNNBasicM = KNNBasic(k=5,sim_options={'user_based':False})
    start_time = time.time()
    perfKNNBasicM = evaluate(algoKNNBasicM, data3Folds, measures=['RMSE'])
    end_time = time.time()
    print_perf(perfKNNBasicM)
    print(end_time - start_time, '\n\n')    
