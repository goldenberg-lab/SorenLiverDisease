# Program plan 
# 1. Load and preprocess the approrpriate dataset (TLC, MUHC, Expert)
# 2. Create file to track all relevant metrics for each sample for each model (Only need it for ENS2 at appropriate params)
# 3. For each bootstrap sample, sample n% records from dataset, calculate performance metrics, and log in appropriate file 
# 4. Sort all performance metrics, calculate mean, and calculate confidence intervals by taking 2.5%th percentile and 97.5%th percentile 
import numpy as np
np.set_printoptions(precision=10)
import pandas as pd
import pyodbc as db
from sklearn.externals import joblib 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_curve 
from sklearn.metrics import precision_recall_curve 
from sklearn.utils import resample 
from sklearn.metrics import auc 
from numpy import trapz
from beautifultable import BeautifulTable
from helper import *
import time as tm

def my_confusion_matrix(truth, pred):
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    
    for i in range(0,len(truth)):
        if(pred[i] == 0 and truth[i] == 0):
            tn += 1
        elif(pred[i] == 4 and truth[i] == 4):
            tp += 1
        elif(pred[i] == 4 and truth[i] == 0):
            fp += 1
        elif(pred[i] == 0 and truth[i] == 4):
            fn += 1
    
    try:        
        tot = tp + fp + tn + fn
    except ZeroDivisionError:
        tot = np.nan
    
    try:
        sens = tp/(tp + fn)
    except ZeroDivisionError:
        sens = np.nan

    try:
        spec = tn/(tn + fp)
    except ZeroDivisionError:
        spec = np.nan

    try:
        ppv = tp/(tp + fp)
    except ZeroDivisionError:
        ppv = np.nan

    try:    
        npv = tn/(tn + fn)
    except ZeroDivisionError:
        npv = np.nan
    
    try:
        acc = (tp + tn)/(tot)    
    except ZeroDivisionError:
        acc = np.nan
        
    return sens, spec, ppv, npv, acc   
    
def get_sampled_data(data, size):
    dataset = resample(data,n_samples=size)
    Y_test = data['Fibrosis_True']
    return dataset[['Fibrosis_MB', 'Fibrosis_KP', 'Fibrosis_GS', 'Fibrosis_RK', 'Fibrosis_HK']], Y_test

path = "C:/Users/Soren/Desktop/Thesis/Data Analysis/Hold-out Test Sets/final_2000-2014no(hb,wbc)/bootstrap_results/"
conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Soren\Desktop\Thesis\Fibrosis.accdb;')
cnxn = db.connect(conn_str)
cursor = cnxn.cursor()

sql = "SELECT Fibrosis_MB, Fibrosis_KP, Fibrosis_GS, Fibrosis_RK, Fibrosis_HK, Fibrosis_True FROM _ExpertPredsCombined WHERE bx_date IS NOT NULL"
data = pd.read_sql(sql, cnxn)

n_iterations = 1000
n_size = int(len(data)*1)

EXP_maj = np.empty([8,n_iterations])
EXP_avg = np.empty([8,n_iterations])

for i in range(0,n_iterations):
    dataset, Y_test = get_sampled_data(data, n_size)
    dataset['maj_pred'] = np.where(dataset.sum(axis=1) > 10, 4, 0)
    maj_sens, maj_spec, maj_ppv, maj_npv, maj_acc = my_confusion_matrix(Y_test, dataset['maj_pred'].tolist())    
    EXP_maj[:,i] = 100*np.transpose(np.array([maj_sens, maj_spec, maj_ppv, maj_npv, maj_acc, np.nan, np.nan, 1]))
    
    # Method 2. Calculate performance for each individual doctor, then average that metric and add that as the performance for that iteration 
    MB_sens, MB_spec, MB_ppv, MB_npv, MB_acc = my_confusion_matrix(Y_test, dataset['Fibrosis_MB'].tolist())
    KP_sens, KP_spec, KP_ppv, KP_npv, KP_acc = my_confusion_matrix(Y_test, dataset['Fibrosis_KP'].tolist())
    GS_sens, GS_spec, GS_ppv, GS_npv, GS_acc = my_confusion_matrix(Y_test, dataset['Fibrosis_GS'].tolist())
    RK_sens, RK_spec, RK_ppv, RK_npv, RK_acc = my_confusion_matrix(Y_test, dataset['Fibrosis_RK'].tolist())
    HK_sens, HK_spec, HK_ppv, HK_npv, HK_acc = my_confusion_matrix(Y_test, dataset['Fibrosis_HK'].tolist())

    sens = np.mean([MB_sens, KP_sens, GS_sens, RK_sens, HK_sens])
    spec = np.mean([MB_spec, KP_spec, GS_spec, RK_spec, HK_spec])
    ppv = np.mean([MB_ppv, KP_ppv, GS_ppv, RK_ppv, HK_ppv])
    npv = np.mean([MB_npv, KP_npv, GS_npv, RK_npv, HK_npv])
    acc = np.mean([MB_acc, KP_acc, GS_acc, RK_acc, HK_acc])
    
    EXP_avg[:,i] = 100*np.transpose(np.array([sens, spec, ppv, npv, acc, np.nan, np.nan, 1]))
    print('Completed iteration # ' + str(i))

EXP_maj_df = pd.DataFrame.from_records(EXP_maj)
EXP_maj_df = EXP_maj_df.rename(index={0: 'sens',1: 'spec',2: 'ppv',3: 'npv',4: 'acc',5: 'AUROC',6: 'AUPRC',7: 'det'})
EXP_maj_df.to_excel(path + 'EXP_majority_vote.xlsx')

EXP_avg_df = pd.DataFrame.from_records(EXP_avg)
EXP_avg_df = EXP_avg_df.rename(index={0: 'sens',1: 'spec',2: 'ppv',3: 'npv',4: 'acc',5: 'AUROC',6: 'AUPRC',7: 'det'})
EXP_avg_df.to_excel(path + 'EXP_averaged.xlsx')


