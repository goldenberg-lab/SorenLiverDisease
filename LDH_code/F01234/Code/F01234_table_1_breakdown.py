import os
import sys
import copy 
import pickle 
import numpy as np
import pandas as pd 
import pyodbc as db 
import matplotlib.pyplot as plt 

from sklearn.metrics import auc 
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import precision_recall_curve 

from beautifultable import BeautifulTable

pd.options.mode.chained_assignment = None  # default='warn'

thesis_path = '/Volumes/Chamber of Secrets/Thesis'
algdir = thesis_path + '/Code/Lancet Digital Health Code/F01234_model_files/03-29-2021 Extraction'
os.chdir(algdir)

datakey = 'TE'
features = [ 'Age', 'Sex', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 
            'Creatinine', 'INR', 'BMI', 'Platelets', 'Diabetes',  
            'Fibrosis', 'patientID', 'bx_date', 'reckey_enc', 'TE', 'TE_date']
feat_dict = {'age': 0, 'sex': 1, 'albumin': 2, 'alp': 3, 'alt': 4, 'ast': 5, 'bilirubin': 6, 
             'creatinine': 7, 'inr': 8, 'bmi': 9, 'platelets': 10, 'diabetes': 11}

cont_preds = ['Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'Creatinine', 'INR', 'Platelets', 'BMI']
cat_preds = ['Sex', 'Diabetes']
etio = [ 'Alcohol', 'Autoimmune Hepatitis', 'Cholestasis', 'DrugInducedLiverInjury',
 'Hepatitis B','Hepatitis C','MixedEnzymeAbnormality','NAFL', 'BiliaryCirrhosis',
 'SclerosingCholangitis', 'WilsonsDisease', 'Etiology',
 'Neoplasm', 'Other']

datapath = thesis_path + '/Data/Sept2020FinalDatasets/'
path_dict = {'Toronto': datapath + 'Toronto.xlsx',
              'Expert': datapath + 'Expert.xlsx',
              'McGill': datapath + 'McGill.xlsx',
              'NAFL': datapath + 'NAFLD.xlsx', 
              'TE': datapath + 'TE.xlsx', 
              'Combined': datapath + 'Combined.xlsx'}

dataset = pd.read_excel(path_dict[datakey], index_col=None, engine='openpyxl')
dataset['orig_fibrosis'] = dataset['Fibrosis']
dataset['Fibrosis'] = np.where(dataset['Fibrosis'] >= 3, 4, 0)

F012 = dataset.loc[dataset['orig_fibrosis'] <= 2]
F34 = dataset.loc[dataset['orig_fibrosis'] >= 3]

def print_info(datakey, df):
    print(datakey)
    print('N=%d' % len(df))
    for col in F012.columns.tolist():
        if col in cont_preds: 
            print('%s: %0.2f +/- %0.2f\n' % (col, df[col].mean(), df[col].std()))
        elif col in (cat_preds): 
            n_pos = pd.DataFrame(df[col].value_counts()).iloc[1]
            p_pos = 100*n_pos/len(df)
            print('%s: %d, %0.2f%%\n' % (col, n_pos, p_pos))
        elif col in (etio): 
            try:
                n_pos = pd.DataFrame(df[col].value_counts()).iloc[1]
                p_pos = 100*n_pos/len(df)
                print('%s: %d, %0.2f%%\n' % (col, n_pos, p_pos))
            except: 
                print('No positive cases of %s: \n' % (col))
                
print_info(datakey, F34)
