# Step 1. (DONE) Load combined dataset 
# Step 2. (DONE) Impute all data based on training set distribution 
# Step 3. (DONE) Separate positives and negatives 


# Step 3. Impute and scale negatives based on training set distribution 
# Step 4. Oversample the negatives to get the right prevalence 
# Step 4. Add a fixed amount of Gaussian noise to each feature e.g. N(0,0.05)
# Step 5. Transform the dataset back to unscaled version using training set mean and standard deviations 
# Step 6. Sample positives and transformed negatives to assemble dataset 
# Step 7. Make predictions on the dataset 
# Step 8. Repeat process 100 times to get distribution of performance metrics 

# Substep 1a) Repeat with APRI only to see if it will work, then add other algorithms 

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
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'

class MLA():
    def __init__(self, name, threshold, probs, label): 
        self.name = name 
        self.preds = (probs >= threshold)*1
        self.cm = get_confusion_matrix(label, self.preds)
        self.pm = get_performance_metrics(self.cm)
        self.pm['per_det'] = 1
        
class BIO(): 
    def __init__(self, name, lower, upper, df):
        self.name = name
        self.probs = np.where(df[name + '_vals'] >= upper, 1, 0.5)
        self.probs = np.where(df[name + '_vals'] <= lower, 0, self.probs)
        
        df['indet'] = np.where(self.probs == 0.5, 1, 0)
        self.percent_det = 1 - df['indet'].sum()/len(df)
        df = df.loc[df['indet'] == 0].reset_index(drop=True)
        df['preds'] = np.where(df[name + '_vals'] >= upper, 1, 0) # This is the line that causes problems for TE

        self.cm = get_confusion_matrix(df['target'], df['preds'])
        self.pm = get_performance_metrics(self.cm)
        self.pm['per_det'] = self.percent_det
        
def get_confusion_matrix(label, preds):
    cm = pd.DataFrame({'pred': preds, 'label': label})
    cm['TN'] = np.where((cm['pred'] == 0) & (cm['label'] == 0), 1, 0)
    cm['FP'] = np.where((cm['pred'] == 1) & (cm['label'] == 0), 1, 0)
    cm['FN'] = np.where((cm['pred'] == 0) & (cm['label'] == 1), 1, 0)
    cm['TP'] = np.where((cm['pred'] == 1) & (cm['label'] == 1), 1, 0)
    cm['null'] = np.where(cm['pred'].isnull(), 1, 0)
    return cm.sum()

def get_performance_metrics(cm): 
    TP = cm['TP']
    TN = cm['TN']
    FP = cm['FP']
    FN = cm['FN']
    
    sens = TP/(TP + FN)
    spec = TN/(TN + FP)
    ppv = TP/(TP + FP)
    npv = TN/(TN + FN)
    
    pm = {'sens': sens, 'spec': spec, 'ppv': ppv, 
          'npv': npv}
    return pm 
    

res = []
noise = 0.05
num_trials = 1000
dk = 'Combined'

thesis_path = '/Volumes/Chamber of Secrets/Thesis'
algdir = '/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/models'
data = pd.read_excel(thesis_path + '/Data/Sept2020FinalDatasets/' + dk + '.xlsx', index_col=None, engine='openpyxl')
data['target'] = np.where(data['Fibrosis'] <= 2, 0, 1)
features = {'Age': 'age', 
                     'Sex': 'sex', 
                     'Albumin': 'albumin', 
                     'ALP': 'alp', 
                     'ALT': 'alt', 
                     'AST': 'ast', 
                     'Bilirubin': 'bilirubin', 
                     'Creatinine': 'creatinine', 
                     'INR': 'inr', 
                     'BMI': 'bmi', 
                     'Platelets': 'platelets',
                     'Diabetes': 'diabetes'}
lower_features = list(features.values())

num_features = ['albumin', 'alp', 'alt', 'ast', 'bilirubin', 'creatinine', 'inr', 'bmi', 'platelets']
data = data[list(features.keys()) + ['target']]
data.rename(columns=features, inplace=True)

with open(algdir + '/trained_models_imp_scl.pkl', 'rb') as file: 
    unpickler = pickle.Unpickler(file)
    outputs = unpickler.load() 
    scl_info = outputs['scl_info']
    SVM_model = outputs['svm_model']
    RFC_model = outputs['rfc_model']
    GBC_model = outputs['gbc_model']
    LOG_model = outputs['log_model']
    MLP_model = outputs['mlp_model']
    
# Impute based on training set distribution here
# Impute and scale based on training set distribution here
null_cols = pd.DataFrame(data.isnull().sum(axis=0)).rename(columns={0: 'null_count'})
null_cols = null_cols.loc[null_cols['null_count'] != 0]
for col in null_cols.index: 
    data[col] = np.where(data[col].isnull(), scl_info[col + '_mean']  , data[col])

data_pos = data.loc[data['target'] == 1]
data_neg = data.loc[data['target'] == 0]

trial_pos_tuple = [(t, np) for np in range(10,60,10) for t in range(0, num_trials)]

for t in tqdm(trial_pos_tuple):
    i = t[0]
    num_pos = t[1]

    num_neg = 10000 - num_pos
    prev = (num_pos/10000)    
    #print('%0.2f%% - %d' %(100*prev, i))
    
    neg = data_neg.sample(n=num_neg, replace=True, random_state=i)
    pos = data_pos.sample(n=num_pos, replace=True, random_state=i)
    
    neg_scl = neg.copy()
    for col in lower_features: 
        neg_scl[col] = (neg[col] - scl_info[col + '_mean'])/scl_info[col + '_std']

    neg_scl_noised = neg_scl.copy() 
    for col in num_features: 
        neg_scl_noised[col] += np.random.normal(0, noise, size=(len(neg_scl_noised)))
        
    neg_noised_unscl = neg_scl_noised.copy() 
    for col in lower_features: 
        neg_noised_unscl[col] = neg_scl_noised[col]*scl_info[col + '_std'] + scl_info[col + '_mean']

    ALL_df = pd.concat([pos, neg_noised_unscl])
    ALL_df_scl = ALL_df.copy() 
    for col in lower_features: 
        ALL_df_scl[col] = (ALL_df_scl[col] - scl_info[col + '_mean'])/scl_info[col + '_std']
    
    xti = ALL_df[lower_features].values
    xts = ALL_df_scl[lower_features].values 
    y_true = ALL_df['target']
    
    # Make predictions for each algorithm 
    # Get performance metrics for each algorithm 
    # Store performance metrics for each algorithm at each prevalence/experiment 
    # Save performance metrics to disk so that we 
    
    ALL_df['SVM_probs'] = SVM_model.predict_proba(xts)[:,1]    # SVM: Trained on Scaled Values
    ALL_df['RFC_probs'] = RFC_model.predict_proba(xti)[:,1]    # RFC: Trained on Unscaled Values
    ALL_df['GBC_probs'] = GBC_model.predict_proba(xti)[:,1]    # GBC: Trained on Unscaled Values
    ALL_df['LOG_probs'] = LOG_model.predict_proba(xts)[:,1]    # LOG: Trained on Scaled Values 
    ALL_df['MLP_probs'] = MLP_model.predict_proba(xts)[:,1]    # MLP: Trained on Scaled Values 
    ALL_df['ENS_probs'] = ALL_df[['SVM_probs', 'RFC_probs', 'GBC_probs', 'LOG_probs', 'MLP_probs']].mean(axis=1)
    ALL_df['APRI_vals'] = (100/35)*ALL_df['ast']/ALL_df['platelets']
    ALL_df['FIB4_vals'] = ALL_df['age']*ALL_df['ast']/(ALL_df['platelets']*(ALL_df['alt']**0.5))

    # 
    ENS_TE = MLA('ENS_TE(0.465)', 0.465, ALL_df['ENS_probs'], ALL_df['target'])
    ENS_EXP = MLA('ENS_EXP(0.6)', 0.6, ALL_df['ENS_probs'], ALL_df['target'])
    
    # APRI calculations 
    apri_det_df = ALL_df.loc[(ALL_df['APRI_vals'] <= 1) | (ALL_df['APRI_vals'] >= 2)]
    apri_indet_df = ALL_df.loc[(ALL_df['FIB4_vals'] > 1) | (ALL_df['FIB4_vals'] < 2)]
    APRI = BIO('APRI', 1, 2, ALL_df[['APRI_vals', 'target']])
    APRI.name = 'APRI(1,2)'
    ENS_APRI_det = MLA('ENS_APRI_det(0.736)', 0.736, apri_det_df['ENS_probs'], apri_det_df['target'])
    ENS_APRI_indet = MLA('ENS_APRI_indet(0.736)', 0.736, apri_indet_df['ENS_probs'], apri_indet_df['target'])
    ENS_APRI_ALL = MLA('ENS_APRI_ALL(0.736)', 0.736, ALL_df['ENS_probs'], ALL_df['target'])

    # FIB4 calculations 
    fib4_det_df = ALL_df.loc[(ALL_df['FIB4_vals'] <= 1.45) | (ALL_df['FIB4_vals'] >= 3.25)]
    fib4_indet_df = ALL_df.loc[(ALL_df['FIB4_vals'] > 1.45) & (ALL_df['FIB4_vals'] < 3.25)]
    FIB4 = BIO('FIB4', 1.45, 3.25, ALL_df[['FIB4_vals', 'target']])
    FIB4.name = 'FIB4(1.45,3.25)'
    ENS_FIB4_det = MLA('ENS_FIB4_det(0.6)', 0.6, fib4_det_df['ENS_probs'], fib4_det_df['target'])
    ENS_FIB4_indet = MLA('ENS_FIB4_indet(0.6)', 0.6, fib4_indet_df['ENS_probs'], fib4_indet_df['target'])
    ENS_FIB4_ALL = MLA('ENS_FIB4_ALL(0.6)', 0.6, ALL_df['ENS_probs'], ALL_df['target'])

    algs = [APRI, ENS_APRI_det, ENS_APRI_indet, ENS_APRI_ALL, 
            FIB4, ENS_FIB4_det, ENS_FIB4_indet, ENS_FIB4_ALL, 
            ENS_TE, ENS_EXP]    
    
    metrics = ['sens', 'spec', 'ppv', 'npv', 'per_det']
    
    for alg in algs: 
        res_dict = {'prevalence': prev, 
                    'trial': i, 
                    'alg': alg.name}
        for met in metrics: 
            res_dict[met] = alg.pm[met]
        res.append(res_dict)
            
#results = pd.DataFrame(res)
#results.to_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/prevalence modelling/pm_gaussian_noise(0.5).csv')
#results = results.groupby(['prevalence', 'alg']).mean()

# for col in results.columns.tolist(): 
#     results[col] *= 100
    




    
    


    



# def get_cm(label, preds):
#     cm = pd.DataFrame({'pred': preds, 'label': label})
#     cm['TN'] = np.where((cm['pred'] == 0) & (cm['label'] == 0), 1, 0)
#     cm['FP'] = np.where((cm['pred'] == 1) & (cm['label'] == 0), 1, 0)
#     cm['FN'] = np.where((cm['pred'] == 0) & (cm['label'] == 1), 1, 0)
#     cm['TP'] = np.where((cm['pred'] == 1) & (cm['label'] == 1), 1, 0)
#     cm['null'] = np.where(cm['pred'].isnull(), 1, 0)
#     return cm.sum()
    
# def get_pm(cm): 
#     TP = cm['TP']
#     TN = cm['TN']
#     FP = cm['FP']
#     FN = cm['FN']
    
#     sens = TP/(TP + FN)
#     spec = TN/(TN + FP)
#     PPV = TP/(TP + FP)
#     NPV = TN/(TN + FN)
        
#     return {'sens': sens, 'spec': spec, 'ppv': PPV, 'npv': NPV}   

# num_trials = 1000
# data = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/predictions/Combined.csv', index_col=0)
# data = data[['APRI_vals', 'APRI_probs', 'FIB4_vals', 'FIB4_probs', 'ENS1_probs', 'orig_fibrosis']]
# data['ENS_TE_preds'] = np.where(data['ENS1_probs'] >= 0.465, 1, 0)
# data['ENS_EXP_preds'] = np.where(data['ENS1_probs'] >= 0.6, 1, 0)
# data['target'] = np.where(data['orig_fibrosis'] <= 2, 0, 1)

# negatives = data.loc[data['target'] == 0]
# positives = data.loc[data['target'] == 1]

# res = []

# for num_pos in range(10,60,10): # Prevalence as a number from 10 to 50 out of 10000
#     num_neg = 10000 - num_pos
#     prev = (100*num_pos/num_neg)
    
#     for i in range(0,num_trials):
#         print('%0.2f%% - %d' %(prev, i))
#         neg = negatives.sample(n=num_neg, replace=True, random_state=i)
#         pos = positives.sample(n=num_pos, replace=True, random_state=i)

#         ALL_df = pd.concat([pos, neg])
#         APRI_df = ALL_df.loc[ALL_df['APRI_probs'] != 0.5][['APRI_probs', 'target']]
#         FIB4_df = ALL_df.loc[ALL_df['FIB4_probs'] != 0.5][['FIB4_probs', 'target']]
        
#         len_ALL_df = len(ALL_df)
#         len_APRI_df = len(APRI_df)
#         len_FIB4_df = len(FIB4_df)
        
#         APRI_per_det = 100*len_APRI_df/len_ALL_df
#         FIB4_per_det = 100*len_FIB4_df/len_ALL_df
#         ENS_TE_per_det = 100
#         ENS_EXP_per_det = 100
        
#         APRI_pm = get_pm(get_cm(APRI_df['target'], APRI_df['APRI_probs']))
#         FIB4_pm = get_pm(get_cm(FIB4_df['target'], FIB4_df['FIB4_probs']))
#         ENS_TE_pm = get_pm(get_cm(ALL_df['target'], ALL_df['ENS_TE_preds']))
#         ENS_EXP_pm = get_pm(get_cm(ALL_df['target'], ALL_df['ENS_EXP_preds']))
        
#         alg_dict = {'APRI': APRI_pm, 'FIB4': FIB4_pm, 'ENS_TE': ENS_TE_pm, 'ENS_EXP': ENS_EXP_pm}
#         alg_pdet = {'APRI': APRI_per_det, 'FIB4': FIB4_per_det, 'ENS_TE': ENS_TE_per_det, 'ENS_EXP': ENS_EXP_per_det}
        
#         for key in alg_dict.keys(): 
#             alg = alg_dict[key]
#             res_dict = {'prevalence': prev, 'trial': i, 'algorithm': key, 
#                         'sens': alg['sens'], 
#                         'spec': alg['spec'], 
#                         'ppv': alg['ppv'], 
#                         'npv': alg['npv'], 
#                         'per_det': alg_pdet[key]}
#             res.append(res_dict)
# results = pd.DataFrame(res)
# results.sort_values(by=['prevalence', 'algorithm', 'trial'], inplace=True)
# results.to_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/prevalence modelling/pm_results.csv')
