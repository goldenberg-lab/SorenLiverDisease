import os
import copy 
import numpy as np
import pandas as pd 
import pyodbc as db 
import matplotlib.pyplot as plt 

from sklearn.metrics import auc 
from sklearn.externals import joblib
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import precision_recall_curve 

from beautifultable import BeautifulTable

pd.options.mode.chained_assignment = None  # default='warn'

# TEMP 

def get_confusion_matrix(label, preds):
    cm = pd.DataFrame({'pred': preds, 'label': label})
    cm['TN'] = np.where((cm['pred'] == 0) & (cm['label'] == 0), 1, 0)
    cm['FP'] = np.where((cm['pred'] == 1) & (cm['label'] == 0), 1, 0)
    cm['FN'] = np.where((cm['pred'] == 0) & (cm['label'] == 1), 1, 0)
    cm['TP'] = np.where((cm['pred'] == 1) & (cm['label'] == 1), 1, 0)
    cm['null'] = np.where(cm['pred'].isnull(), 1, 0)
    return cm.sum()

def my_auc_prob(label, probs): 
    results = []
    
    new_probs = np.unique(probs.to_numpy(copy=True))
    mid_probs = new_probs[:-1] + np.diff(new_probs)/2
    mid_probs = np.unique(np.append(mid_probs, [0,1]))
    
    for t in mid_probs: 
        cm = get_confusion_matrix(label, (probs >= t))
        results.append([t, cm['TP'], cm['FP'], cm['FN'], cm['TN']])
        
    df = pd.DataFrame.from_records(results, columns=['thresh', 'TP', 'FP', 'FN', 'TN'])
    df['fpr'] = df['FP']/(df['FP'] + df['TN'])
    df['tpr'] = df['TP']/(df['TP'] + df['FN'])
    df['prc'] = df['TP']/(df['TP'] + df['FP'])
    
    auroc_df = df[['thresh', 'fpr', 'tpr']]
    auroc_tempf = auroc_df.drop_duplicates(subset=['tpr'], keep='first')
    auroc_templ = auroc_df.drop_duplicates(subset=['tpr'], keep='last')
    auroc_df_2 = pd.concat([auroc_tempf, auroc_templ])
    auroc_df_2.sort_values(by=['thresh'], ascending = [True], inplace=True)  
    auroc_tempf2 = auroc_df_2.drop_duplicates(subset=['fpr'], keep='first')
    auroc_templ2 = auroc_df_2.drop_duplicates(subset=['fpr'], keep='last')
    auroc_df_2 = pd.concat([auroc_tempf2, auroc_templ2])
    auroc_df_2.sort_values(by=['thresh'], ascending = [True], inplace=True)
    auroc_df_2.reset_index(drop=True, inplace=True)    
    auroc = auc(auroc_df_2['fpr'], auroc_df_2['tpr'])
    
    return auroc


def BIO_auc(df, name):
    vals = name + '_vals' 
    probs = name + '_probs' 
    bio_df = df[[vals, 'target']].loc[df[probs] != 0.5]
    bio_df[probs] = bio_df[vals]/max(bio_df[vals])
    return roc_auc_score(bio_df['target'], bio_df[probs])

def ENS_auc(df, low, high):
    ENS_df = df[['ENS3_probs', 'target', 'APRI_probs', 'FIB4_probs']]
    ENS_df['af'] = ENS_df['APRI_probs'] + ENS_df['FIB4_probs']
    ENS_df['indet'] = np.where(ENS_df['ENS3_probs'].between(low, high), 1, 0)
    ENS_df['indet'] = np.where((ENS_df['af'] == 0) | (ENS_df['af'] == 2), 0, ENS_df['indet'])
    ENS_df = ENS_df[['ENS3_probs', 'target']].loc[ENS_df['indet'] == 0]
    return roc_auc_score(ENS_df['target'], ENS_df['ENS3_probs'])


# Experiment # 1: For each test set, and for each combination of fibrosis stages, 
# Calculate AUROCs and put it in a table in the supplementary material 

# Step 1. Read in the data 
datakey = 'TE'
datapath = '/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/Predictions/F014_models/'
os.chdir(datapath)

data = pd.read_csv(datakey + '.csv', index_col=0)

if (datakey is 'TE'):
    data['TE_probs'] = 1

data.drop(columns={'SVM_probs', 'RFC_probs', 'GBC_probs', 'LOG_probs', 'MLP_probs'}, inplace=True)

# Step 2. Breakdown of fibrosis stages for each test set: 
tor_exp =       {1: {'n': [0], 'p':[4]}, 
                 2: {'n': [1], 'p':[4]},
                 3: {'n': [0,1], 'p':[4]},
                 4: {'n': [0], 'p':[1]},
                }

mcgill_nafl_te = {1: {'n': [0], 'p':[3]}, 
                 2: {'n': [0], 'p':[4]},
                 3: {'n': [0], 'p':[3,4]},
                 
                 4: {'n': [1], 'p':[3]}, 
                 5: {'n': [1], 'p':[4]},
                 6: {'n': [1], 'p':[3,4]},
                 
                 7: {'n': [2], 'p':[3]}, 
                 8: {'n': [2], 'p':[4]},
                 9: {'n': [2], 'p':[3,4]},
                 
                 10: {'n': [0,1], 'p':[3]}, 
                 11: {'n': [0,1], 'p':[4]},
                 12: {'n': [0,1], 'p':[3,4]},
                 
                 13: {'n': [1,2], 'p':[3]}, 
                 14: {'n': [1,2], 'p':[4]},
                 15: {'n': [1,2], 'p':[3,4]},
                 
                 16: {'n': [0,1,2], 'p':[3]}, 
                 17: {'n': [0,1,2], 'p':[4]},
                 18: {'n': [0,1,2], 'p':[3,4]},
                 
                 20: {'n': [0], 'p':[1]}, 
                 21: {'n': [1], 'p':[2]},
                 22: {'n': [2], 'p':[3]},
                 23: {'n': [3], 'p':[4]},}

# Okay. Next, I need the right cutoffs for each algorithm under each experiment. 
# Ensemble threshold dictionary
etd = {'Toronto': (0.150,0.775),
        'Expert': (0.675,0.675),
        'McGill': (0.450,0.875),
        'NAFL':   (0.225,0.775),
        'TE':     (0.775,0.775)} 

alg_name = {'Toronto': 'ENS2b',
            'Expert':  'ENS2c',
            'McGill':  'ENS2b', 
            'NAFL':    'ENS2b', 
            'TE':      'ENS2d'}

# Relevant columns to extract
rel_cols = {'Toronto': ['APRI_vals', 'FIB4_vals', 'ENS3_probs', 'APRI_probs', 'FIB4_probs', 'orig_fibrosis'],
            'Expert': ['APRI_vals', 'FIB4_vals', 'ENS3_probs', 'APRI_probs', 'FIB4_probs', 'EXP_probs', 'orig_fibrosis'],
            'McGill': ['APRI_vals', 'FIB4_vals', 'ENS3_probs', 'APRI_probs', 'FIB4_probs', 'orig_fibrosis'],
            'NAFL': ['APRI_vals', 'FIB4_vals', 'NAFL_vals', 'ENS3_probs', 'APRI_probs', 'FIB4_probs', 'orig_fibrosis', 'NAFL_probs'],
            'TE': ['APRI_vals', 'FIB4_vals', 'TE_vals', 'ENS3_probs', 'APRI_probs', 'FIB4_probs', 'TE_probs', 'orig_fibrosis']}

# Toronto:      APRI vs. FIB4 vs. ENS2(0.25, 0.45) vs. ENS2b (0.15, 0.775)
# Expert:       APRI vs. FIB4 vs. EXPERT(0.5, 0.5) vs. ENS2(0.25, 0.45) vs. ENS2c (0.675)
# McGill:         APRI vs. FIB4 vs. ENS2 vs. ENS2b (0.45, 0.875)
# NAFL:        APRI vs. FIB4 vs. NFS vs. ENS2 vs. ENS2b 
# TE:           APRI vs. FIB4 vs. TE vs. ENS2 vs. ENS2d

num_iters = 100


if (datakey == 'Toronto' or datakey == 'Expert'):
    exps = tor_exp
else:
    exps = mcgill_nafl_te


results = []

for exp in exps.keys():
    pos_class = exps[exp]['p']
    neg_class = exps[exp]['n']
    
    all_df = data.loc[data['orig_fibrosis'].isin(pos_class + neg_class)]
    all_df = all_df[rel_cols[datakey]]
    all_df['target'] = np.where(all_df['orig_fibrosis'].isin(pos_class), 1, 0)

    # Okay. I want to run each experiment 1k times so I have error bars to report. 
    # Step 1. Outer dictionary to store performance and error bars of each alg 
    # Step 2. Inner dictionary to store AUC at each iteration. 
    
    
    experiment = 'F' + ''.join([str(n) for n in neg_class]) + ' vs. F'+''.join([str(p) for p in pos_class])

    
    trials = [] 
    
    for i in range(0, num_iters): 

        try:
            df = all_df.sample(frac=1, replace=True, random_state=i)
            
            # Dictionary to store results
            inner = {}
            
            # Now, for each algorithm, I need to calculate an AUROC 
            inner['APRI'] = 100*BIO_auc(df, 'APRI')
            inner['FIB4'] = 100*BIO_auc(df, 'FIB4')
        
            
            if (datakey == 'Expert'):
                inner['EXP'] = 100*my_auc_prob(df['target'], df['EXP_probs'])
        
            if (datakey == 'NAFL'):
                inner['NAFL'] = 100*BIO_auc(df, 'NAFL')
                
            if (datakey == 'TE'):
                inner['TE'] = 100*BIO_auc(df, 'TE')
                
            inner['ENS2'] = 100*ENS_auc(df, 0.25, 0.45)
            inner[alg_name[datakey]] = 100*ENS_auc(df, etd[datakey][0], etd[datakey][1])  
            
            trials.append(inner)
        except ValueError:
            print('Error: Only 1 class detected, skipping!')

    # Use standard error, it is better suited in this context 
    
    all_trials = pd.DataFrame.from_records(trials)
    at_mn = pd.DataFrame(all_trials.mean(axis=0)).rename(columns={0: 'mean'})
    at_se = pd.DataFrame(all_trials.std(axis=0)/len(all_trials)).rename(columns={0: 'SE'})
    at = at_mn.merge(at_se, left_index=True, right_index=True, how='left').round(1)
    at['desc'] = at['mean'].astype(str) + ' \u00B1 ' + at['SE'].astype(str)   
    at = at[['desc']].transpose()
    at['i'] = experiment
    at.index = at['i']
    at.drop(columns={'i'}, inplace=True)
    
    results.append(at)

res = pd.concat(results)

savedir = '/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/AUROC_FibStage_prevalence_analysis/EXP_1_results/'

#res.to_excel(savedir + datakey + '.xlsx')
        

    # Okay. I have now calculated the AUROCs for Each Experiment. 
    # I should store the results in a dataframe and visualize to see if it makes sense. 
    #     
    
    


# What is the algorithm I need to implement? 
# For the current dataset 
# For the current definition of positive and negative classes 
# Calculate the AUROC of each algorithm 
# Remember to exclude patients in the indeterminate zones 


# Okay. Experiment 1 is 

















# McGill, NAFL, TE: 
    # F0 vs. F3    D
    # F0 vs. F4    D
    # F0 vs. F34   D
    
    # F1 vs. F3    D
    # F1 vs. F4    D
    # F1 vs. F34   D
    
    # F2 vs. F3    D
    # F2 vs. F4    D
    # F2 vs. F34   D
    
    # F01 vs. F3   D
    # F01 vs. F4   D
    # F01 vs. F34  D
    
    # F12 vs. F3   D
    # F12 vs. F4   D
    # F12 vs. F34  D
    
    # F012 vs. F3    D
    # F012 vs. F4    D
    # F0123 vs. F34  D
    
    # Other experiments, to see how well our model ranks # Even though it wasn't trained for this 
    # F0 vs. F1   D
    # F1 vs. F2   D
    # F2 vs. F3   D
    # F3 vs. F4   D
    
