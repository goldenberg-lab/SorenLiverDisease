import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.metrics import auc 
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score 

def get_aucs(prev, exp):
    res = []

    for row in prev.iterrows(): 
        r = row[1]
        num_F0 = r['F0']*10  
        num_F1 = r['F1']*10
        num_F2 = r['F2']*10
        num_F3 = r['F3']*10
        num_F4 = r['F4']*10
        
        F0 = pd.DataFrame() 
        F1 = pd.DataFrame() 
        F2 = pd.DataFrame() 
        F3 = pd.DataFrame() 
        F4 = pd.DataFrame() 
           
        F0['pred'] = np.random.uniform(low=exp['F0']['low'], high=exp['F0']['high'], size=num_F0)
        F1['pred'] = np.random.uniform(low=exp['F1']['low'], high=exp['F1']['high'], size=num_F1)
        F2['pred'] = np.random.uniform(low=exp['F2']['low'], high=exp['F2']['high'], size=num_F2)
        F3['pred'] = np.random.uniform(low=exp['F3']['low'], high=exp['F3']['high'], size=num_F3)
        F4['pred'] = np.random.uniform(low=exp['F4']['low'], high=exp['F4']['high'], size=num_F4)
        
        F0['fibrosis'] = 0
        F1['fibrosis'] = 1
        F2['fibrosis'] = 2
        F3['fibrosis'] = 3
        F4['fibrosis'] = 4
        
        F0['target'] = 0
        F1['target'] = 0
        F2['target'] = 0
        F3['target'] = 1
        F4['target'] = 1
        
        data = pd.concat([F0, F1, F2, F3, F4])
        auc = roc_auc_score(data['target'], data['pred'])
        
        res.append({'F0': r['F0'], 
                        'F1': r['F1'], 
                        'F2': r['F2'], 
                        'F3': r['F3'], 
                        'F4': r['F4'],
                        'auc': auc})
    
    results = pd.DataFrame(res)
    return results 

# Step 1. (DONE) Create 1000 patients 
# Step 2. (DONE) Define upper and lower bounds for each fibrosis stage 
# Step 3. (DONE) Iterate over permutations of different prevalence profiles 
# Step 4. For each prevalence profile: 
    # Generate a data sample consisting of that many patients of class key who have probability ranging from low to high 
    # Calculate AUC on that distribution 
    # Plot AUC vs. distribution for the different prevalence profiles 

np.random.seed(0)
prevalences = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Code/fib_stage_prev_vs_aucs_experiment_2/prevalences.csv')
e2_no = {'F0': {'low': 0, 'high': 0.2},
          'F1': {'low': 0.2, 'high': 0.4},
          'F2': {'low': 0.4, 'high': 0.6},
          'F3': {'low': 0.6, 'high': 0.8},
          'F4': {'low': 0.8, 'high': 1}}

e2_sm = {'F0': {'low': 0, 'high': 0.225},
          'F1': {'low': 0.175, 'high': 0.425},
          'F2': {'low': 0.375, 'high': 0.625},
          'F3': {'low': 0.575, 'high': 0.825},
          'F4': {'low': 0.875, 'high': 1}}

e2_md = {'F0': {'low': 0, 'high': 0.25},
          'F1': {'low': 0.15, 'high': 0.45},
          'F2': {'low': 0.35, 'high': 0.65},
          'F3': {'low': 0.55, 'high': 0.85},
          'F4': {'low': 0.75, 'high': 1}}


e2_lg = {'F0': {'low': 0, 'high': 0.6},
          'F1': {'low': 0.1, 'high': 0.7},
          'F2': {'low': 0.2, 'high': 0.8},
          'F3': {'low': 0.3, 'high': 0.9},
          'F4': {'low': 0.4, 'high': 1.0}}


e2_mx = {'F0': {'low': 0, 'high': 1},
          'F1': {'low': 0, 'high': 1},
          'F2': {'low': 0, 'high': 1},
          'F3': {'low': 0, 'high': 1},
          'F4': {'low': 0, 'high': 1}}


no_overlap = get_aucs(prevalences, e2_no)
sm_overlap = get_aucs(prevalences, e2_sm)
md_overlap = get_aucs(prevalences, e2_md)
lg_overlap = get_aucs(prevalences, e2_lg)
mx_overlap = get_aucs(prevalences, e2_mx)

# Read in actual Toronto and McGill Data and Get Distributions 
path = '/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/predictions/'

tor = pd.read_csv(path + 'Toronto.csv', index_col=0)
mcg = pd.read_csv(path + 'McGill.csv', index_col=0)
exp = pd.read_csv(path + 'Expert.csv', index_col=0)
nafl = pd.read_csv(path + 'NAFL.csv', index_col=0)
te = pd.read_csv(path + 'TE.csv', index_col=0)

algs = {'tor': ['ENS1_probs', 'APRI_vals', 'FIB4_vals', 'orig_fibrosis'], 
        'mcg': ['ENS1_probs', 'APRI_vals', 'FIB4_vals', 'orig_fibrosis'],
        'exp': ['ENS1_probs', 'EXP_probs', 'orig_fibrosis'],
        'nafl': ['ENS1_probs', 'NFS_probs', 'orig_fibrosis'], 
        'te': ['ENS1_probs', 'TE_probs', 'orig_fibrosis']}

dfs = {'tor': tor, 
       'mcg': mcg, 
       'exp': exp, 
       'nafl': nafl, 
       'te': te}

for key in ['exp']:#dfs.keys():     
    print(key)
    
    stage_res = []
    
    df = dfs[key][algs[key]]
    
    for F in sorted(df['orig_fibrosis'].unique()): 
        tdf = df.loc[df['orig_fibrosis'] == F]
        
        tdf_mean = tdf.groupby(['orig_fibrosis']).mean() 
        tdf_25 = tdf.groupby(['orig_fibrosis']).quantile(0.025)
        tdf_975 = tdf.groupby(['orig_fibrosis']).quantile(0.975)
        
        for col in tdf_mean.columns: 
            tdf_mean.rename(columns={col: col + '_mean'}, inplace=True)
            tdf_25.rename(columns={col: col + '_25'}, inplace=True)
            tdf_975.rename(columns={col: col + '_975'}, inplace=True)
        
        ntdf = tdf_mean.merge(tdf_25, left_index=True, right_index=True, how='left')
        ntdf = ntdf.merge(tdf_975, left_index=True, right_index=True, how='left')
        stage_res.append(ntdf.round(2))
        
        
        
    stage_df = pd.concat(stage_res)
    break
        



