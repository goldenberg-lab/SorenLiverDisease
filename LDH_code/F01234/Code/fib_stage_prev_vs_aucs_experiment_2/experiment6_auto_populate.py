import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score

fs = [
    {'n': [0], 'p': [1]},
    {'n': [0], 'p': [2]},
    {'n': [0], 'p': [3]},
    {'n': [0], 'p': [4]},
    {'n': [0], 'p': [2,3,4]}, 
    {'n': [1], 'p': [2]},
    {'n': [1], 'p': [3]},
    {'n': [1], 'p': [4]},
    {'n': [1], 'p': [2,3,4]}, 
    {'n': [2], 'p': [3]},
    {'n': [2], 'p': [4]},
    {'n': [3], 'p': [4]},
    {'n': [0,1], 'p': [2]},
    {'n': [0,1], 'p': [3]},
    {'n': [0,1], 'p': [2,3]},
    {'n': [0,1], 'p': [2,4]}, 
    {'n': [0,1], 'p': [3,4]}, 
    {'n': [0,1], 'p': [4]},
    {'n': [0,1], 'p': [2,3,4]},
    {'n': [1,2], 'p': [3]},
    {'n': [1,2], 'p': [4]},
    {'n': [1,2], 'p': [3,4]},
    {'n': [0,1,2], 'p': [3]},
    {'n': [0,1,2], 'p': [4]},
    {'n': [0,1,2], 'p': [3,4]},
    {'n': [0,1,2,3], 'p': [4]},
    ]

# Steps: 
# Step 1. Load in the appropriate dataset 
# Step 2. Loop through fs, filter dataset labels based on these stages 
# Step 3. Assign target based on these stages 
# Step 4. Exclude any required patients for APRI/FIB4/NFS 
# Step 5. Report AUC 
# Step 6. Save dataset

dk = 'McGill'

# Toronto & McGill
# data = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/predictions/' + dk +  '.csv')


# data = data[['ENS1_probs', 'APRI_vals', 'FIB4_vals', 'APRI_probs', 'FIB4_probs', 'orig_fibrosis']]
# res = []
# for f in fs: 
#     stages = f['n'] + f['p']
#     tdata = data.loc[data['orig_fibrosis'].isin(stages)] # Filter out uneccessary fiboris stages 
#     tdata['target'] = np.where(tdata['orig_fibrosis'] >= min(f['p']), 1, 0)
    
#     pos = tdata.loc[tdata['target'] == 1]
#     neg = tdata.loc[tdata['target'] == 0]
    
#     if (len(pos) == 0 or len(neg) == 0): 
#         ENS_AUC = np.nan 
#         APRI_AUC = np.nan 
#         FIB4_AUC = np.nan 
#     else: 
            
#         # ENS AUC
#         ENS_AUC = roc_auc_score(tdata['target'], tdata['ENS1_probs'])
        
#         apri_tdata = tdata# .loc[tdata['APRI_probs'] != 0.5]
#         fib4_tdata = tdata# .loc[tdata['FIB4_probs'] != 0.5]
#         APRI_AUC = roc_auc_score(apri_tdata['target'], apri_tdata['APRI_vals']/max(apri_tdata['APRI_vals']))
#         FIB4_AUC = roc_auc_score(fib4_tdata['target'], fib4_tdata['FIB4_vals']/max(fib4_tdata['FIB4_vals']))
    
#     res.append({'n': 'F' + ''.join(str(x) for x in f['n']), 
#                 'p': 'F' + ''.join(str(x) for x in f['p']), 
#                 'ENS_AUC': ENS_AUC, 
#                 'APRI_AUC': APRI_AUC, 
#                 'FIB4_AUC': FIB4_AUC})

# EXPERT
# data = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/predictions/' + dk +  '.csv')
# data = data[['ENS1_probs', 'EXP_probs', 'orig_fibrosis']]

# res = [] 
# for f in fs: 
#     stages = f['n'] + f['p']
#     tdata = data.loc[data['orig_fibrosis'].isin(stages)] # Filter out uneccessary fiboris stages 
#     tdata['target'] = np.where(tdata['orig_fibrosis'] >= min(f['p']), 1, 0)
    
#     pos = tdata.loc[tdata['target'] == 1]
#     neg = tdata.loc[tdata['target'] == 0]
    
#     if (len(pos) == 0 or len(neg) == 0): 
#         ENS_AUC = np.nan 
#         EXP_AUC = np.nan 
#     else: 
            
#         # ENS AUC
#         ENS_AUC = roc_auc_score(tdata['target'], tdata['ENS1_probs'])
#         EXP_AUC = roc_auc_score(tdata['target'], tdata['EXP_probs'])
        
#     res.append({'n': 'F' + ''.join(str(x) for x in f['n']), 
#                 'p': 'F' + ''.join(str(x) for x in f['p']), 
#                 'ENS_AUC': ENS_AUC, 
#                 'EXP_AUC': EXP_AUC})

# NAFL
dk = 'NAFL'
data = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/predictions/' + dk +  '.csv')
data = data[['ENS1_probs', 'NAFL_probs', 'NAFL_vals', 'orig_fibrosis']]

res = []
for f in fs: 
    stages = f['n'] + f['p']
    tdata = data.loc[data['orig_fibrosis'].isin(stages)] # Filter out uneccessary fiboris stages 
    tdata['target'] = np.where(tdata['orig_fibrosis'] >= min(f['p']), 1, 0)
    
    pos = tdata.loc[tdata['target'] == 1]
    neg = tdata.loc[tdata['target'] == 0]
    
    if (len(pos) == 0 or len(neg) == 0): 
        ENS_AUC = np.nan 
        NFS_AUC = np.nan 
    else: 
            
        # ENS AUC
        ENS_AUC = roc_auc_score(tdata['target'], tdata['ENS1_probs'])
        nfs_tdata = tdata # .loc[tdata['NAFL_probs'] != 0.5]
        NFS_AUC = roc_auc_score(nfs_tdata['target'], nfs_tdata['NAFL_vals']/max(nfs_tdata['NAFL_vals']))
       
    
    res.append({'n': 'F' + ''.join(str(x) for x in f['n']), 
                'p': 'F' + ''.join(str(x) for x in f['p']), 
                'ENS_AUC': ENS_AUC, 
                'NFS': NFS_AUC})

# TE
# dk = 'TE'
# data = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/predictions/' + dk +  '.csv')
# data = data[['ENS1_probs', 'TE_vals', 'orig_fibrosis']]

# res = []
# for f in fs: 
#     stages = f['n'] + f['p']
#     tdata = data.loc[data['orig_fibrosis'].isin(stages)] # Filter out uneccessary fiboris stages 
#     tdata['target'] = np.where(tdata['orig_fibrosis'] >= min(f['p']), 1, 0)
    
#     pos = tdata.loc[tdata['target'] == 1]
#     neg = tdata.loc[tdata['target'] == 0]
    
#     if (len(pos) == 0 or len(neg) == 0): 
#         ENS_AUC = np.nan 
#         TE_AUC = np.nan 
#     else: 
            
#         # ENS AUC
#         ENS_AUC = roc_auc_score(tdata['target'], tdata['ENS1_probs'])
#         TE_AUC = roc_auc_score(tdata['target'], tdata['TE_vals']/max(tdata['TE_vals']))
       
    
#     res.append({'n': 'F' + ''.join(str(x) for x in f['n']), 
#                 'p': 'F' + ''.join(str(x) for x in f['p']), 
#                 'ENS_AUC': ENS_AUC, 
#                 'TE_AUC': TE_AUC})


results = pd.DataFrame(res).round(3)
results.to_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/fibrosis_stage_prevalence_vs_auroc/all_stages/' + dk + '.csv')
    
        
