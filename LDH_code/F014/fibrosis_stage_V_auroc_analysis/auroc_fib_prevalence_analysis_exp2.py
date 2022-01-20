import os
import copy 
import numpy as np
import pandas as pd 
import pyodbc as db 
import scipy.stats as st
import matplotlib.pyplot as plt 

from sklearn.metrics import auc 
from sklearn.externals import joblib
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import precision_recall_curve 

from beautifultable import BeautifulTable
from pandas.plotting import table

pd.options.mode.chained_assignment = None  # default='warn'


# Experiment # 2: Standardize fibrosis prevalences across TLC and MUHC, and demonstrate 
# that the AUROC is in fact different even when fibrosis stage prevalences are identical. 
# Use this to explain the underlying causal reasons why standardizing based on class labels is incorrect. 
# Also mention the limitations of the method proposed in the paper 
#  Can't standardize using FibroTest Regression, it is a different method based on different parameters 
#  Can't generate a regression curve with our own models, as this is the first place it is being proposed 
#  Models should be compared on the same patients, as stated in the paper previoulsy cited, which we did,
#  justifying why AUCs do not need to be standardized, standardization is outside the scope of this work
#  Add sentence to limitations about AUC cannot be directly compared between models, as 
#  prevalence of positive to negative class labels as well as the function used to calculate risk for patients based
#  on the input parameters 
# We have multiple etiologies of liver disease, it's not accurate to compare across 


#  Can generate a standardization curve with our own model 
#  Models should be compared on the same patients (quote the conclusion of the cited paper)
#  which we did, and demonstrate that this is the only way to avoid bias introduced by fibrosis stages 
#  Also add a sentence in the limitations about why the reported AUCs of algorithms cannot be 
#  compared along different projects 


# Okay. Break the task down into steps.  
# Step 1: Load in the TLC and MUHC datasets.                        (DONE)
# Step 2: Get proportions of F014                                   (DONE)
# Step 3: Sample both TLC and MUHC datasets                         (DONE) 

# Step 3: Calculate AUROCs on Bootstraped samples from those datasets 
# Step 4: Repeat this process 1000 times to have distribution of AUCs and difference between algorithm performance 
# Step 5: Calculate each algorithm's AUC on MUHC and TLC data 
# Step 6: Do a statistical test and demonstrate that AUC is significantly different 
#         even when proportion of F0, F1, and F4 is the same 


def get_indet(df):
    df['af'] = df['APRI_probs'] + df['FIB4_probs']
    df['indet'] = np.where(df['ENS3_probs'].between(0.25, 0.45) & (~df['af'].isin([0,2])), 1, 0)
    return df 

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

### PART 1: RUNNING THE EXPERIMENTS AND GETTING PERFORMANCE METRICS 
# Part 1. Generate the individual AUC calculations for different fibrosis stage prevalences for each alg 
# Across the TLC and MUHC datasets. 

# datapath = '/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/Predictions/F014_models/'
# os.chdir(datapath)

# cols = ['APRI_vals', 'APRI_probs', 'FIB4_vals', 'FIB4_probs', 'ENS3_probs', 'orig_fibrosis']

# tor = get_indet(pd.read_csv('Toronto.csv', index_col=0)[cols]).rename(columns={'orig_fibrosis': 'orig'})
# tor['target'] = np.where(tor['orig'] == 4, 1, 0)
# tor0 = tor.loc[tor['orig'] == 0]
# tor1 = tor.loc[tor['orig'] == 1]
# tor4 = tor.loc[tor['orig'] == 4]
# ntor = len(tor)

# mcg = get_indet(pd.read_csv('McGill.csv', index_col=0)[cols]).rename(columns={'orig_fibrosis': 'orig'})
# mcg['target'] = np.where(mcg['orig'] == 4, 1, 0)
# mcg0 = mcg.loc[mcg['orig'] == 0]
# mcg1 = mcg.loc[mcg['orig'] == 1]
# mcg4 = mcg.loc[mcg['orig'] == 4]
# nmcg = len(mcg)

# lc = 0
# n_iters = 1000
# results = []

# for p4 in np.linspace(0.1, 0.9, num=9):
#     p01 = 1 - p4

#     pf0 = np.linspace(0.1, 0.9, num=9)

#     for p0 in pf0: 
#         p1 = 1 - p0 
        
#         p0n = p0*p01
#         p1n = (1-p0)*p01
#         p4n = p4

#         #print('F0: %4.2f, F1: %4.2f, F4: %4.2f, Total: %4.2f' % (100*p0n, 100*p1n, 100*p4n, 100*(p1n + p0n + p4n)))
    
#         # What I can do is store all of these into a dataframe, then reload for analysis. 
#         # This will allow me to checkpoint my work. 
#         for i in range(0, n_iters):
#             print('%0.2f%% percent completed' % (100*lc/81000))
            
#             t0 = tor0.sample(n=int(p0n*ntor), replace=True, random_state=i)
#             t1 = tor1.sample(n=int(p1n*ntor), replace=True, random_state=i)
#             t4 = tor4.sample(n=int(p4n*ntor), replace=True, random_state=i)
#             tor_i = pd.concat([t0, t1, t4])
            
#             m0 = mcg0.sample(n=int(p0n*nmcg), replace=True, random_state=i)
#             m1 = mcg1.sample(n=int(p1n*nmcg), replace=True, random_state=i)
#             m4 = mcg4.sample(n=int(p4n*nmcg), replace=True, random_state=i)            
#             mcg_i = pd.concat([m0, m1, m4])
            
#             # Okay. I have now calculated AUCs for this iteration. I want to store them all 
#             tor_APRI = BIO_auc(tor_i, 'APRI')
#             tor_FIB4 = BIO_auc(tor_i, 'FIB4')
#             tor_ENS2 = ENS_auc(tor_i, 0.25, 0.45)
            
#             mcg_APRI = BIO_auc(mcg_i, 'APRI')
#             mcg_FIB4 = BIO_auc(mcg_i, 'FIB4')
#             mcg_ENS2 = ENS_auc(mcg_i, 0.25, 0.45)
                
#             res_dict = {}
#             res_dict['lc'] = lc
#             res_dict['pF0'] = p0n
#             res_dict['pF1'] = p1n
#             res_dict['pF4'] = p4n
#             res_dict['tor_APRI'] = tor_APRI
#             res_dict['mcg_APRI'] = mcg_APRI
#             res_dict['tor_FIB4'] = tor_FIB4
#             res_dict['mcg_FIB4'] = mcg_FIB4
#             res_dict['tor_ENS2'] = tor_ENS2 
#             res_dict['mcg_ENS2'] = mcg_FIB4
#             lc += 1
            
#             results.append(res_dict)
            
# savedir = '/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/AUROC_FibStage_prevalence_analysis/EXP_2_results/'

# df_results = pd.DataFrame.from_records(results)
# df_results.to_hdf(savedir + 'experiment_aucs(key=auc).h5', key='auc')
    
        # Okay. I now have proportion of F0, F1, and F4 for each of my experiments. 
        
### PART 2: Demonstrating that the AUC between the datasets has a statistically significant difference despite the 
#           two datasets having the same proportion of Fibrosis stages in each experiment. 
savedir = '/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/AUROC_FibStage_prevalence_analysis/EXP_2_results/'
dfr = pd.read_hdf(savedir + 'experiment_aucs(key=auc).h5', key='auc').drop(columns={'lc'})

exps = dfr[['pF0', 'pF1', 'pF4']].drop_duplicates()
APRI_df = dfr[['pF0', 'pF1', 'pF4', 'tor_APRI', 'mcg_APRI']]
APRI_df['APRI_diff'] = APRI_df['tor_APRI'] - APRI_df['mcg_APRI']

# APRI_df_mean = APRI_df.groupby(['pF0', 'pF1', 'pF4']).mean().reset_index()
# APRI_df_2p5 = APRI_df.groupby(['pF0', 'pF1', 'pF4']).quantile(0.025).reset_index()

stat_results = []

for row in exps.iterrows():
    
    res_dict = {}
    res_dict['pF0'] = 100*row[1]['pF0']
    res_dict['pF1'] = 100*row[1]['pF1']
    res_dict['pF4'] = 100*row[1]['pF4']
    
    temp = dfr.loc[(dfr['pF0'] == row[1]['pF0']) & (dfr['pF1'] == row[1]['pF1']) & (dfr['pF4'] == row[1]['pF4'])]
    
    _, APRI_mwu = st.mannwhitneyu(temp['tor_APRI'], temp['mcg_APRI'])
    _, FIB4_mwu = st.mannwhitneyu(temp['tor_FIB4'], temp['mcg_FIB4'])
    _, ENS2_mwu = st.mannwhitneyu(temp['tor_ENS2'], temp['mcg_ENS2'])
    
    _, APRI_wtt = st.ttest_ind(temp['tor_APRI'], temp['mcg_APRI'], equal_var=False)
    _, FIB4_wtt = st.ttest_ind(temp['tor_FIB4'], temp['mcg_FIB4'], equal_var=False)
    _, ENS2_wtt = st.ttest_ind(temp['tor_ENS2'], temp['mcg_ENS2'], equal_var=False)
    
    _, APRI_ks = st.ks_2samp(temp['tor_APRI'], temp['mcg_APRI'])
    _, FIB4_ks = st.ks_2samp(temp['tor_FIB4'], temp['mcg_FIB4'])
    _, ENS2_ks = st.ks_2samp(temp['tor_ENS2'], temp['mcg_ENS2'])
    
    
    name  = '0: %0.2f, 1:%0.2f, 4: %0.2f, p-value = %0.4f - APRI' % (row[1]['pF0'], row[1]['pF1'], row[1]['pF4'], APRI_wtt)
    
    # plt.figure()
    # plt.hist(temp['tor_ENS2'], bins=100, label='tor')
    # plt.hist(temp['mcg_ENS2'], bins=100, label='mcg')
    # plt.title(name)
    # plt.grid(True)
    # plt.legend()
    
    res_dict['APRI_mwu'] = APRI_mwu
    res_dict['FIB4_mwu'] = FIB4_mwu
    res_dict['ENS2_mwu'] = ENS2_mwu
    
    res_dict['APRI_wtt'] = APRI_wtt
    res_dict['FIB4_wtt'] = FIB4_wtt
    res_dict['ENS2_wtt'] = ENS2_wtt
    
    res_dict['APRI_ks'] = APRI_ks
    res_dict['FIB4_ks'] = FIB4_ks
    res_dict['ENS2_ks'] = ENS2_ks    
    stat_results.append(res_dict)
    
stats = pd.DataFrame.from_records(stat_results)
stats['APRI max p-value'] = stats[['APRI_mwu', 'APRI_wtt', 'APRI_ks']].max(axis=1)
stats['FIB4 max p-value'] = stats[['FIB4_mwu', 'FIB4_wtt', 'FIB4_ks']].max(axis=1)
stats['ENS2 max p-value'] = stats[['ENS2_mwu', 'ENS2_wtt', 'ENS2_ks']].max(axis=1)
stats.sort_values(by=['pF4', 'pF1', 'pF0'], inplace=True)
stats.reset_index(drop=True, inplace=True)
stats.rename(columns={'pF0': '% F0', 'pF1': '% F1', 'pF4': '% F4'},inplace=True)
#stat, p = st.mannwhitneyu(f_data, e_data)

stats1 = stats.loc[(stats.index >= 60) & (stats.index < 80)].round(2)
props1 = stats1[['% F0', '% F1', '% F4']].transpose()
# for col in props1.columns.tolist():
#     props1[col] = props1[col].astype(int)
#stats2 = stats.loc[stats.index >= 41]

# plt.figure(figsize=(40.5,10))
# plt.ylim([0,0.1])
# plt.xlim([0,81])
# plt.plot([0, 81], [0.05, 0.05], color='red', linewidth=10)
# plt.bar(stats.index, stats['APRI_maxp'], alpha=0.8, label='APRI', width=0.1)
# plt.bar(stats.index, stats['FIB4_maxp'], alpha=0.8, label='FIB4', width=0.1)
# plt.bar(stats.index, stats['ENS2_maxp'], alpha=0.8, label='ENS2', width=0.1)
# plt.legend()
# plt.grid(True)

# columns = [str(i) for i in stats.index]
# rows = ['pF0', 'pF1', 'pF4']

# plt.savefig('temp.png')

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
#(table=stats1[['% F0', '% F1', '% F4']].transpose(), ax=ax, kind='bar', )
table(ax, props1, cellLoc='center', colLabels=None)
stats1[['APRI max p-value', 'FIB4 max p-value', 'ENS2 max p-value']].plot(ax=ax, kind='bar')
ax.set_ylim([0,0.06])
ax.grid(True)
ax.xaxis.set_ticklabels([])
ax.set_title('Fibrosis Stage Prevalence vs. p-value of TLC_AUC_dist = MUHC_AUC_dist')
ax.set_ylabel('p-value')
ax.plot([0, max(stats1.index)], [0.05, 0.05], color='red', linewidth=2)


# FIB4_df = df_results[['pF0', 'pF1', 'pF4', 'tor_FIB4', 'mcg_FIB4']]
# ENS2_df = df_results[['pF0', 'pF1', 'pF4', 'tor_ENS2', 'mcg_ENS2']]

