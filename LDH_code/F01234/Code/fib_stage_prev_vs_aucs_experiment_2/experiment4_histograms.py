import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# Load in Toronto and Mcgill datasets 
# Generate histogram of risk scores assigned to F0, F1, and F4 patients across each algorithm 
# Demonstrate that AUC changes occur because of this

tor = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/predictions/Toronto.csv')
mcg = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/predictions/McGill.csv')

tor = tor[['APRI_vals', 'FIB4_vals', 'ENS1_probs', 'orig_fibrosis']]
mcg = mcg[['APRI_vals', 'FIB4_vals', 'ENS1_probs', 'orig_fibrosis']]

tor['ENS1_probs'] *= 100
mcg['ENS1_probs'] *= 100

tor.rename(columns={'APRI_vals': 'APRI', 
                    'FIB4_vals': 'FIB-4', 
                    'ENS1_probs': 'ENS'}, 
           inplace=True)
mcg.rename(columns={'APRI_vals': 'APRI', 
                    'FIB4_vals': 'FIB-4', 
                    'ENS1_probs': 'ENS'}, 
           inplace=True)


# for f in range(0, 5): 
#     ttor = tor.loc[tor['orig_fibrosis'] == f]
#     tmcg = mcg.loc[mcg['orig_fibrosis'] == f]
    
#     for col in ['APRI', 'FIB-4', 'ENS1']:
#         plt.figure()
#         plt.hist(ttor[col], density=True, label='TLC', alpha=0.5, linewidth=1.2, edgecolor='black')
#         plt.hist(tmcg[col], density=True, label='MUHC', alpha=0.5, linewidth=1.2, edgecolor='black')
#         plt.title(col + ' F' + str(f) + ' Risk Overlap Distribution')
#         plt.xlabel('Risk Score')
#         plt.ylabel('Density')
#         plt.grid(True)
#         plt.legend()
    
# Histogram 2: Across positive and negative examples in both datasets 
tor['target'] = np.where(tor['orig_fibrosis'] >= 3, 1, 0)
tor['origin'] = 'TLC'
tor_pos = tor.loc[tor['target'] == 1]
tor_neg = tor.loc[tor['target'] == 0]

mcg['target'] = np.where(mcg['orig_fibrosis'] >= 3, 1, 0)
mcg['origin'] = 'MUHC'
mcg_pos = mcg.loc[mcg['target'] == 1]
mcg_neg = mcg.loc[mcg['target'] == 0]


# Goal: Show that the score distribution b/w positives and negatives is different between the two datasets for the same algorithm 
# ENS, Toronto
# for col in ['ENS', 'APRI', 'FIB-4']:
#     plt.figure() 
#     plt.hist(tor_pos[col], bins=20, alpha=0.5, density=False, color='red',
#              linewidth=1.2, edgecolor='black', label='Advanced Fibrosis (F34)')
#     plt.hist(tor_neg[col], bins=20, alpha=0.5, density=False, color='green', 
#              linewidth=1.2, edgecolor='black', label='Non-Advanced Fibrosis (F012)')
#     plt.legend()
#     plt.grid(True)
#     plt.xlabel('Risk Score')
#     plt.ylabel('Frequency')
#     plt.title(col + ' Distribution of Risk Scores between Classes, TLC Dataset')
    
#     # ENS, McGill
#     plt.figure() 
#     plt.hist(mcg_pos[col], bins=20, alpha=0.5, density=False, color='red',
#              linewidth=1.2, edgecolor='black', label='Advanced Fibrosis (F34)')
#     plt.hist(mcg_neg[col], bins=20, alpha=0.5, density=False, color='green', 
#              linewidth=1.2, edgecolor='black', label='Non-Advanced Fibrosis (F012)')
#     plt.legend()
#     plt.grid(True)
#     plt.xlabel('Risk Score')
#     plt.ylabel('Frequency')
#     plt.title(col + ' Distribution of Risk Scores between Classes, MUHC Dataset')

comb = pd.concat([tor, mcg])
comb.rename(columns={'orig_fibrosis': 'Fibrosis Stage'}, inplace=True)


# ENSEMBLE 
# plt.figure(figsize=(9,3))
# ax = sns.violinplot(x='Fibrosis Stage', y='ENS', hue='origin', data=comb, 
#                     palette='Set2', split=True, inner='quartile', bw=0.2)
# ax.yaxis.grid(True)
# ax.legend(loc=4)
# ax.set_ylabel('ENS Risk Score (%)')
# ax.set_title('ENS Risk Score Distribution by Fibrosis Stage')

# APRI 
# print(len(comb))
# comb = comb.loc[comb['APRI'] <= 5]
# print(len(comb))

# plt.figure(figsize=(9,3))
# ax = sns.violinplot(x='Fibrosis Stage', y='APRI', hue='origin', data=comb, 
#                     palette='Set2', split=True, inner='quartile', bw=0.2)
# ax.yaxis.grid(True)
# ax.legend(loc=1)
# ax.set_ylabel('APRI Score')
# ax.set_title('APRI Risk Score Distribution by Fibrosis Stage')

# FIB4 
# print(len(comb))
# comb = comb.loc[comb['FIB-4'] <= 15]
# print(len(comb))
# plt.figure(figsize=(9,3))
# ax = sns.violinplot(x='Fibrosis Stage', y='FIB-4', hue='origin', data=comb, 
#                     palette='Set2', split=True, inner='quartile', bw=0.2)
# ax.yaxis.grid(True)
# ax.legend(loc=9)
# ax.set_ylabel('FIB-4 Score')
# ax.set_title('FIB-4 Risk Score Distribution by Fibrosis Stage')

# import sys 
# sys.exit()



# X axis is APRI_neg, APRI_pos, FIB4_neg, FIB4_pos, ENS_neg, ENS_pos
plt.figure(figsize=(9,3))
ADF_df = comb[['APRI', 'target', 'origin']]
ADF_df['type'] = np.where(ADF_df['target'] == 1, 'APRI+', 'APRI-')
ADF_df['type2'] = np.where(ADF_df['origin'] == 'TLC', 'APRI_TLC', 'APRI_MUHC')
ADF_df.rename(columns={'APRI': 'pred'}, inplace=True)
ADF_df = ADF_df.loc[ADF_df['pred'] <= 4]
ADF_df['pred'] = ADF_df['pred']/4*100

FDF_df = comb[['FIB-4', 'target', 'origin']]
FDF_df['type'] = np.where(FDF_df['target'] == 1, 'FIB4+', 'FIB4-')
FDF_df['type2'] = np.where(FDF_df['origin'] == 'TLC', 'FIB4_TLC', 'FIB4_MUHC')
FDF_df.rename(columns={'FIB-4': 'pred'}, inplace=True)
FDF_df = FDF_df.loc[FDF_df['pred'] <= 10]
FDF_df['pred'] = FDF_df['pred']/10*100

EDF_df = comb[['ENS', 'target', 'origin']]
EDF_df['type'] = np.where(EDF_df['target'] == 1, 'ENS+', 'ENS-')
EDF_df['type2'] = np.where(EDF_df['origin'] == 'TLC', 'ENS_TLC', 'ENS_MUHC')
EDF_df.rename(columns={'ENS': 'pred'}, inplace=True)

df = pd.concat([ADF_df, FDF_df, EDF_df])
#df = df.loc[df['origin'] == 'TLC']

df['target'] = np.where(df['target'] == 1, 'F34 (TP)', 'F012 (TN)')


ax = sns.violinplot(x='type2', y='pred', hue='target', data=df, 
                    palette={'F012 (TN)': 'lightgreen', 'F34 (TP)': 'lightcoral'}, 
                    split=True, inner='quartile', bw=0.1)
ax.yaxis.grid(True)
ax.legend(loc=9)
ax.set_xlabel('Algorithm')
ax.set_ylabel('Normalized Risk Score')
ax.set_title('Normalized Risk Score Distribution by Algorithm and Dataset')
