import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score

sns.set_theme()

# Generate the data for the experiment 
data = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Code/fib_stage_prev_vs_aucs_experiment_2/prevalences.csv')
data = data.loc[~data['Study'].isnull()]

risk = {'F0': (0,0.1999), 
        'F1': (0.2,0.3999),
        'F2': (0.4,0.5999),
        'F3': (0.6,0.7999),
        'F4': (0.8,0.9999)}

def gen_data_uniform_risk(p, r, rc):
    """
    p: Prevalence profile
    r: Risk profile (overlap)
    rc: Delta risk (+/-) for F1, F2, and F3
    """
    
    r['F0'] = (r['F0'][0], r['F0'][1] + rc/2)
    r['F1'] = (r['F1'][0] - rc, r['F1'][1] + rc)
    r['F2'] = (r['F2'][0] - rc, r['F2'][1] + rc)
    r['F3'] = (r['F3'][0] - rc, r['F3'][1] + rc)
    r['F4'] = (r['F4'][0] - rc/2, r['F4'][1])
    
    dfs = []
    for i in range(0,5): 
        s = 'F' + str(i)
        df = pd.DataFrame() 
        df['pred'] = np.random.uniform(low=r[s][0], high=r[s][1], size=int(p[s]*10))
        df['fibrosis'] = i
        dfs.append(df)
    dataset = pd.concat(dfs)
    dataset['target'] = np.where(dataset['fibrosis'] >= 2, 1, 0)
    
    return dataset 

# Logic of the experiment: 
# For each prevalence profile: 
    # Start with no risk overlap 
    # For each overlap setting:
        # Randomly generate 1000 patients, with fibrosis stage distribution following 
        # the prevalence distribution 
        # Uniformly assign risk throughout the fibrosis stages based on the distribution 
        # Calculate the AUC when distinguishing between F01 vs. F234 for each 
        # Store the AUC in a separate table 
        

res = []
for row in data.iterrows(): 
    study = row[1]['Study']
    print(study)

    prev = {'F0': row[1]['% F0'],
            'F1': row[1]['% F1'], 
            'F2': row[1]['% F2'], 
            'F3': row[1]['% F3'], 
            'F4': row[1]['% F4'], 
            'DANA': row[1]['my_DANA'],
            'study': row[1]['Study']}

    for r in range(0,21): 
        df = gen_data_uniform_risk(prev, risk.copy(), r/100) 
        
        # For each dataframe, calculate the AUROC 
        auc = roc_auc_score(df['target'], df['pred'])
        
        res_dict = prev.copy() 
        res_dict['auroc'] = auc
        res_dict['risk_overlap'] = r
        res.append(res_dict)
        
        #print('prev: %s, rc: %0.2f, auc: %0.2f' % (prev, r, auc))
        
        
results = pd.DataFrame(res)
results.to_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Code/fib_stage_prev_vs_aucs_experiment_2/prev_vs_overlap.csv')

# After plotting a heat map of how AUC changes across different risk overlap and DANA's, 
# Plot a relative comparison showing that risk_overlap causes much larger effects than fibrosis stage prevalence 


# import sys 
# sys.exit() 

# results = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Code/fib_stage_prev_vs_aucs_experiment_2/prev_vs_overlap.csv',
#                       index_col=0)
results.sort_values(by=['DANA', 'risk_overlap'], inplace=True)
results = results.loc[~results['study'].isin(['TLC'])] # Excluding TLC because it is missing F23
results = results[['study', 'DANA', 'F0', 'F1', 'F2', 'F3', 'F4', 'risk_overlap', 'auroc']]

studies = results[['study']].drop_duplicates()
studies.reset_index(drop=True, inplace=True)
studies['study_num'] = studies.index

results = results.merge(studies, on=['study'], how='left')
dt = results[['study', 'F0', 'F1', 'F2', 'F3', 'F4', 'DANA']].drop_duplicates()

plot_res = results[['study_num', 'risk_overlap', 'auroc']]
pr = plot_res.pivot('study_num', 'risk_overlap', 'auroc')

fig, ax = plt.subplots(figsize=(16,8))
ax = sns.heatmap(pr, annot=True, linewidths=0.2, cbar_kws={'label': 'AUROC'})
ax.set_xlabel('% Overlap Between Risk Scores Assigned to Different Fibrosis Stages')
ax.set_title('The Effect of Fibrosis Stage Prevalence vs. Risk Score Overlap on AUROC Across 22 Reported Prevalence Profiles')
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(False)

n = 22
colLabels = ['% F0', '% F1', '% F2', '% F3', '% F4', 'DANA']
the_table = ax.table(cellText=dt[['F0', 'F1', 'F2', 'F3', 'F4', 'DANA']].values,
                     colLabels = colLabels, 
                     rowLabels = dt['study'].tolist(),
                     bbox=(-0.4, 0.0, 0.4, (n+1)/n))
plt.show() 

dt.reset_index(drop=True, inplace=True)
dt['study_num'] = dt.index

# Get variance across different DANA's and across different fibrosis stages 
var_across_DANAs = pd.DataFrame(pr.var(axis=0)).round(5)
var_across_overlap = pd.DataFrame(pr.var(axis=1)).round(5).reset_index() 
var_across_overlap = var_across_overlap.merge(dt[['study_num', 'DANA']], on=['study_num'], how='left')


plt.figure()
plt.plot(var_across_DANAs.index, var_across_DANAs[0], 'o')
plt.xlabel('% Overlap')
plt.ylabel('AUROC Variance')
plt.title('Variance of Each Column (Over all DANA\'s)')
plt.xlim([-0.05, 20.05])
plt.ylim([0,0.0025])

plt.figure() 
plt.plot(var_across_overlap.DANA, var_across_overlap[0], 'o')
plt.xlabel('DANA')
plt.ylabel('AUROC Variance')
plt.title('Variance of Each Row (Over all % Overlaps)')
plt.ylim([0,0.0025])
plt.xlim([1.4, 2.5])
#plt.plot(var_across_overlap)_acr


# Plot 
