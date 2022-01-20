import numpy as np
import pandas as pd
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score 

# Step 1. Import the dataset with predictions. 
# Step 2. Okay. The predictions and algorithm are correct.
# Step 3. Check how I got APRI, FIB-4, NFS, Expert, and TE predictions. 

def get_confusion_matrix(label, preds):
    cm = pd.DataFrame({'pred': preds, 'label': label})
    cm['TN'] = np.where((cm['pred'] == 0) & (cm['label'] == 0), 1, 0)
    cm['FP'] = np.where((cm['pred'] == 1) & (cm['label'] == 0), 1, 0)
    cm['FN'] = np.where((cm['pred'] == 0) & (cm['label'] == 1), 1, 0)
    cm['TP'] = np.where((cm['pred'] == 1) & (cm['label'] == 1), 1, 0)
    cm['null'] = np.where(cm['pred'].isnull(), 1, 0)
    return cm.sum()

def my_auc_non_prob(label, values): 
    
    results = []
    new_vals = np.unique(values.to_numpy(copy=True))
    mid_vals = new_vals[:-1] + np.diff(new_vals)/2
    mid_vals = np.unique(np.append(mid_vals, [-1000,1000]))

    for t in mid_vals: 
        cm = get_confusion_matrix(label, (values >= t)*1)
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

# Main figure table: Do not exclude indeterminate patients to avoid bias.
# Appendix table: Exclude indeterminate patients for baselines. 

# Step 1. Loading the data 
pred_path = "C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\F01234\\Predictions\\predictions\\"
write_path = "C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\F01234\\Predictions\\AUROC vs. def_advanced_fibrosis_without_indets\\"

dks = ['Toronto', 'NAFL', 'McGill']
dks = ['Toronto', 'NAFL', 'McGill']

combs = {'F0vF1': {'neg': [0], 'pos': [1]}, 
         'F0vF2': {'neg': [0], 'pos': [2]},
         'F0vF3': {'neg': [0], 'pos': [3]},
         'F0vF4': {'neg': [0], 'pos': [4]},
         'F0vF234': {'neg': [0],'pos': [2,3,4]},
         'F1vF2': {'neg': [1], 'pos': [2]},
         'F1vF3': {'neg': [1], 'pos': [3]},
         'F1vF4': {'neg': [1], 'pos': [4]},
         'F1vF234': {'neg': [1], 'pos': [2,3,4]},
         'F2vF3': {'neg': [2], 'pos': [3]},
         'F2vF4': {'neg': [2], 'pos': [4]},
         'F3vF4': {'neg': [3], 'pos': [4]},
         'F01vF2': {'neg': [0,1], 'pos': [2]},
         'F01vF3': {'neg': [0,1], 'pos': [3]},
         'F01vF23':{'neg': [0,1], 'pos': [2,3]},
         'F01vF24':{'neg': [0,1], 'pos': [2,4]},
         'F01vF34':{'neg': [0,1], 'pos': [3,4]},
         'F01vF4': {'neg': [0,1], 'pos': [4]},
         'F01vF234': {'neg': [0,1], 'pos': [2,3,4]},
         'F12vF3': {'neg': [1,2], 'pos': [3]},
         'F12vF4': {'neg': [1,2], 'pos': [4]},
         'F12vF34': {'neg': [1,2], 'pos': [3,4]},
         'F012vF3': {'neg': [0,1,2], 'pos': [3]},
         'F012vF4': {'neg': [0,1,2], 'pos': [4]},
         'F012vF34': {'neg': [0,1,2], 'pos': [3,4]},
         'F0123vF4': {'neg': [0,1,2,3], 'pos': [4]}}

df_combinations = {'Toronto': ['F0vF1', 'F0vF4','F1vF4', 'F01vF4'], 
                   'McGill': list(combs.keys()), 
                   'NAFL': list(combs.keys())}

rel_cols = {'Toronto': ['APRI_vals', 'FIB4_vals', 'orig_fibrosis'],
            'McGill': ['APRI_vals', 'FIB4_vals', 'orig_fibrosis'],
            'NAFL': ['NAFL_vals',  'orig_fibrosis'] ,
            }

rel_baselines = {'Toronto': ['APRI', 'FIB4'],
                 'McGill': ['APRI', 'FIB4'],
                 'NAFL': ['NAFL']}

num_trials=1000

results = []
for dk in dks: 
    results = []
    df = pd.read_csv(pred_path + dk + '.csv', index_col=0) 
    
    for bl in rel_baselines[dk]: 
        tdf = df.loc[df[bl+'_probs'] != 0.5][['orig_fibrosis', bl + '_vals']]
        
        for c in df_combinations[dk]: 
            tdfc = tdf.loc[tdf['orig_fibrosis'].isin(combs[c]['neg'] + combs[c]['pos'])]
            tdfc['label'] = np.where(tdfc['orig_fibrosis'].isin(combs[c]['pos']), 1, 0)
                    
            for i in range(0, num_trials): 
                print(dk, bl, c, i)
                tdfc_i = tdfc.sample(n=len(tdf), replace=True, random_state=i)
                bl_auc_i = my_auc_non_prob(tdfc_i['label'], tdfc_i[bl + "_vals"])
                results.append({'comb': c, 'iter': i, 'alg': bl, 'auroc': bl_auc_i})
         
            result_df = pd.DataFrame.from_records(results)
            result_df.sort_values(by=['comb', 'alg', 'iter'], inplace=True)
            result_df.to_csv(write_path + dk + '.csv')
        
    
    # # Simplify var names so I can use roc_auc_score to calculate AUCs

        
    # for c in df_combinations[dk]:
    #     tdf = df[rel_cols[dk]]
    #     tdf = tdf.loc[tdf['orig_fibrosis'].isin(combs[c]['neg'] + combs[c]['pos'])]
    #     tdf['label'] = np.where(tdf['orig_fibrosis'].isin(combs[c]['pos']), 1, 0)
        
    #     for i in range(0, num_trials): 
    #         print('dk-comb-iter: %s-%s-%s' % (dk, c, i))

    #         tdf_i = tdf.sample(n=len(tdf), random_state=i, replace=True)
            
    #         ENS_auc_i = roc_auc_score(tdf_i['label'], tdf_i['ENS'])
    #         results.append({'comb': c, 'iter': i, 'alg': 'ENS', 'auroc': ENS_auc_i})
            
    #         for bl in rel_baselines[dk]:
    #             bl_auc_i = roc_auc_score(tdf_i['label'], tdf_i[bl])    
    #             results.append({'comb': c, 'iter': i, 'alg': bl, 'auroc': bl_auc_i})

    # result_df = pd.DataFrame.from_records(results)
    # result_df.sort_values(by=['comb', 'alg', 'iter'], inplace=True)
    # 


# Step 1. Load the correct dataframe. (DONE)
# Step 2. Get the combinations I need. (DONE)
# Step 3. For that combination, filter the dataset to the appropriate fibrosis stages (DONE)
# Step 4. For that combintation and dataset, resample 1k times. (DONE)
# Step 5. For that combination and dataset and sample iteration, calculate the AUROC of
# the relevant methods 
# Step 6. Save the AUCS of each method under each definition of advanced vs. non-advanced fibrosis to a CSV file, so that I can then calculate 95% CIs 


