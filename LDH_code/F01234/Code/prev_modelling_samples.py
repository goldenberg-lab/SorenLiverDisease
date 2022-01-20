import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold

def get_cm(label, preds):
    cm = pd.DataFrame({'pred': preds, 'label': label})
    cm['TN'] = np.where((cm['pred'] == 0) & (cm['label'] == 0), 1, 0)
    cm['FP'] = np.where((cm['pred'] == 1) & (cm['label'] == 0), 1, 0)
    cm['FN'] = np.where((cm['pred'] == 0) & (cm['label'] == 1), 1, 0)
    cm['TP'] = np.where((cm['pred'] == 1) & (cm['label'] == 1), 1, 0)
    cm['null'] = np.where(cm['pred'].isnull(), 1, 0)
    return cm.sum()
    
def get_pm(cm): 
    TP = cm['TP']
    TN = cm['TN']
    FP = cm['FP']
    FN = cm['FN']
    
    sens = TP/(TP + FN)
    spec = TN/(TN + FP)
    PPV = TP/(TP + FP)
    NPV = TN/(TN + FN)
        
    return {'sens': sens, 'spec': spec, 'ppv': PPV, 'npv': NPV}   

num_trials = 1000
data = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/predictions/Combined.csv', index_col=0)
data = data[['APRI_vals', 'APRI_probs', 'FIB4_vals', 'FIB4_probs', 'ENS1_probs', 'orig_fibrosis']]
data['ENS_TE_preds'] = np.where(data['ENS1_probs'] >= 0.465, 1, 0)
data['ENS_EXP_preds'] = np.where(data['ENS1_probs'] >= 0.6, 1, 0)
data['target'] = np.where(data['orig_fibrosis'] <= 2, 0, 1)

negatives = data.loc[data['target'] == 0]
positives = data.loc[data['target'] == 1]

res = []

for num_pos in range(10,60,10): # Prevalence as a number from 10 to 50 out of 10000
    num_neg = 10000 - num_pos
    prev = (100*num_pos/num_neg)
    
    for i in range(0,num_trials):
        print('%0.2f%% - %d' %(prev, i))
        neg = negatives.sample(n=num_neg, replace=True, random_state=i)
        pos = positives.sample(n=num_pos, replace=True, random_state=i)

        ALL_df = pd.concat([pos, neg])
        APRI_df = ALL_df.loc[ALL_df['APRI_probs'] != 0.5][['APRI_probs', 'target']]
        FIB4_df = ALL_df.loc[ALL_df['FIB4_probs'] != 0.5][['FIB4_probs', 'target']]
        
        len_ALL_df = len(ALL_df)
        len_APRI_df = len(APRI_df)
        len_FIB4_df = len(FIB4_df)
        
        APRI_per_det = 100*len_APRI_df/len_ALL_df
        FIB4_per_det = 100*len_FIB4_df/len_ALL_df
        ENS_TE_per_det = 100
        ENS_EXP_per_det = 100
        
        APRI_pm = get_pm(get_cm(APRI_df['target'], APRI_df['APRI_probs']))
        FIB4_pm = get_pm(get_cm(FIB4_df['target'], FIB4_df['FIB4_probs']))
        ENS_TE_pm = get_pm(get_cm(ALL_df['target'], ALL_df['ENS_TE_preds']))
        ENS_EXP_pm = get_pm(get_cm(ALL_df['target'], ALL_df['ENS_EXP_preds']))
        
        alg_dict = {'APRI': APRI_pm, 'FIB4': FIB4_pm, 'ENS_TE': ENS_TE_pm, 'ENS_EXP': ENS_EXP_pm}
        alg_pdet = {'APRI': APRI_per_det, 'FIB4': FIB4_per_det, 'ENS_TE': ENS_TE_per_det, 'ENS_EXP': ENS_EXP_per_det}
        
        for key in alg_dict.keys(): 
            alg = alg_dict[key]
            res_dict = {'prevalence': prev, 'trial': i, 'algorithm': key, 
                        'sens': alg['sens'], 
                        'spec': alg['spec'], 
                        'ppv': alg['ppv'], 
                        'npv': alg['npv'], 
                        'per_det': alg_pdet[key]}
            res.append(res_dict)
results = pd.DataFrame(res)
results.sort_values(by=['prevalence', 'algorithm', 'trial'], inplace=True)
results.to_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/prevalence modelling/pm_results.csv')
