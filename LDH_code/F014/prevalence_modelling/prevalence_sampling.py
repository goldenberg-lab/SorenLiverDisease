import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold

def get_confusion_matrix(label, preds):
    cm = pd.DataFrame({'pred': preds, 'label': label})
    cm = cm.loc[cm['pred']!= 0.5]    
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
        
    return {'sens': sens, 'spec': spec}     

folder_path = r'C:\Users\Darth\Desktop\Thesis\Code\Lancet Digital Health Code\Prevalence Modelling\prevalence_modelling_data.xlsx'

num_trials = 1000
data = pd.read_excel(folder_path)
negatives = data.loc[data['Fibrosis'] == 0]
positives = data.loc[data['Fibrosis'] == 1]

APRI_results = []
FIB4_results = []
ENS3_results = []
ENS3e_results = []

for num_pos in range(10,60,10): # Prevalence as a number from 10 to 50 out of 10000

    APRI_sens = []
    APRI_spec = []
    APRI_det = []  
    
    FIB4_sens = []
    FIB4_spec = []
    FIB4_det = []

    ENS3_sens = []
    ENS3_spec = []
    ENS3_det = []
    
    ENS3e_sens = []
    ENS3e_spec = []
    ENS3e_det = []
    
    for i in range(0,num_trials):
        num_neg = 10000 - num_pos
        prev = (100*num_pos/num_neg)
        print('Trial %d - Prevalence=%0.1f%%' % (i, prev))
        
        neg = negatives.sample(n=num_neg, replace=True, random_state=i)
        pos = positives.sample(n=num_pos, replace=True, random_state=i)
        
        df = pd.concat([pos, neg])
        len_df = len(df)
        
        APRI_per_det = np.sum(np.where(df['APRI(1, 2)preds'] != 0.5,1,0))/len_df
        FIB4_per_det = np.sum(np.where(df['FIB4(1.45 , 3.25)_preds'] != 0.5,1,0))/len_df
        ENS3_per_det = np.sum(np.where(df['ENS2(0.25 , 0.45)_preds'] != 0.5,1,0))/len_df
        ENS3e_per_det = np.sum(np.where(df['ENS2(0.525 , 0.7)_preds'] != 0.5,1,0))/len_df
        
        # Why could I not just create 3 temporary datasets and exclude the results?
        APRI_df = df.loc[df['APRI(1, 2)preds'] != 0.5][['reckey_enc', 'Fibrosis', 'APRI(1, 2)preds']]
        FIB4_df = df.loc[df['FIB4(1.45 , 3.25)_preds'] != 0.5][['reckey_enc', 'Fibrosis', 'FIB4(1.45 , 3.25)_preds']]
        ENS3_df = df.loc[df['ENS2(0.25 , 0.45)_preds'] != 0.5][['reckey_enc', 'Fibrosis', 'ENS2(0.25 , 0.45)_preds']]
        ENS3e_df = df.loc[df['ENS2(0.525 , 0.7)_preds'] != 0.5][['reckey_enc', 'Fibrosis', 'ENS2(0.525 , 0.7)_preds']]

        APRI_cm = get_confusion_matrix(APRI_df['Fibrosis'], APRI_df['APRI(1, 2)preds'])
        FIB4_cm = get_confusion_matrix(FIB4_df['Fibrosis'], FIB4_df['FIB4(1.45 , 3.25)_preds'])
        ENS3_cm = get_confusion_matrix(ENS3_df['Fibrosis'], ENS3_df['ENS2(0.25 , 0.45)_preds'])
        ENS3e_cm = get_confusion_matrix(ENS3e_df['Fibrosis'], ENS3e_df['ENS2(0.525 , 0.7)_preds'])
        
        APRI_pm = get_performance_metrics(APRI_cm)
        FIB4_pm = get_performance_metrics(FIB4_cm)
        ENS3_pm = get_performance_metrics(ENS3_cm)
        ENS3e_pm = get_performance_metrics(ENS3e_cm)
        
        APRI_det.append(APRI_per_det)
        APRI_sens.append(APRI_pm['sens'])
        APRI_spec.append(APRI_pm['spec'])
        
        FIB4_det.append(FIB4_per_det)
        FIB4_sens.append(FIB4_pm['sens'])
        FIB4_spec.append(FIB4_pm['spec'])
        
        ENS3_det.append(ENS3_per_det)
        ENS3_sens.append(ENS3_pm['sens'])
        ENS3_spec.append(ENS3_pm['spec'])      
        
        ENS3e_det.append(ENS3e_per_det)
        ENS3e_sens.append(ENS3e_pm['sens'])
        ENS3e_spec.append(ENS3e_pm['spec'])      
                
    APRI_results.append({'prev': prev, 'sens': np.mean(APRI_sens), 'spec': np.mean(APRI_spec), 'det': np.mean(APRI_det), 'sens_data': APRI_sens, 'spec_data': APRI_spec, 'det_data': APRI_det})
    FIB4_results.append({'prev': prev, 'sens': np.mean(FIB4_sens), 'spec': np.mean(FIB4_spec), 'det': np.mean(FIB4_det), 'sens_data': FIB4_sens, 'spec_data': FIB4_spec, 'det_data': FIB4_det})
    ENS3_results.append({'prev': prev, 'sens': np.mean(ENS3_sens), 'spec': np.mean(ENS3_spec), 'det': np.mean(ENS3_det), 'sens_data': ENS3_sens, 'spec_data': ENS3_spec, 'det_data': ENS3_det})
    ENS3e_results.append({'prev': prev, 'sens': np.mean(ENS3e_sens), 'spec': np.mean(ENS3e_spec), 'det': np.mean(ENS3e_det), 'sens_data': ENS3e_sens, 'spec_data': ENS3e_spec, 'det_data': ENS3e_det})
    
APRI_df = pd.DataFrame.from_records(APRI_results)
FIB4_df = pd.DataFrame.from_records(FIB4_results)
ENS3_df = pd.DataFrame.from_records(ENS3_results)
ENS3e_df = pd.DataFrame.from_records(ENS3e_results)

outpath = 'C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\Prevalence Modelling\\'

APRI_df.to_excel(outpath + 'APRI_prev.xlsx')
FIB4_df.to_excel(outpath + 'FIB4_prev.xlsx')
ENS3_df.to_excel(outpath + 'ENS2_prev.xlsx')
ENS3e_df.to_excel(outpath + 'ENS2e_prev.xlsx')

# Okay. What was the confusing part? 
# APRI sensitivities have indeterminate values. What is the correct way to calculate that?
# If the sensitivity/specificity is undefined, I should not include it in the reult



    