# Objective: Rerun code with F01234 models on new data 

# Step 1. (DONE) Get the correct version of sklearn (Models trained on version 0.23.2)
# Step 2. (DONE) Load in the models and the scaling parameters 
# Step 3. (DONE) Load in the old datasets 
# Step 4. Make the predictions on the 5 datasets 
# Step 5. Compare old and new predictions and report to Anna/Mamatha 
# Step 6. Re-plan once I reach this stage. 


# Step 2. Apply the correct imputation and scaling to the dataset 
# Step 3. Run the ML algorithms on it 
# Step 4. Calculate performance metrics, aggregate, and present 
# Step 5. Save the results externally so I can do re-sampling and compare to

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

pd.options.mode.chained_assignment = None  # default='warn'

class MLA():
    def __init__(self, name, threshold, probs, label): 
        self.name = name 
        self.preds = (probs >= threshold)*4
        self.probs = probs
        self.label = label
        self.bin_label = (label/4).astype(int)
        self.cm = get_confusion_matrix(self.label, self.preds)
        self.my_auroc, self.my_auprc, self.df_auroc, self.df_auprc,  = my_auc_prob(label, probs)
        self.pm = get_performance_metrics(self.cm)
        self.percent_det = 1
        self.uthresh = threshold
        self.lthresh = threshold
        self.pm['auroc'] = self.my_auroc
        self.pm['auprc'] = self.my_auprc
        self.pm['per_det'] = self.percent_det
        self.pm['name'] = name + '(' + str(threshold) + ' , ' + str(threshold) + ')'
        
        self.color = None


def get_confusion_matrix(label, preds):
    cm = pd.DataFrame({'pred': preds, 'label': label})
    cm['TN'] = np.where((cm['pred'] == 0) & (cm['label'] == 0), 1, 0)
    cm['FP'] = np.where((cm['pred'] == 4) & (cm['label'] == 0), 1, 0)
    cm['FN'] = np.where((cm['pred'] == 0) & (cm['label'] == 4), 1, 0)
    cm['TP'] = np.where((cm['pred'] == 4) & (cm['label'] == 4), 1, 0)
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
    acc = (TP + TN)/(TP + TN + FP + FN)
    
    pm = {'sens': sens, 'spec': spec, 'ppv': ppv, 
          'npv': npv, 'acc': acc}
    
    return pm 

def my_auc_prob(label, probs): 
    results = []
    
    new_probs = np.unique(probs.to_numpy(copy=True))
    mid_probs = new_probs[:-1] + np.diff(new_probs)/2
    mid_probs = np.unique(np.append(mid_probs, [0,1]))
    
    for t in mid_probs: 
        cm = get_confusion_matrix(label, (probs >= t)*4)
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
    
    auprc_df = df[['thresh', 'tpr', 'prc']]
    auprc_tempf = auprc_df.drop_duplicates(subset=['tpr'], keep='first')
    auprc_templ = auprc_df.drop_duplicates(subset=['tpr'], keep='last')
    auprc_df_2 = pd.concat([auprc_tempf, auprc_templ])
    auprc_df_2 = auprc_df_2.loc[~auprc_df_2['prc'].isnull()]
    auprc_df_2.sort_values(by=['thresh', 'tpr', 'prc'], inplace=True)
    auprc_df_2.reset_index(drop=True, inplace=True)
    l2 = len(auprc_df_2)
    auprc_df_2.loc[l2] = [1, 0, auprc_df_2.iloc[l2-1]['prc']]
    
    auroc = auc(auroc_df_2['fpr'], auroc_df_2['tpr'])
    auprc = auc(auprc_df_2['tpr'], auprc_df_2['prc'])
    
    return auroc, auprc, auroc_df_2, auprc_df_2

def my_auc_non_prob(label, values): 
    
    results = []
    new_vals = np.unique(values.to_numpy(copy=True))
    mid_vals = new_vals[:-1] + np.diff(new_vals)/2
    mid_vals = np.unique(np.append(mid_vals, [-1000,1000]))

    for t in mid_vals: 
        cm = get_confusion_matrix(label, (values >= t)*4)
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
    
    auprc_df = df[['thresh', 'tpr', 'prc']]
    auprc_tempf = auprc_df.drop_duplicates(subset=['tpr'], keep='first')
    auprc_templ = auprc_df.drop_duplicates(subset=['tpr'], keep='last')
    auprc_df_2 = pd.concat([auprc_tempf, auprc_templ])
    auprc_df_2 = auprc_df_2.loc[~auprc_df_2['prc'].isnull()]
    auprc_df_2.sort_values(by=['thresh', 'tpr', 'prc'], inplace=True)
    auprc_df_2.reset_index(drop=True, inplace=True)
    l2 = len(auprc_df_2)
    auprc_df_2.loc[l2] = [1, 0, auprc_df_2.iloc[l2-1]['prc']]
    
    auroc = auc(auroc_df_2['fpr'], auroc_df_2['tpr'])
    auprc = auc(auprc_df_2['tpr'], auprc_df_2['prc'])
    
    return auroc, auprc, auroc_df_2, auprc_df_2

def write_performance_metrics(algs, key, sql_dict):
    col_heads = []
    sens_row =  []
    spec_row = []
    ppv_row = []
    npv_row = []
    acc_row = []
    AUROC_row = []
    AUPRC_row = []
    det_row = []
    
    for alg in algs: 
        col_heads.append('%s\n%0.3f - %0.3f' % (alg.name, alg.lthresh, alg.uthresh))
        sens_row.append('%0.1f' % (100*alg.pm['sens']))
        spec_row.append('%0.1f' % (100*alg.pm['spec']))
        ppv_row.append('%0.1f' % (100*alg.pm['ppv']))
        npv_row.append('%0.1f' % (100*alg.pm['npv']))
        acc_row.append('%0.1f' % (100*alg.pm['acc']))
        AUROC_row.append('%0.1f' % (100*alg.my_auroc))
        AUPRC_row.append('%0.1f' % (100*alg.my_auprc))
        det_row.append('%0.1f' % (100*alg.percent_det))
    
    table = BeautifulTable(maxwidth=200)
    table.columns.header = col_heads
    table.rows.header = ['sens', 'spec', 'ppv', 'npv', 'acc', 'AUROC', 'AUPRC', '%det']    
    table.rows[0] = sens_row
    table.rows[1] = spec_row
    table.rows[2] = ppv_row
    table.rows[3] = npv_row
    table.rows[4] = acc_row
    table.rows[5] = AUROC_row
    table.rows[6] = AUPRC_row
    table.rows[7] = det_row
    print(table)

thesis_path = '/Volumes/Chamber of Secrets/Thesis'
algdir = '/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/models'
os.chdir(algdir)

datakey = 'Expert'
datakey = 'Toronto'
datakey = 'McGill'
datakey = 'NAFL'
datakey = 'TE'


features = [ 'Age', 'Sex', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 
            'Creatinine', 'INR', 'BMI', 'Platelets', 'Diabetes',  
            'Fibrosis', 'patientID', 'bx_date', 'reckey_enc', 'TE', 'TE_date']
feat_dict = {'age': 0, 'sex': 1, 'albumin': 2, 'alp': 3, 'alt': 4, 'ast': 5, 'bilirubin': 6, 
             'creatinine': 7, 'inr': 8, 'bmi': 9, 'platelets': 10, 'diabetes': 11}

#predictors = ['Sex', 'Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'Creatinine', 'INR', 'Platelets', 'BMI', 'Diabetes']
sql_dict = {'Toronto': 'SELECT * FROM _TorontoHoldOut30 WHERE missingness <= 3',
            'Toronto-NAFL': 'SELECT * FROM _TorontoHoldOut30 WHERE missingness <= 3 AND NAFL=1',
            'Toronto-TE': 'SELECT * FROM _TorontoHoldOut30 WHERE missingness <= 3 AND TE IS NOT NULL',
            'Combined': 'SELECT * FROM _Toronto_McGill_non_TE WHERE missingness <= 3 AND Fibrosis IS NOT NULL AND NEOPLASM=0',
            'NAFL': 'SELECT * FROM _Toronto_McGill_non_TE WHERE missingness <= 3 AND Fibrosis IS NOT NULL AND NEOPLASM=0 AND NAFL=1',
            'TE': 'SELECT * FROM _Toronto_McGill_TE WHERE missingness <= 3 AND Fibrosis IS NOT NULL AND NEOPLASM=0 AND TE IS NOT NULL',
            'Toronto-McGill-TE-NAFLD': 'SELECT * FROM _Toronto_McGill_TE WHERE missingness <= 3 AND Fibrosis IS NOT NULL AND NEOPLASM=0 AND TE IS NOT NULL AND NAFL=1',
            'McGill': 'SELECT * FROM _McGillData WHERE NEOPLASM=0 AND missingness <= 3 AND Fibrosis IS NOT NULL',
            'McGill-NAFL': 'SELECT * FROM _McGillData WHERE NEOPLASM=0 AND missingness <= 3 AND Fibrosis IS NOT NULL AND NAFL=1',
            'McGill-TE': 'SELECT * FROM _McGillData_August2020 WHERE NEOPLASM=0 AND missingness <= 3 AND Fibrosis IS NOT NULL AND TE IS NOT NULL',
            'Expert': 'SELECT * FROM _ExpertPredsCombined WHERE bx_date IS NOT NULL AND missingness <= 3'}

datapath = thesis_path + '/Data/Sept2020FinalDatasets/'
path_dict = {'Toronto': datapath + 'Toronto.xlsx',
              'Expert': datapath + 'Expert.xlsx',
              'McGill': datapath + 'McGill.xlsx',
              'NAFL': datapath + 'NAFLD.xlsx', 
              'TE': datapath + 'TE.xlsx', 
              'Combined': datapath + 'Combined.xlsx'} 

alg_name = {'Toronto': 'ENS2b',
            'Expert': 'ENS2c',
            'McGill': 'ENS2b', 
            'NAFL': 'ENS2b', 
            'TE': 'ENS2d',
            'Combined': 'ENS_NULL'}

dataset = pd.read_excel(path_dict[datakey], index_col=None, engine='openpyxl')
dataset['orig_fibrosis'] = dataset['Fibrosis']
dataset = dataset[features + ['orig_fibrosis']]
dataset = dataset.loc[dataset['Fibrosis'].isin([0,1,2,3,4])]
dataset['Fibrosis'] = np.where(dataset['Fibrosis'] <= 3, 0, 4)


with open(algdir + '/trained_models_imp_scl.pkl', 'rb') as file: 
    unpickler = pickle.Unpickler(file)
    outputs = unpickler.load() 

scl_info = outputs['scl_info']
SVM_model = outputs['svm_model']
RFC_model = outputs['rfc_model']
GBC_model = outputs['gbc_model']
LOG_model = outputs['log_model']
MLP_model = outputs['mlp_model']

# All models use all features, so I don't need to extract specifics 
SVMF = [feat_dict[f] for f in outputs['svmp']['features']]
RFCF = [feat_dict[f] for f in outputs['rfcp']['features']]
GBCF = [feat_dict[f] for f in outputs['gbcp']['features']]
LOGF = [feat_dict[f] for f in outputs['logp']['features']]
MLPF = [feat_dict[f] for f in outputs['mlpp']['features']]



# Impute and scale data
X_test_unimp = dataset[[ 'Age', 'Sex', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 
            'Creatinine', 'INR', 'BMI', 'Platelets', 'Diabetes']]
X_test_imp = X_test_unimp.copy()
for col in X_test_unimp.columns.tolist(): 
    X_test_imp[col] = np.where(X_test_unimp[col].isnull(), scl_info[col.lower() + '_mean'], X_test_unimp[col])

X_test_imp_scl = X_test_imp.copy()
for col in X_test_imp.columns.tolist(): 
    X_test_imp_scl[col] = (X_test_imp_scl[col] - scl_info[col.lower() + '_mean'])/scl_info[col.lower() + '_std']

xti = X_test_imp.values
xts = X_test_imp_scl.values

# 
# RFC: Imputed 
# GBC: Imputed 
# LOG: Scaled 
# MLP: Scaled 

# Make Predictions and store as part of the dataframe 
dataset['SVM_probs'] = SVM_model.predict_proba(xts)[:,1]    # SVM: Trained on Scaled Values
dataset['RFC_probs'] = RFC_model.predict_proba(xti)[:,1]    # RFC: Trained on Unscaled Values
dataset['GBC_probs'] = GBC_model.predict_proba(xti)[:,1]    # GBC: Trained on Unscaled Values
dataset['LOG_probs'] = LOG_model.predict_proba(xts)[:,1]    # LOG: Trained on Scaled Values 
dataset['MLP_probs'] = MLP_model.predict_proba(xts)[:,1]    # MLP: Trained on Scaled Values 
dataset['APRI_vals'] = (100/35)*X_test_imp['AST']/X_test_imp['Platelets']
dataset['FIB4_vals'] = X_test_imp['Age']*X_test_imp['AST']/(X_test_imp['Platelets']*(X_test_imp['ALT']**0.5))
dataset['NAFL_vals'] = -1.675 + 0.037*X_test_imp['Age'] + 0.094*X_test_imp['BMI'] + 1.13*X_test_imp['Diabetes'] + 0.99*X_test_imp['AST']/X_test_imp['ALT'] - 0.013*X_test_imp['Platelets'] - 0.66*X_test_imp['Albumin']/10
dataset['TE_vals'] = dataset['TE']
dataset['ENS1_probs'] = dataset[['SVM_probs', 'RFC_probs', 'GBC_probs', 'LOG_probs', 'MLP_probs']].sum(axis=1)/5
dataset['ENS3_probs'] = dataset['ENS1_probs']

# Try comparing only to non-indet APRI 
#dataset = dataset.loc[~dataset['APRI_vals'].between(1.00, 2.00)]
#dataset = dataset.loc[~dataset['FIB4_vals'].between(1.45, 3.25)]


# Create objects which calculate the relevant performance metrics 
SVM = MLA('SVM', 0.4, dataset['SVM_probs'], dataset['Fibrosis'])
RFC = MLA('RFC', 0.4, dataset['RFC_probs'], dataset['Fibrosis'])
GBC = MLA('GBC', 0.4, dataset['GBC_probs'], dataset['Fibrosis'])
LOG = MLA('LOG', 0.4, dataset['LOG_probs'], dataset['Fibrosis'])
MLP = MLA('MLP', 0.4, dataset['MLP_probs'], dataset['Fibrosis'])
ENS = MLA('ENS', 0.55, dataset['ENS1_probs'], dataset['Fibrosis'])
ENS.color = 'red'
# ENS1_FIB4 = MLA('ENS1', 0.5, dataset['ENS1_probs'], dataset['Fibrosis'])
# ENS1_TE = MLA('ENS1', 0.465, dataset['ENS1_probs'], dataset['Fibrosis'])
# ENS1_EXP = MLA('ENS1', 0.6, dataset['ENS1_probs'], dataset['Fibrosis'])

#ENS3 = ENS('ENS3', 0.25, 0.45, dataset[['ENS3_probs', 'Fibrosis']], APRI.probs, FIB4.probs)
#ENS3b = ENS('ENS3', etd[datakey][1][0], etd[datakey][1][1], dataset[['ENS3_probs', 'Fibrosis']], APRI.probs, FIB4.probs)


#ENS3c = ENS('ENS3', 0.465, 0.465, dataset[['ENS3_probs', 'Fibrosis']], APRI.probs, FIB4.probs)
#ENS3d = ENS('ENS3', 0.6, 0.6, dataset[['ENS3_probs', 'Fibrosis']], APRI.probs, FIB4.probs)

#ENS3_FIB4 = ENS('ENS3', 0.525, 0.7, dataset[['ENS3_probs', 'Fibrosis']], APRI.probs, FIB4.probs)


# ENS3.name = 'ENS2(0.25 , 0.45)'
# ENS3.color = colours[ENS3.name]

#ENS3b.name = alg_name[datakey] + '(' + str(etd[datakey][1][0]) + ' , ' + str(etd[datakey][1][1]) + ')'
#ENS3b.color = colours[ENS3b.name]

algs = [ENS]


F012 = len(dataset.loc[dataset['Fibrosis'] == 0])
F34 = len(dataset) - F012
print('\n\n' + sql_dict[datakey] + '\n')
print('F012/F34 Split: %d/%d\n' % (F012,F34))
write_performance_metrics(algs, datakey, sql_dict)



# # Remaining tasks: Add random chance line to ROC and PR Curves 
# # Get the right colours and names for the algorithms 
# # Add a table underneath the plots with the AUROC and AUPRC information 

# ############### START OF DISTRIBUTION POINT-PERFORMANCE CODE #########
# # Step 1. Run algorithm
# # Step 2. Get the relevant performance metrics for each algo 
# # Step 3. Save to the same place as the distribution raw data, can be done as a pdf 
# # a = []
# # for alg in algs: 
# #     a.append(alg.pm)
# # b = pd.DataFrame.from_records(a)

# # path = 'C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\ConfInterval_Distribution_Modelling\\Distributions\\' + datakey + '-point.xlsx'
# # b.to_excel(path)

    
# ############### START OF DISTRIBUTION MODELLING CODE #################
# # Step 1. Get all relevant columns from the dataset
# # Step 2. Save a copy of the index + the predictions 
# # Step 3. Load the dataset in a new script 
# # Step 4. Sample that dataset with replacement 1000 times 
# # Step 5. For each sample, calculte performance metrics for the relevant algorithms
# #         at the right thresholds           
# # Step 6. Take the average over all 1000 trials for each subset of the data 
# # Step 7. Use these averages to plot performance and generate distributions 
# # Step 8. Some manual work needed for each dataset, start with what you have. 
# # columns = ['Fibrosis', 'APRI_vals', 'FIB4_vals', 'NAFL_vals', 'ENS3_probs', 'APRI_probs', 'FIB4_probs', 'NAFL_probs']
# # dist_df = dataset[columns]
# # if (datakey == 'Expert'):
# #     dist_df = dist_df.merge(expert_dataset[['EXP_probs']], left_index=True, right_index=True, how='left')
# # if (datakey == 'TE'):
# #     dist_df = dist_df.merge(dataset[['TE_vals']], left_index=True, right_index=True, how='left')

# # outpath = 'C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\ConfInterval_Distribution_Modelling\\Predictions\\' + datakey + '.xlsx'
# # dist_df.to_excel(outpath)

# ############### START OF FEATURE IMPORTANCE CODE #####################

# # num_trials = 10
# # temp_dataset = pd.DataFrame(X_test_imp_scl[:,:12], columns=predictors)
# # temp_dataset = temp_dataset.merge(dataset['Fibrosis'], left_index=True, right_index=True, how='left')

# # ft_imp_res = []

# # for feat in predictors:
    
# #     SVM_a = []
# #     RFC_a = []
# #     GBC_a = []
# #     LOG_a = []
# #     MLP_a = []
# #     ENS1_a = []
# #     ENS3_a = []
    
# #     for nt in range(0, num_trials):       
# #         print('Feat: %s , trial: %d' % (feat, nt))
# #         temp_df = temp_dataset.drop(columns={feat})
# #         feat_df = temp_dataset[[feat]].sample(frac=1, random_state=nt).reset_index(drop=True)
# #         temp_df = temp_df.merge(feat_df, left_index=True, right_index=True, how='left')
# #         temp_df = temp_df[predictors + ['Fibrosis']]
        
# #         temp_df_np = temp_df.iloc[:,:12].values
        
# #         # Make predictions for each algorithm 
# #         temp_df['SVM_probs'] = SVM_model.predict_proba(temp_df_np[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]])[:,1]
# #         temp_df['RFC_probs'] = RFC_model.predict_proba(temp_df_np[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])[:,1]
# #         temp_df['GBC_probs'] = GBC_model.predict_proba(temp_df_np[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]])[:,1]
# #         temp_df['LOG_probs'] = LOG_model.predict_proba(temp_df_np[:,[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]])[:,1]
# #         temp_df['MLP_probs'] = MLP_model.predict_proba(temp_df_np[:,[0, 1, 2, 3, 4, 5, 8, 9, 10, 11]])[:,1]
# #         temp_df['APRI_vals'] = (100/35)*temp_df['AST']/temp_df['Platelets']
# #         temp_df['FIB4_vals'] = temp_df['Age']*temp_df['AST']/(temp_df['Platelets']*(temp_df['ALT']**0.5))
# #         temp_df['ENS1_probs'] = temp_df[['SVM_probs', 'RFC_probs', 'GBC_probs', 'LOG_probs', 'MLP_probs']].sum(axis=1)/5
# #         temp_df['ENS3_probs'] = temp_df['ENS1_probs']
        
# #         temp_APRI = BIO('APRI', 1, 2, temp_df[['APRI_vals', 'Fibrosis']])
# #         temp_FIB4 = BIO('FIB4', 1.45, 3.25, temp_df[['FIB4_vals', 'Fibrosis']])

# #         SVM_a.append(MLA('SVM', 0.4, temp_df['SVM_probs'], temp_df['Fibrosis']).my_auroc)
# #         RFC_a.append(MLA('RFC', 0.4, temp_df['RFC_probs'], temp_df['Fibrosis']).my_auroc)
# #         GBC_a.append(MLA('GBC', 0.4, temp_df['GBC_probs'], temp_df['Fibrosis']).my_auroc)
# #         LOG_a.append(MLA('LOG', 0.4, temp_df['LOG_probs'], temp_df['Fibrosis']).my_auroc)
# #         MLP_a.append(MLA('MLP', 0.4, temp_df['MLP_probs'], temp_df['Fibrosis']).my_auroc)
# #         ENS1_a.append(ENS('ENS1', 0.45, 0.45, temp_df[['ENS1_probs', 'Fibrosis']], temp_APRI.probs, temp_FIB4.probs).my_auroc)
# #         ENS3_a.append(ENS('ENS3', 0.25, 0.45, temp_df[['ENS3_probs', 'Fibrosis']], temp_APRI.probs, temp_FIB4.probs).my_auroc)

# #     SVM_m = np.mean(SVM_a)
# #     RFC_m = np.mean(RFC_a)
# #     GBC_m = np.mean(GBC_a)
# #     LOG_m = np.mean(LOG_a)
# #     MLP_m = np.mean(MLP_a)
# #     ENS1_m = np.mean(ENS1_a)
# #     ENS3_m = np.mean(ENS3_a)
    
# #     ft_imp_res.append({'feat': feat,
# #                        'SVM_diff': SVM.my_auroc - SVM_m,
# #                        'RFC_diff': RFC.my_auroc - RFC_m,
# #                        'GBC_diff': GBC.my_auroc - GBC_m, 
# #                        'LOG_diff': LOG.my_auroc - LOG_m,
# #                        'ANN_diff': MLP.my_auroc - MLP_m,
# #                        'ENS1_diff': ENS1.my_auroc - ENS1_m,
# #                        'ENS3_25-45_diff': ENS3.my_auroc - ENS3_m})

# # feature_importance_df = pd.DataFrame.from_records(ft_imp_res)
# # path = 'C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\' + datakey +'_feature_importances.xlsx'
# # feature_importance_df.to_excel(path)

# ############## END OF FEATURE IMPORTANCE CODE ###########################

# # Preparing results for prevalence modelling 
# # prev_df = dataset[['reckey_enc', 'Fibrosis']]
# # prev_df['Fibrosis']/=4
# # prev_df['APRI(1, 2)preds'] = APRI.probs
# # prev_df['FIB4(1.45 , 3.25)_preds'] = FIB4.probs
# # prev_df['ENS2(0.25 , 0.45)_preds'] = ENS3.all_probs_df['all_probs']
# # prev_df['ENS2(0.525 , 0.7)_preds'] = ENS3_FIB4.all_probs_df['all_probs']

# # prev_df.to_excel(r'C:\Users\Darth\Desktop\Thesis\Code\Lancet Digital Health Code\Prevalence Modelling\prevalence_modelling_data.xlsx', index=0)

# #PRC plots 

# # Okay. 
# # 1) Import the new data 
# # 2) Filter the new data to relevant columns 
# # 3) Filter the old data to relevant columns 
# # 4) Merge, export, and compare 

# # new_uhn = pd.read_excel(r'C:\Users\Darth\Desktop\Thesis\Data\August 2020 Data\UHN Test Set Patients.xlsx')
# # new_uhn = new_uhn[['MRN', 'age', 'BMI', 'IFG (DM)', 'AST', 'ALT', 'Plt', 'Alb (g/l)', 'NFS']]
# # new_uhn['MRN'] = new_uhn['MRN'].astype(str)
# # new_uhn.rename(columns={'age': 'Age_l', 'BMI': 'BMI_l', 'IFG (DM)': 'Diabetes_l', 
# #                         'AST': 'AST_l', 'ALT': 'ALT_l', 'Plt': 'Platelets_l',
# #                         'Alb (g/l)': 'Albumin_l', 'NFS': 'NAFLD_l' }, inplace=True)
# # new_uhn.sort_values(by=['MRN'], ascending=True, inplace=True)
# # new_uhn = new_uhn.loc[~new_uhn['NAFLD_l'].isnull()]

# # new_dat = dataset[['reckey_enc', 'Age', 'BMI', 'Diabetes', 'AST', 'ALT', 'Platelets', 'Albumin', 'NAFL_vals']]
# # new_dat.rename(columns={'reckey_enc': 'MRN'}, inplace=True)
# # new_dat['MRN'] = new_dat['MRN'].astype(str)
# # new_dat.sort_values(by=['MRN'], ascending=True, inplace=True)
# # new_dat = new_dat.loc[new_dat['MRN'].isin(new_uhn['MRN'])]
# #new_dat = new_dat.merge(new_uhn, left_on=['MRN'], right_on=['MRN'], how='left')

# # precs_py, recs_py, threshs = precision_recall_curve((dataset['Fibrosis']/4).astype(int), dataset['GBC_probs'])
# # py_auprc_df = pd.DataFrame(data=[threshs, recs_py,  precs_py], index=['thresh', 'tpr', 'prc']).transpose()

# # fprs_py, tprs_py, threshs = roc_curve((dataset['Fibrosis']/4).astype(int), dataset['GBC_probs'])
# # py_auroc_df = pd.DataFrame(data=[threshs, fprs_py, tprs_py], index=['thresh', 'fpr', 'tpr']).transpose()

# # Okay. It is not clear why I am getting different values for AURPC. 
# # I need to investigate how Python calulates the precision recall curve under the surface
# # Then, I need to compare that with my algorithm. 
# # Actually do the analysis Mamatha requested

# # plt.plot(auroc_df['thresh'], auroc_df['fpr'], 'r-', label='manual_thresh v. fpr')
# # plt.plot(py_auroc_df['thresh'], py_auroc_df['fpr'], 'g-', label='python_thresh v. fpr')
# # plt.grid(True)
# # plt.xlabel('Threshold')
# # plt.ylabel('False Positive Rate')
# # plt.title('False Positive Rate vs. Threshold, Python vs. Manual')
# # plt.legend()

# # plt.figure()
# # plt.plot(auroc_df['thresh'], auroc_df['tpr'], 'r-', label='manual_thresh v. tpr')
# # plt.plot(py_auroc_df['thresh'], py_auroc_df['tpr'], 'g-', label='python_thresh v. tpr')
# # plt.grid(True)
# # plt.xlabel('Threshold')
# # plt.ylabel('Sensitvity')
# # plt.title('Sensitivity vs. Threshold, Python vs. Manual')
# # plt.legend()

# # plt.figure()
# # plt.plot(auroc_df['fpr'], auroc_df['tpr'], 'rx-', label='my_roc_curve')
# # plt.plot(py_auroc_df['fpr'], py_auroc_df['tpr'], 'go-', label='py_roc_curve')
# # plt.grid(True)
# # plt.xlabel('1 - Specificity')
# # plt.ylabel('Sensitvity')
# # plt.title('Sensitivity vs. False Positive Rate, Python vs. Manual')
# # plt.legend()

# # plt.figure()
# # plt.plot(py_auroc_df['fpr'], py_auroc_df['tpr'], 'go-', label='py_roc_curve')
# # plt.plot(auroc_df_2['fpr'], auroc_df_2['tpr'], 'rx-', label='my_roc_curve')
# # plt.grid(True)
# # plt.xlabel('1 - Specificity')
# # plt.ylabel('Sensitvity')
# # plt.title('Sensitivity vs. False Positive Rate, Python vs. Manual 2')
# # plt.legend()


# # plt.plot(auprc_df['thresh'], auprc_df['prc'], 'r-', label='manual_thresh.v.prec')
# # plt.plot(py_df['thresh'], py_df['prc'], 'g-', label='python_thresh.v.prec')
# # plt.grid(True)
# # plt.xlabel('Threshold')
# # plt.ylabel('Precision')
# # plt.title('Precision v. Threshold, Python vs. Manual')
# # plt.legend()

# # plt.figure()
# # plt.plot(auprc_df['thresh'], auprc_df['tpr'], 'r-', label='manual_thresh.v.rec')
# # plt.plot(py_df['thresh'], py_df['tpr'], 'g-', label='python_thresh.v.rec')
# # plt.grid(True)
# # plt.xlabel('Threshold')
# # plt.ylabel('Recall')
# # plt.title('Recall v. Threshold, Python vs. Manual')
# # plt.legend()
# # plt.show()

# # plt.figure() 
# # plt.plot(py_df['tpr'], py_df['prc'], 'go-', label='py_prc_curve')
# # plt.plot(auprc_df['tpr'], auprc_df['prc'], 'ro-', label='my_prc_curve')
# # plt.grid(True)
# # plt.xlabel('Recall')
# # plt.ylabel('Precision')
# # plt.title('Precision Recall Curve, Python vs. Manual')
# # plt.legend()
# # plt.show()

# # plt.figure() 
# # plt.plot(py_df['tpr'], py_df['prc'], 'go-', label='py_prc_curve')
# # plt.plot(auprc_df_2['tpr'], auprc_df_2['prc'], 'ro-', label='my_prc_curve')
# # plt.grid(True)
# # plt.xlabel('Recall')
# # plt.ylabel('Precision')
# # plt.title('Precision Recall Curve, Python vs. Manual 2')
# # plt.legend()
# # plt.show()

# # print(GBC.py_auprc)
# # print(my_auprc)

# # print(GBC.py_auprc - my_auprc)

# # algs = [SVM]
# # print('Reached here!')

# # for alg in algs: 
# #     print(alg.name)
# #     print('PY_AUROC: %0.6f' % (alg.py_auroc))
# #     print('MY_AUROC: %0.6f' % (alg.my_auroc))
# #     print('PY_AUPRC: %0.6f' % (alg.py_auprc))
# #     print('MY_AUPRC: %0.6f' % (alg.my_auprc))
# #     print('-----')        

# # Okay. I need to do more exploration on the Expert sets. It's important to keep these 

