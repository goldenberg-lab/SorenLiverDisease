import os
import pickle 
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score 

# Step 1. (DONE) Load the data 
# Step 2. (DONE) Rearrange features for each algorithm appropriately 
# Step 3. (DONE) Over 10 trials 
# Step 4. (DONE) Permute each feature 
# Step 5.   Calculate the AUROC for each algorithm 
# Step 6.   Store intermediate results in a dataframe. 

# Step 1. Loading the data 
thesis_path = '/Volumes/Chamber of Secrets/Thesis'
algdir = '/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/models'
os.chdir(algdir)

dk = 'Combined'
num_trials = 1000
features = [ 'Age', 'Sex', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 
            'Creatinine', 'INR', 'BMI', 'Platelets', 'Diabetes',  
            'Fibrosis', 'patientID', 'bx_date', 'reckey_enc', 'TE', 'TE_date']
feat_dict = {'Age': 0, 'Sex': 1, 'Albumin': 2, 'ALP': 3, 'ALT': 4, 'AST': 5, 'Bilirubin': 6, 
             'Creatinine': 7, 'INR': 8, 'BMI': 9, 'Platelets': 10, 'Diabetes': 11}


datapath = thesis_path + '/Data/Sept2020FinalDatasets/'
path_dict = {'Toronto': datapath + 'Toronto.xlsx',
              'Expert': datapath + 'Expert.xlsx',
              'McGill': datapath + 'McGill.xlsx',
              'NAFL': datapath + 'NAFLD.xlsx', 
              'TE': datapath + 'TE.xlsx', 
              'Combined': datapath + 'Combined.xlsx'} 

aurocs = {'Toronto':  {'SVM': 0.841, 'RFC': 0.873, 'GBC': 0.853, 'LOG': 0.812, 'MLP': 0.87,  'ENS': 0.87}, 
          'McGill':   {'SVM': 0.699, 'RFC': 0.711, 'GBC': 0.707, 'LOG': 0.683, 'MLP': 0.715, 'ENS': 0.716},
          'Combined': {'SVM': 0.725, 'RFC': 0.748, 'GBC': 0.739, 'LOG': 0.711, 'MLP': 0.747, 'ENS': 0.748}}

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
# SVMF = [feat_dict[f] for f in outputs['svmp']['features']]
# RFCF = [feat_dict[f] for f in outputs['rfcp']['features']]
# GBCF = [feat_dict[f] for f in outputs['gbcp']['features']]
# LOGF = [feat_dict[f] for f in outputs['logp']['features']]
# MLPF = [feat_dict[f] for f in outputs['mlpp']['features']]

res = []

for dk in aurocs.keys(): 
    dataset = pd.read_excel(path_dict[dk], index_col=None, engine='openpyxl')
    dataset['orig_fibrosis'] = dataset['Fibrosis']
    dataset['target'] = np.where(dataset['Fibrosis'] >= 3, 1, 0)
    
    
    for feat in feat_dict.keys(): 
        for i in range(0,num_trials): 
            print('DK: %s, Feat: %s, %d' % (dk, feat, i))
            # Impute and scale data
            X_test_unimp = dataset[[ 'Age', 'Sex', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 
                        'Creatinine', 'INR', 'BMI', 'Platelets', 'Diabetes']]
            X_test_imp = X_test_unimp.copy()
            
            for col in X_test_unimp.columns.tolist(): 
                X_test_imp[col] = np.where(X_test_unimp[col].isnull(), scl_info[col.lower() + '_mean'], X_test_unimp[col])
                
            X_test_imp[feat] = np.random.permutation(X_test_imp[feat])
            
            X_test_imp_scl = X_test_imp.copy()
            for col in X_test_imp.columns.tolist(): 
                X_test_imp_scl[col] = (X_test_imp_scl[col] - scl_info[col.lower() + '_mean'])/scl_info[col.lower() + '_std']
            
            xti = X_test_imp.values
            xts = X_test_imp_scl.values
            
            dataset['SVM_probs'] = SVM_model.predict_proba(xts)[:,1]    # SVM: Trained on Scaled Values
            dataset['RFC_probs'] = RFC_model.predict_proba(xti)[:,1]    # RFC: Trained on Unscaled Values
            dataset['GBC_probs'] = GBC_model.predict_proba(xti)[:,1]    # GBC: Trained on Unscaled Values
            dataset['LOG_probs'] = LOG_model.predict_proba(xts)[:,1]    # LOG: Trained on Scaled Values 
            dataset['MLP_probs'] = MLP_model.predict_proba(xts)[:,1]    # MLP: Trained on Scaled Values 
            dataset['ENS_probs'] = dataset[['SVM_probs', 'RFC_probs', 'GBC_probs', 'LOG_probs', 'MLP_probs']].sum(axis=1)/5
    
            svm_auroc = aurocs[dk]['SVM'] - roc_auc_score(dataset['target'], dataset['SVM_probs'])
            rfc_auroc = aurocs[dk]['RFC'] - roc_auc_score(dataset['target'], dataset['RFC_probs'])
            gbc_auroc = aurocs[dk]['GBC'] - roc_auc_score(dataset['target'], dataset['GBC_probs'])
            log_auroc = aurocs[dk]['LOG'] - roc_auc_score(dataset['target'], dataset['LOG_probs'])
            mlp_auroc = aurocs[dk]['MLP'] - roc_auc_score(dataset['target'], dataset['MLP_probs'])
            ens_auroc = aurocs[dk]['ENS'] - roc_auc_score(dataset['target'], dataset['ENS_probs'])
            
            algs_dict = {'SVM': svm_auroc, 'RFC': rfc_auroc, 'GBC': gbc_auroc, 'LOG': log_auroc, 'MLP': mlp_auroc, 'ENS': ens_auroc}
            for alg in algs_dict.keys():
                res_dict = {'dataset': dk, 'feature': feat, 'i': i, 'alg': alg, 'del_auroc': algs_dict[alg]}
                res.append(res_dict) 
                
results = pd.DataFrame(res)
results['del_auroc'] *= 100
results.sort_values(by=['dataset', 'feature', 'alg', 'i'], inplace=True)
results.to_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/feature_importance/feature_importances.csv')