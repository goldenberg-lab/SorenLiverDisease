# Program plan 
# 1. Load and preprocess the approrpriate dataset (TLC, MUHC, Expert)
# 2. Create file to track all relevant metrics for each sample for each model (Only need it for ENS2 at appropriate params)
# 3. For each bootstrap sample, sample n% records from dataset, calculate performance metrics, and log in appropriate file 
# 4. Sort all performance metrics, calculate mean, and calculate confidence intervals by taking 2.5%th percentile and 97.5%th percentile 
import numpy as np
np.set_printoptions(precision=10)
import pandas as pd
import pyodbc as db
from sklearn.externals import joblib 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_curve 
from sklearn.metrics import precision_recall_curve 
from sklearn.utils import resample 
from sklearn.metrics import auc 
from numpy import trapz
from beautifultable import BeautifulTable
from helper import *
import time as tm

def my_confusion_matrix(truth, pred):
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    
    for i in range(0,len(truth)):
        if(pred[i] == 0 and truth[i] == 0):
            tn += 1
        elif(pred[i] == 4 and truth[i] == 4):
            tp += 1
        elif(pred[i] == 4 and truth[i] == 0):
            fp += 1
        elif(pred[i] == 0 and truth[i] == 4):
            fn += 1
    
    try:        
        tot = tp + fp + tn + fn
    except ZeroDivisionError:
        tot = np.nan
    
    try:
        sens = tp/(tp + fn)
    except ZeroDivisionError:
        sens = np.nan

    try:
        spec = tn/(tn + fp)
    except ZeroDivisionError:
        spec = np.nan

    try:
        ppv = tp/(tp + fp)
    except ZeroDivisionError:
        ppv = np.nan

    try:    
        npv = tn/(tn + fn)
    except ZeroDivisionError:
        npv = np.nan
    
    try:
        acc = (tp + tn)/(tot)    
    except ZeroDivisionError:
        acc = np.nan
        
    return sens, spec, ppv, npv, acc
    
def non_prob_aucs(Y_test, probs, test_vals, test_truth):
    try: 
        AUROC, fprs, tprs = my_auroc_non_prob(test_vals, test_truth)
    except ValueError:
        AUROC = np.nan
    try:
        AUPRC, recs, precs, prc_curve = my_auprc_non_prob(test_vals, test_truth)
    except ValueError:
        AUPRC = np.nan  
    except IndexError:
        AUPRC = np.nan
    return AUROC, AUPRC

def ENS_class(av, ap, fb, Xp_test, Yp_test, params):
	indet_count = 0
	Yp_pred_new = []
	Yp_prob_new = []
	Yp_test_new = []
	det_indices = []
	probabilities = av.tolist()
   
	for i in range(0, len(Xp_test)):
		if (probabilities[i] >= (params['threshold'] + params['indet_range_high'])):
			Yp_pred_new.append(4)
			Yp_test_new.append(Yp_test[i])
			Yp_prob_new.append(probabilities[i])
			det_indices.append(i)
		elif (probabilities[i] < (params['threshold'] - params['indet_range_low'])):
			Yp_pred_new.append(0)
			Yp_test_new.append(Yp_test[i])
			Yp_prob_new.append(probabilities[i])
			det_indices.append(i)
		else:
			if(ap[i] == 1 and fb[i] == 1):
				Yp_pred_new.append(4)
				Yp_test_new.append(Yp_test[i])
				Yp_prob_new.append(probabilities[i])
				det_indices.append(i)
			elif(ap[i] == 0 and fb[i] == 0):
				Yp_pred_new.append(0)
				Yp_test_new.append(Yp_test[i])
				Yp_prob_new.append(probabilities[i])
				det_indices.append(i)
			else:
				indet_count += 1
	return Yp_pred_new, Yp_prob_new, np.array(Yp_test_new), indet_count, det_indices

def get_sampled_data(data, size):
    dataset = resample(data,n_samples=size)
    X_test_unimp = dataset.iloc[:,0:len(features) - 4].values
    X_test_imp = imputer.transform(X_test_unimp)
    X_test_imp_scl = scaler.transform(X_test_imp)
    Y_test = dataset.iloc[:,len(features)-4].values
    return X_test_unimp, X_test_imp, X_test_imp_scl, Y_test, dataset

# Load imputer, scaler, and ML models
imputer = joblib.load('imputer.joblib')
scaler = joblib.load('scaler.joblib')
SVM_model = joblib.load('SVM.joblib')
RFC_model = joblib.load('RFC.joblib')
GBC_model = joblib.load('GBC.joblib')
LOG_model = joblib.load('LOG.joblib')
MLP_model = joblib.load('MLP.joblib')
ENS3_params = joblib.load('ENS3.joblib')
ENS3_params['threshold'] = 0.45
ENS3_params['indet_range_low'] = 0.2
ENS3_params['indet_range_high'] = 0.0
ENS3_params['indets'] = 'aprifib4'

path = "C:/Users/Soren/Desktop/Thesis/Data Analysis/Hold-out Test Sets/final_2000-2014no(hb,wbc)/bootstrap_results/"
conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Soren\Desktop\Thesis\Fibrosis.accdb;')
cnxn = db.connect(conn_str)
cursor = cnxn.cursor()

ds = 'Expert'

if (ds == 'Toronto'):
    sql = "SELECT * FROM _TorontoHoldOut30 WHERE missingness <= 3"
elif (ds == 'McGill'):
    sql = "SELECT * FROM _McGillData Where Neoplasm=0 AND missingness <= 3 AND (Fibrosis=0 OR Fibrosis=1 or Fibrosis=4)"
elif (ds == 'Expert'):
    sql = "SELECT * FROM _ExpertPredsCombined WHERE bx_date IS NOT NULL AND missingness <= 3"

data = pd.read_sql(sql, cnxn)
data['Fibrosis'] = np.where(data['Fibrosis'] == 4, 4, 0)
features = ['Sex', 'Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'Creatinine', 'INR', 'Platelets', 'BMI', 'Diabetes', 'Hb', 'WBC', 'Fibrosis', 'patientID', 'bx_date', 'reckey_enc']
data = data[features]


n_iterations = 1000
n_size = int(len(data)*1)

#APRI_FIB4_results = np.empty([8,n_iterations]) 
APRI_ENS3_results = np.empty([8,n_iterations]) 
#FIB4_ENS3_results = np.empty([8,n_iterations])
#EXPE_results = np.empty([8,n_iterations])

#print('Number of records in dataset: ' + str(len(dataset)))
#print('F01/F4: ' + str(len(dataset.loc[dataset['Fibrosis'] == 0])) + '/' + str(len(dataset.loc[dataset['Fibrosis'] == 4])))

# Okay. Now, I need to pick the number of 

for i in range(0,n_iterations):
    X_test_unimp, X_test_imp, X_test_imp_scl, Y_test, dataset = get_sampled_data(data, n_size)
    
    APRI_preds, APRI_probs, APRI_test, APRI_indet_count, APRI_values = APRI_class(X_test_imp[:,[0, 5, 9]], Y_test)
    APRI_precs_py, APRI_recs_py, thresholds = precision_recall_curve(Y_test/4, APRI_probs)
    APRI_sens, APRI_spec, APRI_ppv, APRI_npv, APRI_acc = my_confusion_matrix(APRI_test,APRI_preds)
    APRI_AUROC, APRI_AUPRC = non_prob_aucs(Y_test, APRI_probs, APRI_values, APRI_test)
    APRI_dets = 1-APRI_indet_count/n_size
            
    FIB4_preds, FIB4_probs, FIB4_test, FIB4_indet_count, FIB4_values = FIB4_class(X_test_imp[:,[1, 4, 5, 9]], Y_test)
#    FIB4_precs_py, FIB4_recs_py, thresholds = precision_recall_curve(Y_test/4, FIB4_probs)
#    FIB4_sens, FIB4_spec, FIB4_ppv, FIB4_npv, FIB4_acc = my_confusion_matrix(FIB4_test,FIB4_preds)
#    FIB4_AUROC, FIB4_AUPRC = non_prob_aucs(Y_test, FIB4_probs, FIB4_values, FIB4_test)
#    FIB4_dets = 1-FIB4_indet_count/n_size

    ML_probs = np.empty([np.size(X_test_imp_scl,0),6])
    ML_probs[:,0] = SVM_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]])[:,1] 
    ML_probs[:,1] = RFC_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])[:,1]
    ML_probs[:,2] = GBC_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]])[:,1]
    ML_probs[:,3] = LOG_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]])[:,1]
    ML_probs[:,4] = MLP_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 8, 9, 10, 11]])[:,1]
    ML_probs[:,5] = np.sum(ML_probs[:,0:6], axis=1)/5
    
    ENS3_preds, ENS3_probs, ENS3_test, ENS3_indet_count, ENS3_det_indices = ENS_class(ML_probs[:,5], APRI_probs, FIB4_probs,  X_test_imp_scl, Y_test, ENS3_params)
    ENS3_sens, ENS3_spec, ENS3_ppv, ENS3_npv, ENS3_acc = my_confusion_matrix(ENS3_test,ENS3_preds)
    ENS3_dets = 1-ENS3_indet_count/n_size

    try:
        ENS3_precs_py, ENS3_recs_py, thresholds = precision_recall_curve(ENS3_test/4, ENS3_probs)
        ENS3_AUPRC = auc(ENS3_recs_py, ENS3_precs_py)
    except ValueError:
        ENS3_AUPRC = np.nan

    try: 
        ENS3_fprs_py, ENS3_tprs_py, threshold = roc_curve(ENS3_test/4, ENS3_probs)
        ENS3_AUROC = roc_auc_score(np.asarray(ENS3_test)/4, ENS3_probs)
    except ValueError:
        ENS3_AURPC = np.nan
   
#    APRI_FIB4_results[:,i] = (-1)*100*np.transpose(np.array([APRI_sens-FIB4_sens, APRI_spec-FIB4_spec, APRI_ppv-FIB4_ppv, APRI_npv-FIB4_npv, APRI_acc-FIB4_acc, APRI_AUROC-FIB4_AUROC, APRI_AUPRC-FIB4_AUPRC, APRI_dets-FIB4_dets]))
    APRI_ENS3_results[:,i] = (-1)*100*np.transpose(np.array([APRI_sens-ENS3_sens, APRI_spec-ENS3_spec, APRI_ppv-ENS3_ppv, APRI_npv-ENS3_npv, APRI_acc-ENS3_acc, APRI_AUROC-ENS3_AUROC, APRI_AUPRC-ENS3_AUPRC, APRI_dets-ENS3_dets]))
#    FIB4_ENS3_results[:,i] = (-1)*100*np.transpose(np.array([FIB4_sens-ENS3_sens, FIB4_spec-ENS3_spec, FIB4_ppv-ENS3_ppv, FIB4_npv-ENS3_npv, FIB4_acc-ENS3_acc, FIB4_AUROC-ENS3_AUROC, FIB4_AUPRC-ENS3_AUPRC, FIB4_dets-ENS3_dets]))
    print('Completed iteration # ' + str(i))

#APRI_FIB4_df = pd.DataFrame.from_records(APRI_FIB4_results)
#APRI_FIB4_df = APRI_FIB4_df.rename(index={0: 'sens',1: 'spec',2: 'ppv',3: 'npv',4: 'acc',5: 'AUROC',6: 'AUPRC',7: 'det'})
#APRI_FIB4_df.to_excel(path + ds + '_APRI_FIB4_bootstrap.xlsx')
#
APRI_ENS3_df = pd.DataFrame.from_records(APRI_ENS3_results)
APRI_ENS3_df = APRI_ENS3_df.rename(index={0: 'sens',1: 'spec',2: 'ppv',3: 'npv',4: 'acc',5: 'AUROC',6: 'AUPRC',7: 'det'})
APRI_ENS3_df.to_excel(path + ds + '_APRI_ENS3_bootstrap.xlsx')

#FIB4_ENS3_df = pd.DataFrame.from_records(FIB4_ENS3_results)
#FIB4_ENS3_df = FIB4_ENS3_df.rename(index={0: 'sens',1: 'spec',2: 'ppv',3: 'npv',4: 'acc',5: 'AUROC',6: 'AUPRC',7: 'det'})
#FIB4_ENS3_df.to_excel(path + ds + '_FIB4_ENS3_bootstrap.xlsx')
# Okay. I now need to find 95% Confidence intervals. 
# Best method is to save these results somewhere. 

# Repeat the same process for the expert set