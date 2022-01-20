import numpy as np
import pandas as pd
import pyodbc as db
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

def show(text):
	print(text)
	pause()

def pause():
	input('Press enter to continue!')

def APRI_class(Xp_test, Yp_test):
	indet_count = 0
	Yp_pred_new = []
	Yp_prob_new = []
	Yp_test_new = []
	det_indices = []
	apri_values = []

	for i in range(0, len(Xp_test)):
		AST_upper = 31 if Xp_test[i,0] == 0 else 19
		AST = Xp_test[i,1]
		Plt = Xp_test[i,2]
		APRI = (100*AST/AST_upper)/(Plt)
        
#		print('AST: ' + str(AST))
#		print('AST_Upper: ' + str(AST_upper))
#		print('PLT: ' + str(Plt))
#		print('APRI: ' + str(APRI)) 
#		input('Press enter to continue')
        
        
		if (APRI >= 2):
			apri_values.append(APRI)
			Yp_pred_new.append(4)
			Yp_test_new.append(Yp_test[i])
			Yp_prob_new.append(1)
			det_indices.append([i])
		elif (APRI <=0.5):
			apri_values.append(APRI)
			Yp_pred_new.append(0)
			Yp_test_new.append(Yp_test[i])
			Yp_prob_new.append(0)
			det_indices.append([i])
		else:
			indet_count += 1
			Yp_prob_new.append(0.5)
	return Yp_pred_new, Yp_prob_new, Yp_test_new, indet_count, apri_values

def FIB4_class(Xp_test, Yp_test):
	indet_count = 0
	Yp_pred_new = []
	Yp_prob_new = []
	Yp_test_new = []
	det_indices = []
	fib4_values = []

	for i in range(0, len(Xp_test)):
		age = Xp_test[i,0]
		ALT = Xp_test[i,1]
		AST = Xp_test[i,2]
		Plt = Xp_test[i,3]

		FIB4 = age*AST/(Plt*(ALT)**0.5)
		if (FIB4 >= 3.25):
			fib4_values.append(FIB4)
			Yp_pred_new.append(4)
			Yp_prob_new.append(1)
			Yp_test_new.append(Yp_test[i])
			det_indices.append([i])
		elif (FIB4 <=1.45):
			fib4_values.append(FIB4)
			Yp_pred_new.append(0)
			Yp_prob_new.append(0)
			Yp_test_new.append(Yp_test[i])
			det_indices.append([i])
		else:
			indet_count += 1
			Yp_prob_new.append(0.5)
	return Yp_pred_new, Yp_prob_new, Yp_test_new, indet_count, fib4_values

def my_auprc_non_prob(test_values, Yp_test):
    thresholds = np.linspace(0, np.max(test_values), 101)
    
    tprs = []
    precs = []
    
    for t in thresholds: 
        Yp_pred = (test_values >= t)*4

        cm = confusion_matrix(Yp_test, Yp_pred)
		
        tp = cm[1,1]
        fp = cm[0,1]
        fn = cm[1,0]

        if (tp + fn == 0):
            tprs.append(np.nan)
        else:
            tprs.append(tp/(tp+fn))
	
        if (tp + fp == 0):
            precs.append(np.nan)
        else:
            precs.append(tp/(tp+fp))	
      
#        print('Threshold: ' + str(t))
#        print(Yp_pred)
#        print(Yp_test)
#        print('TPRS: ' + str(tprs[len(tprs)-1]))
#        print('PRECS: ' + str(precs[len(precs)-1]))
#        print('Threshold: ' + str(t))
#        print('TPRS:')
#        print(tprs)
#        print('PRECS:')
#        print(precs)
#        print('')
#        input('Press enter to continue!')
        
    prc_curve = []
    
    for i in range(0, len(tprs) - 1):
        if (np.isnan(tprs[i]) == False and np.isnan(precs[i]) == False):
            prc_curve.append((tprs[i], precs[i]))
    #prc_curve = np.sort(prc_curve, axis=0)
    sorted_tprs = []
    sorted_precs = []
    
    for i in range(0,len(prc_curve)-1):
        sorted_tprs.append(prc_curve[i][0])
        sorted_precs.append(prc_curve[i][1])
    auprc = auc(sorted_tprs, sorted_precs, reorder=True)
        
    return auprc, sorted_tprs, sorted_precs

data = 'McGill' # McGill
missThresh = 3

conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Darth\Desktop\Thesis\Data\Fibrosis.accdb;')
cnxn = db.connect(conn_str)
cursor = cnxn.cursor()

if (data == 'Toronto'):
    sql = "SELECT * FROM _TorontoHoldOut30"
if (data == 'McGill'):
    sql = "SELECT * FROM _McGillData"

dataset = pd.read_sql(sql, cnxn)
dataset = dataset.loc[dataset['missingness'] <= missThresh]
#dataset = dataset.loc[dataset['AST'] < 1000]
dataset = dataset.loc[(dataset['Fibrosis'] == 0) | (dataset['Fibrosis'] == 1) | (dataset['Fibrosis'] == 4)]
dataset['Fibrosis'] = np.where(dataset['Fibrosis'] == 4, 4, 0)
features = ['Sex', 'Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'Creatinine', 'INR', 'Platelets', 'BMI', 'Diabetes', 'Hb', 'WBC', 'Fibrosis', 'patientID', 'bx_date', 'reckey_enc']
dataset = dataset[features]
imputer = joblib.load('imputer.joblib')
scaler = joblib.load('scaler.joblib')
X_test_unimp = dataset.iloc[:,0:len(features) - 4].values
X_test_imp = imputer.transform(X_test_unimp)
X_test_imp_scl = scaler.transform(X_test_imp)
Y_test = dataset.iloc[:,len(features)-4].values

#Loading the algorithms used in the ensemble model
APRI_preds, APRI_probs, APRI_test, APRI_indet_count, APRI_values = APRI_class(X_test_imp[:,[0, 5, 9]], Y_test)
APRI_precs_py, APRI_recs_py, thresholds = precision_recall_curve(Y_test/4, APRI_probs)
APRI_cm = confusion_matrix(APRI_test,APRI_preds)
APRI_AUPRC, APRI_recs, APRI_precs = my_auprc_non_prob(APRI_values, APRI_test)

FIB4_preds, FIB4_probs, FIB4_test, FIB4_indet_count, FIB4_values = FIB4_class(X_test_imp[:,[1, 4, 5, 9]], Y_test)
FIB4_precs_py, FIB4_recs_py, thresholds = precision_recall_curve(Y_test/4, FIB4_probs)
FIB4_cm = confusion_matrix(FIB4_test,FIB4_preds)
FIB4_AUPRC, FIB4_recs, FIB4_precs = my_auprc_non_prob(FIB4_values, FIB4_test)


plt.title('PRC curve for ' + data + ' dataset - Manual calculation')
plt.xlabel('Sensitivity')
plt.ylabel('PPV')
plt.plot(APRI_recs, APRI_precs, 'o-', label=('APRI, auprc=%0.2f'% (APRI_AUPRC)))
plt.plot(FIB4_recs, FIB4_precs, 'o-', label=('FIB4, auprc=%0.2f'% (FIB4_AUPRC)))
plt.plot([0,1],[1,0], 'k+--')
plt.legend(loc=3)
plt.grid(b=True)
plt.show()