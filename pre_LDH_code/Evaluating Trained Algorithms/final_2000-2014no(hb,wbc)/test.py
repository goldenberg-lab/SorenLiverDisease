import os 
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
import sklearn

os.chdir(r'C:\Users\Darth\Desktop\Thesis\Code\Evaluating Trained Algorithms\final_2000-2014no(hb,wbc)')

def information(ds):
    features = ['Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'Creatinine', 'INR', 'Platelets', 'BMI']
    for feat in features:
        print('Feature: ' + feat)
        print('Mean:    ' + str(np.mean(ds[feat].dropna())))
        print('2.5th:   ' + str(np.percentile(ds[feat].dropna(), 2.5)))
        print('97.5th:   ' + str(np.percentile(ds[feat].dropna(), 97.5)))
        print('')
        
def area_by_shoelace(x,y):
    return abs(sum(i*j for i, j in zip(x,y[1:])) + x[-1]*y[0] - sum(i*j for i,j in zip(x[1:],y)) - x[0]*y[-1])/2

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
		AST_upper = 35 #31 if Xp_test[i,0] == 0 else 19
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
		elif (APRI <=1):
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

def NAFLD_class(Xp_test, Yp_test):
	indet_count = 0
	Yp_pred_new = []
	Yp_prob_new = []
	Yp_test_new = []
	det_indices = []
	nafld_values = []

	for i in range(0, len(Xp_test)):
		age = Xp_test[i,0]
		albumin = Xp_test[i,1]
		ALT = Xp_test[i,2]
		AST = Xp_test[i,3]
		Plt = Xp_test[i,4]
		BMI = Xp_test[i,5]
		Diab = Xp_test[i,6]
        
#		print('Age: ' + str(age))
#		print('Alb: ' + str(albumin))
#		print('ALT: ' + str(ALT))
#		print('AST: ' + str(AST))
#		print('Plt: ' + str(Plt))
#		print('BMI: ' + str(BMI))
#		print('Diab: ' + str(Diab))

		NAFLD = -1.675 + 0.037*age + 0.094*BMI + 1.13*Diab + 0.99*(AST/ALT) - 0.013*Plt - 0.66*albumin/10
#		input('NAFLD index was : ' + str(NAFLD))
		if (NAFLD >= 0.676):
			nafld_values.append(NAFLD)
			Yp_pred_new.append(4)
			Yp_prob_new.append(1)
			Yp_test_new.append(Yp_test[i])
			det_indices.append([i])
		elif (NAFLD <=-1.455):
			nafld_values.append(NAFLD)
			Yp_pred_new.append(0)
			Yp_prob_new.append(0)
			Yp_test_new.append(Yp_test[i])
			det_indices.append([i])
		else:
			indet_count += 1
			Yp_prob_new.append(0.5)
	return Yp_pred_new, Yp_prob_new, Yp_test_new, indet_count, nafld_values

def ENS_class(s,r,g,l,k,m, ap, fb, Xp_test, Yp_test, params, excluded_algs):
	indet_count = 0
	Yp_pred_new = []
	Yp_prob_new = []
	Yp_test_new = []
	det_indices = []
	probabilities = []
	#algs = [s,r,g,l,k,m,]
	algs = []

	if ('SVM' not in excluded_algs):
		algs.append(s)
	if ('RFC' not in excluded_algs):
		algs.append(r)
	if ('GBC' not in excluded_algs):
		algs.append(g)
	if ('LOG' not in excluded_algs):
		algs.append(l)
	if ('KNN' not in excluded_algs):
		algs.append(k)
	if ('MLP' not in excluded_algs):
		algs.append(m)
    
	indet_f4 = 0
	indet_f0 = 0

	for i in range(0, len(Xp_test)):
		prob_sum = 0
		terms = 0
		weight_sum = 0
		f4_votes = 0
		f0_votes = 0

		for alg in algs:
			prob_sum += alg[i]
			terms += 1
		probabilities.append(prob_sum/terms)
#		if ((ap[i] == 0.5 and params['indets'] == 'APRI') or (fb[i] == 0.5 and params['indets'] == 'FIB4')):
#				indet_count += 1
#				continue
		p=('%.2f' %  (probabilities[i]))
		if (probabilities[i] >= (params['threshold'] + params['indet_range_high'])):
			#print('Guessed 4 for probability: ' + str(probabilities[i]) + ' at threshold ' + str(params['threshold'] + params['indet_range_high']))        
			Yp_pred_new.append(4)
			Yp_test_new.append(Yp_test[i])
			Yp_prob_new.append(probabilities[i])
			det_indices.append(i)
			print(p + ' guessed F4')
		elif (probabilities[i] < (params['threshold'] - params['indet_range_low'])):
			Yp_pred_new.append(0)
			Yp_test_new.append(Yp_test[i])
			Yp_prob_new.append(probabilities[i])
			#print('Guessed 0 for probability: ' + str(probabilities[i]) + ' at threshold ' + str(params['threshold'] - params['indet_range_low']))        
			det_indices.append(i)
			print(p + ' guessed F01')
		else:
			if(params['indet_guess_mode'] == 'none'):
				indet_count += 1
			elif(params['indet_guess_mode'] == 'aprifib4'):
				if(ap[i] == 1 and fb[i] == 1):
					#print('Guessed indeterminate, but APRI and FIB4 both said 4')                
					Yp_pred_new.append(4)
					Yp_test_new.append(Yp_test[i])
					Yp_prob_new.append(probabilities[i])
					det_indices.append(i)
					print(p + ' guessed F4 w/ AF')
				elif(ap[i] == 0 and fb[i] == 0):
					#print('Guessed indeterminate, but APRI and FIB4 both said 0')                    
					Yp_pred_new.append(0)
					Yp_test_new.append(Yp_test[i])
					Yp_prob_new.append(probabilities[i])
					det_indices.append(i)
					print(p + ' guessed F01 w/ AF')
				else:
					print('Guessed indeterminate on an ' + str(Yp_test[i]))                    
					indet_count += 1
#					if (Yp_test[i] == 4):
#					   indet_f4 += 1
#					else:
#					   indet_f0 += 1
#    
#	print('ENS3 # of indet F4s: ' + str(indet_f4))
#	print('ENS3 # of indet F01s: ' + str(indet_f0))
	print(Yp_pred_new)                
	return Yp_pred_new, Yp_prob_new, Yp_test_new, indet_count, det_indices

def write_performance_metrics(cms, names, aurocs, man_aurocs, auprcs, man_auprcs, indet_array, threshold_array, excluded_algs, noshow):
	from beautifultable import BeautifulTable
	table = BeautifulTable(max_width=300)
	table.append_column(' ', ['sensitivity', 'specificity', 'PPV', 'NPV', 'accuracy', 'Manual AUROC', 'Manual AUPRC', '% indet', 'Dec. Thresh. (%)'])
	count = 0
	for cm in cms:
		if (names[count] in excluded_algs):
			print('Did not include ' + str(names[count]) + 'results!')
			count += 1 
			continue 
		if (names[count] in noshow):
			#print('Did not show ' + str(names[count]) + 'results!')
			count += 1 
			continue      
#		if (names[count] == 'APRI' or names[count] == 'FIB4'):
#        		count += 1
#        		continue
		#if (names[count] != 'ENS1' and names[count] != 'ENS2' and names[count] != 'ENS3'):
        #		count += 1
        #		continue
		tot = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1]
		tp = cm[1,1]
		fp = cm[0,1]
		tn = cm[0,0]
		fn = cm[1,0]

		accuracy = 100*(tp + tn)/(tot)
		precision = 100*tp/(tp + fp)
		negPredVal = 100*tn/(tn + fn)
		sensitivity = 100*tp/(tp + fn)
		falseNegRate = 100*fn/(fn + tp)
		specificity = 100*tn/(tn + fp)
		falsePosRate = 100*fp/(fp + tn)
		f1 = 100*2*precision*sensitivity/(precision + sensitivity)
		auc = aurocs[count]
		man_aucs = man_aurocs[count]
		prc = auprcs[count]
		man_prcs = man_auprcs[count]
		indt = indet_array[count]
		thresh = threshold_array[count]
		table.append_column(names[count], [(' %0.1f' % sensitivity), (' %0.1f' % specificity), (' %0.1f' % precision), (' %0.1f' % negPredVal), (' %0.1f' % accuracy), (' %0.1f' % (100*man_aucs)), (' %0.1f' % (100*man_prcs)), (' %0.1f' % (100*indt)), thresh])
		count += 1
	print(table)
    
def my_auprc_non_prob(test_values, Yp_test):
    thresholds = np.linspace(0, np.max(test_values)+0.5, 1000)
    
    tprs = []
    precs = []
    
    for t in thresholds: 
        Yp_pred = (test_values >= t)*4

        cm = my_confusion_matrix(Yp_test, Yp_pred)
		
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
    prc_tprs = np.array([])
    prc_prcs = np.array([])
    
    for i in range(0, len(tprs) - 1):
        tpr = tprs[i]
        pre = precs[i]
    
        if (np.isnan(tpr) == True or np.isnan(pre) == True):
            continue 
        if (np.isnan(tpr) == False and np.isnan(pre) == False):
            prc_tprs = np.append(prc_tprs, tpr)
            prc_prcs = np.append(prc_prcs, pre)

    if (prc_tprs[-1] == 0 and prc_prcs[-1] == 0):
        prc_tprs = np.append(prc_tprs, np.array([1]))
        prc_prcs = np.append(prc_prcs, np.array([0]))
    else:
        prc_tprs = np.append(prc_tprs, np.array([0,0,1]))
        prc_prcs = np.append(prc_prcs, np.array([1,0,0]))

    auprc1 = area_by_shoelace(prc_tprs, prc_prcs)
    
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
#    print('Original method: ' + str(auprc))
#    print('Shoelace method: ' + str(auprc1))
#    input('Press enter to continue!')
    
    return auprc1, sorted_tprs, sorted_precs, prc_curve

def my_auroc_non_prob(test_values, Yp_test):
	thresholds = np.linspace(np.min(test_values),np.max(test_values),1000)

	fprs = []
	tprs = []
    
	for t in thresholds:
		Yp_pred = (test_values >= t)*4
		cm = my_confusion_matrix(Yp_test, Yp_pred)
		
		tp = cm[1,1]
		fp = cm[0,1]
		tn = cm[0,0]
		fn = cm[1,0]
	
		if (fp + tn == 0):
			fprs.append(np.nan)
		else:
			fprs.append(fp/(fp+tn))

		if (tp + fn == 0):
			tprs.append(np.nan)
		else:
			tprs.append(tp/(tp+fn))
	    
	roc_curve = [(0,0),(1,1)]
    
	for i in range(0,len(fprs)-1):
		if (np.isnan(fprs[i]) == False and np.isnan(tprs[i]) == False):
			roc_curve.append((fprs[i], tprs[i]))
         
	roc_curve = np.sort(roc_curve, axis=0)
	sorted_fprs = []
	sorted_tprs = []    
    
	for i in range(0,len(roc_curve)-1):
		sorted_fprs.append(roc_curve[i][0])
		sorted_tprs.append(roc_curve[i][1])
	auroc = auc(sorted_fprs, sorted_tprs)
    
	return auroc, sorted_fprs, sorted_tprs

def my_auprc_prob(probs, Yp_test):
	from sklearn.metrics import auc
	from numpy import trapz
	tprs = []
	precs = []
	threshs = []

	for i in range(0,250):
		thresh = i/250
		Yp_pred = (np.asarray(probs) > np.asarray(thresh))*4
		cm = my_confusion_matrix(Yp_test, Yp_pred)
		
		tp = cm[1,1]
		fp = cm[0,1]
		fn = cm[1,0]
        
        
		if (tp + fn == 0 or tp + fp == 0):
				tprs.append(np.nan)
				precs.append(np.nan)			
				threshs.append(np.nan)
		else:
				precs.append(tp/(tp+fp))			
				tprs.append(tp/(tp+fn))
				threshs.append(thresh)

	tprs.append(1)
	precs.append(0)
	threshs.append(np.nan)    
    
	tprs.append(0)
	precs.append(1)   
	threshs.append(np.nan)    

	prc_curve = []
	prc_tprs = np.array([])
	prc_prcs = np.array([])
    
	for i in range(0, len(tprs) - 1):
		tpr = tprs[i]
		pre = precs[i]
    
		if (np.isnan(tpr) == True or np.isnan(pre) == True):
			continue 
		if (np.isnan(tpr) == False and np.isnan(pre) == False):
			if (tpr == 1 and pre == 0):
				continue
			prc_tprs = np.append(prc_tprs, tpr)
			prc_prcs = np.append(prc_prcs, pre)

	if (prc_tprs[-1] == 0 and prc_prcs[-1] == 0):
		prc_tprs = np.append(prc_tprs, np.array([1]))
		prc_prcs = np.append(prc_prcs, np.array([0]))
	else:
		prc_tprs = np.append(prc_tprs, np.array([0,0,1]))
		prc_prcs = np.append(prc_prcs, np.array([1,0,0]))

#	for i in range(0, len(prc_tprs)):
#		print('(%0.3f, %0.3f)' % (prc_tprs[i], prc_prcs[i]))

	auprc1 = area_by_shoelace(prc_tprs, prc_prcs)


	for i in range(0,max(len(tprs), len(precs))):
		if (np.isnan(tprs[i]) == False and np.isnan(precs[i]) == False):
			prc_curve.append([tprs[i],  precs[i]])
	sorted_tprs = []
	sorted_precs = []
    
	for i in range(0,len(prc_curve)):
		sorted_tprs.append(prc_curve[i][0])
		sorted_precs.append(prc_curve[i][1])
	auprc = auc(sorted_tprs, sorted_precs, reorder=True)
	
	#temp = pd.DataFrame(data=[tprs, precs])
	#temp = temp.transpose()
	#temp.to_csv('temp_AUPRC_check.csv')
    
#	print('Old method AUPRC: ' + str(auprc))
#	print('Shoelace method AURPC: ' + str(auprc1))
#	input('Batman') 

	#return auprc, sorted_tprs, sorted_precs
	return auprc1, tprs, precs

def my_auroc_prob(probs, Yp_test):
	from sklearn.metrics import auc
	fprs = []
	tprs = []

	for i in range(0,3001):
		thresh = i/3000
		Yp_pred = (np.asarray(probs) > np.asarray(thresh))*4
		cm = my_confusion_matrix(Yp_test, Yp_pred)
        		
		tp = cm[1,1]
		fp = cm[0,1]
		tn = cm[0,0]
		fn = cm[1,0]
        
		if (fp + tn == 0):
			fprs.append(np.nan)
		else:
			fprs.append(fp/(fp+tn))

		if (tp + fn == 0):
			tprs.append(np.nan)
		else:
			tprs.append(tp/(tp+fn))
            
	roc_curve = [(0,0),(1,1)]

	for i in range(0,len(fprs)-1):
		if (np.isnan(fprs[i]) == False and np.isnan(tprs[i]) == False):
			roc_curve.append((fprs[i],  tprs[i]))
            
	roc_curve = np.sort(roc_curve, axis=0)
	sorted_fprs = []
	sorted_tprs = []
	for i in range(0,len(roc_curve)-1):
		sorted_fprs.append(roc_curve[i][0])
		sorted_tprs.append(roc_curve[i][1])
	auroc = auc(sorted_fprs, sorted_tprs)
	return auroc, sorted_fprs, sorted_tprs

def my_confusion_matrix(truth, pred):
    TN_count = 0
    TP_count = 0
    FP_count = 0
    FN_count = 0
    
    for i in range(0,len(truth)):
        if(pred[i] == 0 and truth[i] == 0):
            TN_count += 1
        elif(pred[i] == 4 and truth[i] == 4):
            TP_count += 1
        elif(pred[i] == 4 and truth[i] == 0):
            FP_count += 1
        elif(pred[i] == 0 and truth[i] == 4):
            FN_count += 1

    cm = np.ndarray(shape=(2,2))
    cm[1,1] = TP_count
    cm[0,1] = FP_count
    cm[0,0] = TN_count
    cm[1,0] = FN_count
    return cm
# Testing with ICES data to make sure the rest of the code works. This code is only loaded at runtime, and is not saved anywhere in the directories being extracted.
#dataset = pd.read_csv('/dshroot/projects/hspe/dsh0990.111.003/user_data/ssabetsarvestany/dataset_assembly/dataset_bx_30.csv')
#dataset['sex'] = np.where(dataset['sex'] == 'Male', 0, 1)
#dataset['reckey_enc'] = dataset['reckey_enc'].astype(str)
#dataset['patientID'] = dataset['reckey_enc'] + '|' + dataset['bx_date']
#dataset = dataset.sort_values(by='bx_date', ascending=False)
#dataset = dataset.reset_index(drop=True)
#dataset = dataset.loc[dataset['bx_date'] >= '2012-01-01']
#dataset = dataset.loc[dataset['with_olis_missingness'] <= 3]
#dataset = dataset.rename(columns={'calc_bmi': 'bmi'})
#features = ['sex', 'age', 'albumin', 'alp', 'alt', 'ast', 'tot_bil', 'creatinine', 'inr', 'platelets', 'bmi', 'diabetes', 'hb', 'wbc', 'fibrosis', 'patientID', 'bx_date', 'reckey_enc']
#dataset = dataset[features]

desc_string = ''
data = 'McGill' 
missThresh = 3
exc = []

no_show = []# ['SVM', 'RFC', 'GBC', 'LOG', 'MLP']
exc_alg = ['KNN']

desc_string += data + ' holdout set, missingness <=' + str(missThresh)

if ('Age' in exc):
    desc_string += ', Age <= 60'
if ('Albumin' in exc):
    desc_string += ', Albumin >= 30'

#Code to import Toronto 2015-2016 and Mcgill datasets once everything has been extracted
conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Darth\Desktop\Thesis\Data\Fibrosis.accdb;')
cnxn = db.connect(conn_str)
cursor = cnxn.cursor()

if (data == 'Toronto'):
    sql = "SELECT * FROM _TorontoHoldOut30"
if (data == 'McGill'):
    sql = "SELECT * FROM _McGillData WHERE Neoplasm=0 AND FIBROSIS IS NOT NULl"
if (data == 'Expert'):
    sql = "SELECT * FROM _ExpertPredsCombined WHERE bx_date IS NOT NULL"#_ExpertPredsCombined"

dataset = pd.read_sql(sql, cnxn)

#ds1 = pd.read_sql("SELECT * FROM _TorontoHoldOut30 WHERE [WilsonsDisease]=1 OR OTHER=1", cnxn)
#ds2 = pd.read_sql("SELECT * FROM _McGillData WHERE NEOPLASM=0 AND ([WilsonsDisease]=1 OR OTHER=1)", cnxn)
#dataset = ds1.append(ds2)
#print(dataset.columns.tolist())
#dataset = dataset.loc[dataset['Other'] == 1]
#dataset = dataset.loc[dataset['AST'] < 1000]
dataset = dataset.reset_index(drop=True)
dataset = dataset.loc[dataset['missingness'] <= missThresh]
#dataset = dataset.loc[(dataset['Fibrosis'] == 0) | (dataset['Fibrosis'] == 1) | (dataset['Fibrosis'] == 4)]
#dataset['Fibrosis'] = np.where(dataset['Fibrosis'] == 2, 4, dataset['Fibrosis'])
#dataset['Fibrosis'] = np.where(dataset['Fibrosis'] == 3, 4, dataset['Fibrosis'])
dataset['Fibrosis'] = np.where(dataset['Fibrosis'] >= 4, 4, 0)

if ('Age' in exc):
    dataset = dataset.loc[dataset['Age'] <= 60]

if ('Albumin' in exc):
    dataset = dataset.loc[dataset['Albumin'] >= 30]

#exp_data = pd.read_sql("SELECT * FROM _ExpertPreds", cnxn)
#exp_data = exp_data.reset_index(drop=True)
#exp_data['Fibrosis'] = np.where(exp_data['Fibrosis'] == 'F 4', 4, 0)

#F01s = dataset.loc[dataset['Fibrosis'] == 0]
#F4s = dataset.loc[dataset['Fibrosis'] == 4]
#F01s = F01s.sample(len(F4s), random_state=20)
#dataset = F01s.append(F4s, ignore_index=False, verify_integrity=True)
#dataset = dataset.sample(frac=1, random_state=0)

#print('Number of records: ' + str(len(dataset)))
#print('Number of NAFL records: ' + str(num_nafl))
#print('% NAFL: ' + str(100*num_nafl/len(dataset)))
#print(dataset.columns.tolist())
features = ['Sex', 'Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'Creatinine', 'INR', 'Platelets', 'BMI', 'Diabetes', 'Hb', 'WBC', 'Fibrosis', 'patientID', 'bx_date', 'reckey_enc']
dataset = dataset[features]

F0 = dataset.loc[dataset['Fibrosis'] == 0]
F4 = dataset.loc[dataset['Fibrosis'] == 4]

#information(F0)
#show('Enter to continue')
#information(F4)

#exp_data = exp_data[features]
#dataset['Sex'] = dataset['Sex'].astype(int)

#exp_data = exp_data.loc[exp_data['Age'] < 60]
#exp_data = exp_data.loc[exp_data['Albumin'] >= 30]
#exp_data = exp_data.reset_index(drop=True)

imputer = joblib.load('imputer.joblib')
scaler = joblib.load('scaler.joblib')

# Predicting on Toronto and McGill datasets
X_test_unimp = dataset.iloc[:,0:len(features) - 4].values
X_test_imp = imputer.transform(X_test_unimp)
X_test_imp_scl = scaler.transform(X_test_imp)
Y_test = dataset.iloc[:,len(features)-4].values

# Predicting on Expert Cases
#X_test_unimp = exp_data.iloc[:,0:len(features)-4].values
#X_test_imp = imputer.transform(X_test_unimp)
#X_test_imp_scl = scaler.transform(X_test_imp)
#Y_test = exp_data.iloc[:,len(features)-4].values

cm_array = []
name_array = []
auroc_array = []
my_auroc_array = []
auprc_array = []
my_auprc_array = []
indet_array = []
threshold_array = []

#Loading the algorithms used in the ensemble model
print('Now making predictions for SVM')
SVM_model = joblib.load('SVM.joblib')
SVM_probs = SVM_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]])
SVM_probs = SVM_probs[:,1]
SVM_fprs_py, SVM_tprs_py, threshold = roc_curve(Y_test/4, SVM_probs)
SVM_precs_py, SVM_recs_py, thresholds = precision_recall_curve(Y_test/4, SVM_probs)
SVM_preds = (SVM_probs > 0.4)*4
SVM_cm = my_confusion_matrix(Y_test,SVM_preds)
cm_array.append(SVM_cm)
name_array.append('SVM')
try:
    auroc_array.append(roc_auc_score(Y_test/4, SVM_probs)) # Python calculated AUROC
except ValueError:
    auroc_array.append(np.nan)
try: 
    SVM_AUROC, SVM_fprs, SVM_tprs = my_auroc_prob(SVM_probs, Y_test) # Manually calculated AUROC
    my_auroc_array.append(SVM_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
try:
    auprc_array.append(auc(SVM_recs_py, SVM_precs_py))
except ValueError:
    auprc_array.append(np.nan)
try:
    SVM_AUPRC, SVM_recs, SVM_precs = my_auprc_prob(SVM_probs, Y_test)
    my_auprc_array.append(SVM_AUPRC)    
except ValueError:
    my_auprc_array.append(np.nan)
except IndexError:
    my_auprc_array.append(np.nan)

indet_array.append(0)
threshold_array.append(0.4)

print('Now making predictions for RFC')
RFC_model = joblib.load('RFC.joblib')
RFC_probs = RFC_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
RFC_probs = RFC_probs[:,1]
RFC_fprs_py, RFC_tprs_py, threshold = roc_curve(Y_test/4, RFC_probs)
RFC_precs_py, RFC_recs_py, thresholds = precision_recall_curve(Y_test/4, RFC_probs)
RFC_preds = (RFC_probs > 0.4)*4
RFC_cm = my_confusion_matrix(Y_test,RFC_preds)
cm_array.append(RFC_cm)
name_array.append('RFC') 
try:
    auroc_array.append(roc_auc_score(Y_test/4, RFC_probs)) # Python calculated AUROC
except ValueError:
    auroc_array.append(np.nan)
    
try: 
    RFC_AUROC, RFC_fprs, RFC_tprs = my_auroc_prob(RFC_probs, Y_test) # Manually calculated AUROC
    my_auroc_array.append(RFC_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
    
try:
    auprc_array.append(auc(RFC_recs_py, RFC_precs_py))
except ValueError:
    auprc_array.append(np.nan)

try:
    RFC_AUPRC, RFC_recs, RFC_precs = my_auprc_prob(RFC_probs, Y_test)
    my_auprc_array.append(RFC_AUPRC)    
except ValueError:
    my_auprc_array.append(np.nan)
except IndexError:
    my_auprc_array.append(np.nan)      
indet_array.append(0)
threshold_array.append(0.4)

print('Now making predictions for GBC')
GBC_model = joblib.load('GBC.joblib')
GBC_probs = GBC_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]])
GBC_probs = GBC_probs[:,1]
GBC_fprs_py, GBC_tprs_py, threshold = roc_curve(Y_test/4, GBC_probs)
GBC_precs_py, GBC_recs_py, thresholds = precision_recall_curve(Y_test/4, GBC_probs)
GBC_preds = (GBC_probs > 0.4)*4
GBC_cm = my_confusion_matrix(Y_test,GBC_preds)
cm_array.append(GBC_cm)
name_array.append('GBC')
try:
    auroc_array.append(roc_auc_score(Y_test/4, GBC_probs)) # Python calculated AUROC
except ValueError:
    auroc_array.append(np.nan)
    
try: 
    GBC_AUROC, GBC_fprs, GBC_tprs = my_auroc_prob(GBC_probs, Y_test) # Manually calculated AUROC
    my_auroc_array.append(GBC_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
    
try:
    auprc_array.append(auc(GBC_recs_py, GBC_precs_py))
except ValueError:
    auprc_array.append(np.nan)

try:
    GBC_AUPRC, GBC_recs, GBC_precs = my_auprc_prob(GBC_probs, Y_test)
    my_auprc_array.append(GBC_AUPRC)    
except ValueError:
    my_auprc_array.append(np.nan)
except IndexError:
    my_auprc_array.append(np.nan)        
indet_array.append(0)
threshold_array.append(0.4)


#print('Feature ranking for GBC')
#for i in range(0,11):
#    key = feats[i][0]
#    val = feats[i][1]
#    print('%10s:    %0.2f' % (key, val))
#
#for key in feats: 
#    print('%5s %0.2f'%(key, feats[0](key)))
    
print('Now making predictions for LOG')
LOG_model = joblib.load('LOG.joblib')
LOG_probs = LOG_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]])
LOG_probs = LOG_probs[:,1]
LOG_fprs_py, LOG_tprs_py, threshold = roc_curve(Y_test/4, LOG_probs)
LOG_precs_py, LOG_recs_py, thresholds = precision_recall_curve(Y_test/4, LOG_probs)
LOG_preds = (LOG_probs > 0.4)*4
LOG_cm = my_confusion_matrix(Y_test,LOG_preds)
cm_array.append(LOG_cm)
name_array.append('LOG')
try:
    auroc_array.append(roc_auc_score(Y_test/4, LOG_probs)) # Python calculated AUROC
except ValueError:
    auroc_array.append(np.nan)
    
try: 
    LOG_AUROC, LOG_fprs, LOG_tprs = my_auroc_prob(LOG_probs, Y_test) # Manually calculated AUROC
    my_auroc_array.append(LOG_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
    
try:
    auprc_array.append(auc(LOG_recs_py, LOG_precs_py))
except ValueError:
    auprc_array.append(np.nan)

try:
    LOG_AUPRC, LOG_recs, LOG_precs = my_auprc_prob(LOG_probs, Y_test)
    my_auprc_array.append(LOG_AUPRC)    
except ValueError:
    my_auprc_array.append(np.nan)    
except IndexError:
    my_auprc_array.append(np.nan)
indet_array.append(0)
threshold_array.append(0.4)

print('Now making predictions for KNN')
KNN_model = joblib.load('KNN.joblib')
KNN_probs = KNN_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
KNN_probs = KNN_probs[:,1]
KNN_fprs_py, KNN_tprs_py, threshold = roc_curve(Y_test/4, KNN_probs)
KNN_precs_py, KNN_recs_py, thresholds = precision_recall_curve(Y_test/4, KNN_probs)
KNN_preds = (KNN_probs > 0.4)*4
KNN_cm = my_confusion_matrix(Y_test,KNN_preds)
cm_array.append(KNN_cm)
name_array.append('KNN')
try:
    auroc_array.append(roc_auc_score(Y_test/4, KNN_probs)) # Python calculated AUROC
except ValueError:
    auroc_array.append(np.nan)
    
try: 
    KNN_AUROC, KNN_fprs, KNN_tprs = my_auroc_prob(KNN_probs, Y_test) # Manually calculated AUROC
    my_auroc_array.append(KNN_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
    
try:
    auprc_array.append(auc(KNN_recs_py, KNN_precs_py))
except ValueError:
    auprc_array.append(np.nan)

try:
    KNN_AUPRC, KNN_recs, KNN_precs = my_auprc_prob(KNN_probs, Y_test)
    my_auprc_array.append(KNN_AUPRC)    
except ValueError:
    my_auprc_array.append(np.nan)  
except IndexError:
    my_auprc_array.append(np.nan)
indet_array.append(0)
threshold_array.append(0.4)

print('Now making predictions for MLP')
MLP_model = joblib.load('MLP.joblib')
MLP_probs = MLP_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 8, 9, 10, 11]])
MLP_probs = MLP_probs[:,1]
MLP_fprs_py, MLP_tprs_py, threshold = roc_curve(Y_test/4, MLP_probs)
MLP_precs_py, MLP_recs_py, thresholds = precision_recall_curve(Y_test/4, MLP_probs)
MLP_preds = (MLP_probs > 0.4)*4
MLP_cm = my_confusion_matrix(Y_test,MLP_preds)
cm_array.append(MLP_cm)
name_array.append('MLP')
try:
    auroc_array.append(roc_auc_score(Y_test/4, MLP_probs)) # Python calculated AUROC
except ValueError:
    auroc_array.append(np.nan) 
try: 
    MLP_AUROC, MLP_fprs, MLP_tprs = my_auroc_prob(MLP_probs, Y_test) # Manually calculated AUROC
    my_auroc_array.append(MLP_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
try:
    auprc_array.append(auc(MLP_recs_py, MLP_precs_py))
except ValueError:
    auprc_array.append(np.nan)
try:
    MLP_AUPRC, MLP_recs, MLP_precs = my_auprc_prob(MLP_probs, Y_test)
    my_auprc_array.append(MLP_AUPRC)    
except ValueError:
    my_auprc_array.append(np.nan)  
except IndexError:
    my_auprc_array.append(np.nan)
    
indet_array.append(0)
threshold_array.append(0.4)

print('Now making predictions for APRI')
APRI_preds, APRI_probs, APRI_test, APRI_indet_count, APRI_values = APRI_class(X_test_imp[:,[0, 5, 9]], Y_test)
APRI_precs_py, APRI_recs_py, thresholds = precision_recall_curve(Y_test/4, APRI_probs)
APRI_cm = my_confusion_matrix(APRI_test,APRI_preds)
cm_array.append(APRI_cm)
name_array.append('APRI')
try:
    auroc_array.append(roc_auc_score(Y_test/4, APRI_probs))
except ValueError:
    auroc_array.append(np.nan)
try: 
    APRI_AUROC, APRI_fprs, APRI_tprs = my_auroc_non_prob(APRI_values, APRI_test)
    my_auroc_array.append(APRI_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
try:
    auprc_array.append(auc(APRI_recs_py, APRI_precs_py))
except ValueError:
    auprc_array.append(np.nan)
try:
    APRI_AUPRC, APRI_recs, APRI_precs, APRI_prc_curve = my_auprc_non_prob(APRI_values, APRI_test)
    my_auprc_array.append(APRI_AUPRC) 
except ValueError:
    my_auprc_array.append(np.nan)    
except IndexError:
    my_auprc_array.append(np.nan)
    
indet_array.append(APRI_indet_count/len(Y_test))
threshold_array.append('n/a')

print('Now making predictions for FIB4')
FIB4_preds, FIB4_probs, FIB4_test, FIB4_indet_count, FIB4_values = FIB4_class(X_test_imp[:,[1, 4, 5, 9]], Y_test)
FIB4_precs_py, FIB4_recs_py, thresholds = precision_recall_curve(Y_test/4, FIB4_probs)
FIB4_cm = my_confusion_matrix(FIB4_test,FIB4_preds)
cm_array.append(FIB4_cm)
name_array.append('FIB4')
try:
    auroc_array.append(roc_auc_score(Y_test/4, FIB4_probs))
except ValueError:
    auroc_array.append(np.nan)
try: 
    FIB4_AUROC, FIB4_fprs, FIB4_tprs = my_auroc_non_prob(FIB4_values, FIB4_test)
    my_auroc_array.append(FIB4_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
try:
    auprc_array.append(auc(FIB4_recs_py, FIB4_precs_py))
except ValueError:
    auprc_array.append(np.nan)
try:
    FIB4_AUPRC, FIB4_recs, FIB4_precs, FIB4_prc_curve = my_auprc_non_prob(FIB4_values, FIB4_test)
    my_auprc_array.append(FIB4_AUPRC) 
except ValueError:
    my_auprc_array.append(np.nan)    
except IndexError:
    my_auprc_array.append(np.nan)
    
indet_array.append(FIB4_indet_count/len(Y_test))
threshold_array.append('n/a')

print('Now making predictions for NAFLD')
NAFLD_preds, NAFLD_probs, NAFLD_test, NAFLD_indet_count, NAFLD_values = NAFLD_class(X_test_imp[:,[1, 2, 4, 5, 9, 10, 11]], Y_test)
NAFLD_precs_py, NAFLD_recs_py, thresholds = precision_recall_curve(Y_test/4, NAFLD_probs)
NAFLD_cm = my_confusion_matrix(NAFLD_test,NAFLD_preds)
cm_array.append(NAFLD_cm)
name_array.append('NAFLD')
try:
    auroc_array.append(roc_auc_score(Y_test/4, NAFLD_probs))
except ValueError:
    auroc_array.append(np.nan)
try: 
    NAFLD_AUROC, NAFLD_fprs, NAFLD_tprs = my_auroc_non_prob(NAFLD_values, NAFLD_test)
    my_auroc_array.append(NAFLD_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
try:
    auprc_array.append(auc(NAFLD_recs_py, NAFLD_precs_py))
except ValueError:
    auprc_array.append(np.nan)
try:
    NAFLD_AUPRC, NAFLD_recs, NAFLD_precs, NAFLD_prc_curve = my_auprc_non_prob(NAFLD_values, NAFLD_test)
    my_auprc_array.append(NAFLD_AUPRC) 
except ValueError:
    my_auprc_array.append(np.nan)    
except IndexError:
    my_auprc_array.append(np.nan)
    
indet_array.append(NAFLD_indet_count/len(Y_test))
threshold_array.append('n/a')

#NAFLD_cm = confusion_matrix(NAFLD_test,NAFLD_preds)
#cm_array.append(NAFLD_cm)
#name_array.append('NAFLD')
#auroc_array.append(roc_auc_score(Y_test/4, NAFLD_probs))
#my_auroc_array.append(my_auroc_non_prob(NAFLD_values, NAFLD_test))
#indet_array.append(NAFLD_indet_count/len(Y_test))
#threshold_array.append('n/a')

thresh = 0.45
low_indet_range = 0.2
high_indet_range = 0.0

ENS1_params = joblib.load('ENS1.joblib')
ENS1_params['threshold'] = thresh
ENS1_params['indets'] = 'None'
ENS1_preds, ENS1_probs, ENS1_test, ENS1_indet_count, ENS1_det_indices = ENS_class(SVM_probs, RFC_probs, GBC_probs, LOG_probs, KNN_probs, MLP_probs, APRI_probs, FIB4_probs, X_test_imp_scl, Y_test, ENS1_params, exc_alg)
ENS1_test = np.array(ENS1_test)
ENS1_fprs_py, ENS1_tprs_py, threshold = roc_curve(ENS1_test/4, ENS1_probs)
ENS1_precs_py, ENS1_recs_py, thresholds = precision_recall_curve(ENS1_test/4, ENS1_probs)
ENS1_cm = my_confusion_matrix(ENS1_test, ENS1_preds)
cm_array.append(ENS1_cm)
name_array.append('ENS1')

try:
    auroc_array.append(roc_auc_score(np.asarray(ENS1_test)/4, ENS1_probs))
except ValueError:
    auroc_array.append(np.nan)
try: 
    ENS1_AUROC, ENS1_fprs, ENS1_tprs = my_auroc_prob(ENS1_probs, ENS1_test)
    my_auroc_array.append(ENS1_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
try:
    auprc_array.append(auc(ENS1_recs_py, ENS1_precs_py))
except ValueError:
    auprc_array.append(np.nan)
try:
    ENS1_AUPRC, ENS1_recs, ENS1_precs = my_auprc_prob(ENS1_probs, ENS1_test)
    my_auprc_array.append(ENS1_AUPRC)
except ValueError:
    my_auprc_array.append(np.nan)
except IndexError:
    my_auprc_array.append(np.nan)
    
indet_array.append(ENS1_indet_count/len(Y_test))
threshold_array.append(ENS1_params['threshold'])

#ENS2_params = joblib.load('ENS2.joblib')
#ENS2_params['threshold'] = thresh
#ENS2_params['indet_range_low'] = low_indet_range
#ENS2_preds, ENS2_probs, ENS2_test, ENS2_indet_count, ENS2_det_indices = ENS_class(SVM_probs, RFC_probs, GBC_probs, LOG_probs, KNN_probs, MLP_probs, APRI_probs, FIB4_probs,  X_test_imp_scl, Y_test, ENS2_params, exc_alg)
#ENS2_cm = confusion_matrix(ENS2_test, ENS2_preds)
#cm_array.append(ENS2_cm)
#name_array.append('ENS2')
#auroc_array.append(roc_auc_score(np.asarray(ENS2_test)/4, ENS2_probs))
#my_auroc_array.append(my_auroc_prob(ENS2_probs, ENS2_test))50.0
#indet_array.append(ENS2_indet_count/len(Y_test))
#threshold_array.append(ENS2_params['threshold'])

print('Now making predictions for ENS3')
ENS3_params = joblib.load('ENS3.joblib')
ENS3_params['threshold'] = thresh
ENS3_params['indet_range_low'] = low_indet_range
ENS3_params['indet_range_high'] = high_indet_range
ENS3_params['indets'] = 'aprifib4'
ENS3_preds, ENS3_probs, ENS3_test, ENS3_indet_count, ENS3_det_indices = ENS_class(SVM_probs, RFC_probs, GBC_probs, LOG_probs, KNN_probs, MLP_probs, APRI_probs, FIB4_probs,  X_test_imp_scl, Y_test, ENS3_params, exc_alg)
ENS3_test = np.array(ENS3_test)
ENS3_fprs_py, ENS3_tprs_py, threshold = roc_curve(ENS3_test/4, ENS3_probs)
ENS3_precs_py, ENS3_recs_py, thresholds = precision_recall_curve(ENS3_test/4, ENS3_probs)
ENS3_cm = my_confusion_matrix(ENS3_test, ENS3_preds)
cm_array.append(ENS3_cm)
name_array.append('ENS3')
try:
    auroc_array.append(roc_auc_score(np.asarray(ENS3_test)/4, ENS3_probs))
except ValueError:
    auroc_array.append(np.nan)
try: 
    ENS3_AUROC, ENS3_fprs, ENS3_tprs = my_auroc_prob(ENS3_probs, ENS3_test)
    my_auroc_array.append(ENS3_AUROC)
except ValueError:
    my_auroc_array.append(np.nan)
try:
    auprc_array.append(auc(ENS3_recs_py, ENS3_precs_py))
except ValueError:
    auprc_array.append(np.nan)
try:
    ENS3_AUPRC, ENS3_recs, ENS3_precs = my_auprc_prob(ENS3_probs, ENS3_test)
    my_auprc_array.append(ENS3_AUPRC)
except ValueError:
    my_auprc_array.append(np.nan)
except IndexError:
    my_auprc_array.append(np.nan)
    
indet_array.append(ENS3_indet_count/len(Y_test))
threshold_array.append(ENS3_params['threshold'])

print(desc_string)
print('Number of records in dataset: ' + str(len(dataset)))
print('F01/F4: ' + str(len(dataset.loc[dataset['Fibrosis'] == 0])) + '/' + str(len(dataset.loc[dataset['Fibrosis'] == 4])))
write_performance_metrics(cm_array, name_array, auroc_array, my_auroc_array, auprc_array, my_auprc_array, indet_array, threshold_array, exc_alg, no_show)

#  Adding point (1,0) to PRC curves
SVM_recs_py = np.insert(SVM_recs_py, 0, np.array([1]))
SVM_precs_py = np.insert(SVM_precs_py, 0, np.array([0]))

RFC_recs_py = np.insert(RFC_recs_py, 0, np.array([1]))
RFC_precs_py = np.insert(RFC_precs_py, 0, np.array([0]))

GBC_recs_py = np.insert(GBC_recs_py, 0, np.array([1]))
GBC_precs_py = np.insert(GBC_precs_py, 0, np.array([0]))

LOG_recs_py = np.insert(LOG_recs_py, 0, np.array([1]))
LOG_precs_py = np.insert(LOG_precs_py, 0, np.array([0]))

KNN_recs_py = np.insert(KNN_recs_py, 0, np.array([1]))
KNN_precs_py = np.insert(KNN_precs_py, 0, np.array([0]))

MLP_recs_py = np.insert(MLP_recs_py, 0, np.array([1]))
MLP_precs_py = np.insert(MLP_precs_py, 0, np.array([0]))
#
plt.rcParams['figure.figsize'] = (4,3.5)

# Plotting ROC and PRC curves - Python Calculations
#plt.title('ROC curve for Toronto dataset')
#plt.title('Figure 1 c) ROC curve for ' + data + ' dataset')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
#plt.plot(SVM_fprs_py, SVM_tprs_py, label=('SVM, auc=%0.2f'% (SVM_AUROC)))
#plt.plot(RFC_fprs_py, RFC_tprs_py, label=('RFC, auc=%0.2f'% (RFC_AUROC)))
#plt.plot(GBC_fprs_py, GBC_tprs_py, label=('GBC, auc=%0.2f'% (GBC_AUROC)))
#plt.plot(LOG_fprs_py, LOG_tprs_py, label=('LOG, auc=%0.2f'% (LOG_AUROC)))
#plt.plot(MLP_fprs_py, MLP_tprs_py, label=('MLP, auc=%0.2f'% (MLP_AUROC)))
plt.plot(APRI_fprs, APRI_tprs, color='cyan', label=('APRI\n0.741' % (APRI_AUROC)))
plt.plot(FIB4_fprs, FIB4_tprs, color='lightblue', label=('FIB-4\n0.712'  % (FIB4_AUROC)))
#plt.plot(ENS1_fprs_py, ENS1_tprs_py, color='orange', linewidth=2, label=(('ENS1'.center(20,' ') + '\nAUROC=%0.3f'.rjust(10, ' ')) % (ENS1_AUROC)))
plt.plot(ENS3_fprs_py, ENS3_tprs_py, color='red', linewidth=2, label=('ENS2\n0.762' % (ENS3_AUROC)))
plt.plot([0,1],[0,1], 'k--', linewidth=1)
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol=4)
plt.grid(True, axis='both', color='gray', linestyle='--')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
frame1 = plt.gca()
frame1.set_facecolor('white')
plt.show()

#plt.title('PRC curve for ' + data + ' dataset')
#plt.title('Figure 1 d) PRC curve for ' + data + ' dataset')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (PPV)')
#plt.plot(SVM_recs_py, SVM_precs_py, label=('SVM, auc=%0.2f'% (auc(SVM_recs_py, SVM_precs_py, reorder=True))))
#plt.plot(RFC_recs_py, RFC_precs_py, label=('RFC, auc=%0.2f'% (auc(RFC_recs_py, RFC_precs_py    ))))		
#plt.plot(GBC_recs_py, GBC_precs_py, label=('GBC, auc=%0.2f'% (auc(GBC_recs_py, GBC_precs_py))))
#plt.plot(LOG_recs_py, LOG_precs_py, label=('LOG, auc=%0.2f'% (auc(LOG_recs_py, LOG_precs_py))))
#plt.plot(MLP_recs_py, MLP_precs_py, label=('MLP, auc=%0.2f'% (auc(MLP_recs_py, MLP_precs_py))))

plt.plot(APRI_recs, APRI_precs, color='cyan', label=('APRI\n0.324' % (APRI_AUPRC)))
plt.plot(FIB4_recs, FIB4_precs, color='lightblue', label=('FIB-4\n0.597' % (FIB4_AUPRC)))
#plt.plot(ENS1_recs_py, ENS1_precs_py, color = 'orange', linewidth=2, label=(('ENS1'.center(20,' ') + '\nAUPRC=%0.3f'.rjust(10, ' ')) % (auc(ENS1_recs_py, ENS1_precs_py))))
plt.plot(ENS3_recs_py, ENS3_precs_py, color = 'red', linewidth=2, label=('ENS2\n0.629' % (auc(ENS3_recs_py, ENS3_precs_py))))
plt.plot([0,1],[1,0], 'k--', linewidth=1)
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2), ncol=4)
plt.grid(True, axis='both', color='gray', linestyle='--')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.01])
frame1 = plt.gca()
frame1.set_facecolor('white')
plt.show()