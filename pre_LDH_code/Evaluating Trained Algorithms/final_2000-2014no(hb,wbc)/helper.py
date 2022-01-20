import numpy as np
np.set_printoptions(precision=3)
from sklearn.metrics import roc_curve 
from sklearn.metrics import precision_recall_curve 
from sklearn.utils import resample 
from sklearn.metrics import auc 
from numpy import trapz
from beautifultable import BeautifulTable

def area_by_shoelace(x,y):
    return abs(sum(i*j for i, j in zip(x,y[1:])) + x[-1]*y[0] - sum(i*j for i,j in zip(x[1:],y)) - x[0]*y[-1])/2

def show(text):
	print(text)
	pause()

def pause():
	input('Press enter to continue!')
    
def my_old_confusion_matrix(truth, pred):
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

def APRI_class(Xp_test, Yp_test):
	indet_count = 0
	Yp_pred_new = []
	Yp_prob_new = []
	Yp_test_new = []
	det_indices = []
	apri_values = []

	for i in range(0, len(Xp_test)):
		AST_upper = 35 
		AST = Xp_test[i,1]
		Plt = Xp_test[i,2]
		APRI = (100*AST/AST_upper)/(Plt)
        
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
    
def my_auprc_non_prob(test_values, Yp_test):
    thresholds = np.linspace(0, np.max(test_values)+0.5, 1000)
    
    tprs = []
    precs = []
    
    for t in thresholds: 
        Yp_pred = (test_values >= t)*4

        cm = my_old_confusion_matrix(Yp_test, Yp_pred)
		
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
    sorted_tprs = []
    sorted_precs = []
    
    for i in range(0,len(prc_curve)-1):
        sorted_tprs.append(prc_curve[i][0])
        sorted_precs.append(prc_curve[i][1])
    auprc = auc(sorted_tprs, sorted_precs, reorder=True)

    return auprc1, sorted_tprs, sorted_precs, prc_curve

def my_auroc_non_prob(test_values, Yp_test):
	thresholds = np.linspace(np.min(test_values),np.max(test_values),1000)

	fprs = []
	tprs = []
    
	for t in thresholds:
		Yp_pred = (test_values >= t)*4
		cm = my_old_confusion_matrix(Yp_test, Yp_pred)
		
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
	tprs = []
	precs = []
	threshs = []

	for i in range(0,250):
		thresh = i/250
		Yp_pred = (np.asarray(probs) > np.asarray(thresh))*4
		cm = my_old_confusion_matrix(Yp_test, Yp_pred)
		
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
		cm = my_old_confusion_matrix(Yp_test, Yp_pred)
        		
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