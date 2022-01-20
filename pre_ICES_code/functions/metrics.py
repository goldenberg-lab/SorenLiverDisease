import pandas as pd
import numpy as np
import math
from beautifultable import BeautifulTable
import itertools 
import matplotlib.pyplot as plt

class metrics(object):
    accuracy = 0; 
    precision = 0;
    sensitivity = 0; 
    specificity = 0; 
    falsePosRate= 0;
    falseNegRate= 0;
    f1 = 0;
    negPredVal = 0;

def my_confusion_matrix(truth, pred):
    TN_count = 0
    TP_count = 0
    FP_count = 0
    FN_count = 0
    
    for i in range(0, len(truth)):
        if (pred[i] == 0 and truth[i] == 0):
            #print(str(i) + ". " + " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "                   TN")
            TN_count += 1
        elif(pred[i] == 4 and truth[i] == 4):
            #print(str(i) + ". "  + " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "             TP")
            TP_count += 1
        elif(pred[i] == 4 and truth[i] == 0):
            #print(str(i) + ". "+ " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "          FP")
            FP_count += 1
        elif(pred[i] == 0 and truth[i] == 4):
            #print(str(i) + ". " + " guessed: " + str(pred[i]) + " but it was " + str(truth[i]) + "   FN")
            FN_count += 1

    cm = np.ndarray(shape=(2,2))
    cm[1,1] = TP_count 
    cm[0,1] = FP_count
    cm[0,0] = TN_count
    cm[1,0] = FN_count
    return cm

def calculate_metrics(count, algorithm):
    cm = algorithm.cm[count]
    
    tot = cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1] 
    tp = cm[1,1]
    fp = cm[0,1]
    tn = cm[0,0]
    fn = cm[1,0]
    
    met = metrics()

    met.accuracy = (tp + tn)/(tot)
    met.precision = tp/(tp + fp)
    met.negPredVal = tn/(tn + fn)
    met.sensitivity = tp/(tp + fn)
    met.specificity = tn/(tn + fp)
    met.falsePosRate = fp/(fp + tn)
    met.falseNegRate = fn/(fn + tp)
    met.f1 = 2*met.precision*met.sensitivity/(met.precision+met.sensitivity)

    if (math.isnan(met.f1) == False):
        algorithm.f1s.append(met.f1)
    if (math.isnan(met.precision) == False):
        algorithm.precs.append(met.precision)
    if (math.isnan(met.negPredVal) == False):
        algorithm.npvs.append(met.negPredVal)
    if (math.isnan(met.sensitivity) == False):
        algorithm.sens.append(met.sensitivity)
    if (math.isnan(met.specificity) == False):
        algorithm.specs.append(met.specificity)
    if (math.isnan(met.accuracy) == False):
        algorithm.accs.append(met.accuracy)
    if (math.isnan(met.falsePosRate) == False):
        algorithm.fps.append(met.falsePosRate)
    if (math.isnan(met.falseNegRate) == False):
        algorithm.fns.append(met.falseNegRate)
    
    print('')
#    print('TP count: {}'.format(tp))
#    print('TN count: {}'.format(tn))
#    print('FP count: {}'.format(fp))
#    print('FN count: {}'.format(fn))
    print('DATASET WITH {} CLASSIFIER:'.format(algorithm.name))
    print('F1:            {} %'.format(round(met.f1*100,2)))
    print('Sensitivity:   {} %'.format(round(met.sensitivity*100,2)))
    print('Specificity:   {} %'.format(round(met.specificity*100,2)))
    print('PPV:           {} %'.format(round(met.precision*100,2)))
    print('NPV:           {} %'.format(round(met.negPredVal*100,2)))
    print('Accuracy:      {} %'.format(round(met.accuracy*100,2)))
    print('FalsePosRate:  {} %'.format(round(met.falsePosRate*100,2)))
    print('FalseNegRate:  {} %'.format(round(met.falseNegRate*100,2)))
    print('')
    return met

def auroc_and_auprc_non_prob(test_values, Yp_test, obj):

    from shapely.geometry import Polygon 
    fprs = [] # False Positive Rates
    tprs = [] # True Positive Rates
    precs = [] # Precisions
    threshs = [] # Thresholds corresponding to each point 
    
    thresholds = np.linspace(0, np.max(test_values))

    for t in thresholds:
        Yp_pred = (test_values >= t) * 4
        cm = my_confusion_matrix(Yp_test, Yp_pred)

        tp = cm[1,1]
        fp = cm[0,1]
        tn = cm[0,0]
        fn = cm[1,0]
       
        fprs.append(fp/(fp + tn))
        tprs.append(tp/(tp + fn))
        precs.append(tp/(tp+fp))
        threshs.append(t)
        
    obj.manual_fprs.append(fprs)
    obj.manual_tprs.append(tprs)
    obj.manual_precs.append(precs)
    
    roc_curve = [];
    roc_polygon = [(1,0)] #creates a empty list where we will append the points to create the polygon
    prc_curve = [];
    prc_polygon = [(0,0),(1,0)] 
    
    for i in range(0,len(fprs)-1):
        fpr = fprs[i]
        tpr = tprs[i]
        pre = precs[i]
        
        if (np.isnan(fpr)==False and np.isnan(tpr) == False):
            roc_curve.append((fpr, tpr))
            roc_polygon.append([fpr,tpr]) #append all xy points for auroc
            
        if (np.isnan(tpr)==False and np.isnan(pre) == False):
            prc_curve.append((tpr, pre))
            prc_polygon.append([tpr,pre]) #append all xy points for auprc
    
    roc_polygon_shape = Polygon(roc_polygon)
    prc_polygon_shape = Polygon(prc_polygon)

    auroc = roc_polygon_shape.area
    auprc = prc_polygon_shape.area
    
    obj.manual_aurocs.append(auroc)
    obj.manual_auprcs.append(auprc)   

def auroc_and_auprc_prob(probs, Yp_test, obj):
    if(obj.name=="ASTALT" or obj.name=="APRI" or obj.name=="FIB4"):
        return 
    
    from shapely.geometry import Polygon 
    fprs = [] # False Positive Rates
    tprs = [] # True Positive Rates
    precs = [] # Precisions
    
    for i in range(0,101):
        thresh = i/100; 
        Yp_pred =  (np.asarray(probs) > np.asarray(thresh)) * 4
        cm = my_confusion_matrix(Yp_test, Yp_pred)
        
        tp = cm[1,1]
        fp = cm[0,1]
        tn = cm[0,0]
        fn = cm[1,0]
        
        truePositiveRate = tp/(tp + fn)
        falsePositiveRate = fp/(fp + tn)
        precision = tp/(tp+fp)

        fprs.append(falsePositiveRate)
        tprs.append(truePositiveRate)
        precs.append(precision)

    obj.manual_fprs.append(fprs)
    obj.manual_tprs.append(tprs)
    obj.manual_precs.append(precs)
        
    roc_curve = [];
    roc_polygon = [(1,0)] #creates a empty list where we will append the points to create the polygon
    prc_curve = [];
    prc_polygon = [(0,0),(1,0)] 
    
    for i in range(0,len(fprs)-1):
        fpr = fprs[i]
        tpr = tprs[i]
        pre = precs[i]
        
        if (np.isnan(fpr)==False and np.isnan(tpr) == False):
            roc_curve.append((fpr, tpr))
            roc_polygon.append([fpr,tpr]) #append all xy points for auroc
       
        if (np.isnan(tpr)==False and np.isnan(pre) == False):
            prc_curve.append((tpr, pre))
            prc_polygon.append([tpr,pre]) #append all xy points for auprc
    
    roc_polygon_shape = Polygon(roc_polygon)
    prc_polygon_shape = Polygon(prc_polygon)
    
    auroc = roc_polygon_shape.area
    auprc = prc_polygon_shape.area
    
    obj.manual_aurocs.append(auroc)
    obj.manual_auprcs.append(auprc)
#    print(auroc)
#    print(auprc)
    
#    plt.figure(1)
#    plt.title('Receiver Operating Characteristic Curve')
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.scatter(svmObj.manual_fprs[1],svmObj.manual_tprs[1])
#    plt.axis([0,1,0,1])
#    plt.show()
#    
#    plt.figure(2)
#    plt.title('Precision-Recall Curve')
#    plt.xlabel('Recall/Sensitivity')
#    plt.ylabel('Precision')
#    plt.scatter(svmObj.manual_tprs[1],svmObj.manual_precs[1])
#    plt.axis([0,1,0,1])
#    plt.show()

def check_performance_metrics(obj):
    for f in range(0, obj.folds+1):
        print(obj.preds[f])
        print(obj.tests[f])
        my_confusion_matrix(obj.tests[f], obj.preds[f])
        calculate_metrics(f, obj)

def print_results(clfObj):
    for obj in clfObj:
        if (obj.isUsed == True):
            print('--------------------------- ' + obj.name + '--------------------------- ')
            print(obj.params)
            print(obj.features)
            print("F1:              %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.f1s)*100, np.std(obj.f1s)*100)) 
            print("Sensitivity:     %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.sens)*100, np.std(obj.sens)*100)) 
            print("Specificity:     %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.specs)*100, np.std(obj.specs)*100))  
            print("Pos Pred Val:   %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.precs)*100, np.std(obj.precs)*100))   
            print("Neg Pred Val:             %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.npvs)*100, np.std(obj.npvs)*100))   
            print("Accuracy:        %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.accs)*100, np.std(obj.accs)*100)) 
            print("False Neg Rate:  %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.fns)*100, np.std(obj.fns)*100))
            print("False Pos Rate:  %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.fps)*100, np.std(obj.fps)*100))
            print("Python AUROC:    %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.aucs)*100, np.std(obj.aucs)*100)) 
            print("Manual AUROC:    %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.manual_aurocs)*100, np.std(obj.manual_aurocs)*100))
            print("Manual AUPRC:    %0.2f%%  (+/- %0.2f%%)" % (np.mean(obj.manual_auprcs)*100, np.std(obj.manual_auprcs)*100))            
            print()     
            print("Cost function score: %0.2f%%" % (np.mean(obj.accs)*100))
            obj.best_score = (np.mean(obj.accs)*100)
            # (np.mean(obj.sens)*100 
            # + np.mean(obj.specs)*100
def print_table(objArray, total, uncertainty):
    headers = [" "]
    f1Row = ['F1']
    sensRow = ['Sensitivity']
    specsRow = ['Specificity']
    precsRow = ['Precision/PPV']
    npvsRow= ['NPV']
    accsRow = ['Accuracy']
    fnsRow = ['False Neg Rate']
    fpsRow = ['False Pos Rate']
    aurocRow = ['Python AUROC']
    manAUROCRow = ['Manual AUROC']
    manAUPRCRow = ['Manual AUPRC']
    indetCountRow = ['% indet']
    table_width = 0
    oc = 1
    
    for obj in objArray:

        if (obj.isUsed == True):
            headers.append(obj.name)
            if (uncertainty == True):
                f1Row.append(('%.2f' % (np.mean(obj.f1s)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.f1s)*100)))
                sensRow.append(('%.2f' % (np.mean(obj.sens)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.sens)*100)))
                specsRow.append(('%.2f' % (np.mean(obj.specs)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.specs)*100)))
                precsRow.append(('%.2f' % (np.mean(obj.precs)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.precs)*100)))
                npvsRow.append(('%.2f' % (np.mean(obj.npvs)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.npvs)*100)))
                accsRow.append(('%.2f' % (np.mean(obj.accs)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.accs)*100)))
                fnsRow.append(('%.2f' % (np.mean(obj.fns)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.fns)*100)))
                fpsRow.append(('%.2f' % (np.mean(obj.fps)*100)) + ' +/- ' + ('%.2f' % (np.std(obj.fps)*100)))
                aurocRow.append(('%.2f' % (np.mean(obj.aucs))) + ' +/- ' + ('%.2f' % (np.std(obj.aucs))))
                manAUROCRow.append(('%.2f' % (np.mean(obj.manual_aurocs))) + ' +/- ' + ('%.2f' % (np.std(obj.manual_aurocs))))
                manAUPRCRow.append(('%.2f' % (np.mean(obj.manual_auprcs))) + ' +/- ' + ('%.2f' % (np.std(obj.manual_auprcs))))
                indetCountRow.append(( '%0.2f' % (100*obj.indeterminate_count/total)))
            else:
                f1Row.append((np.mean(obj.f1s)*100))
                sensRow.append((np.mean(obj.sens)*100))
                specsRow.append((np.mean(obj.specs)*100))
                precsRow.append((np.mean(obj.precs)*100))
                npvsRow.append((np.mean(obj.precs)*100))
                accsRow.append((np.mean(obj.accs)*100))
                fnsRow.append((np.mean(obj.fns)*100))
                fpsRow.append((np.mean(obj.fps)*100))
                aurocRow.append((np.mean(obj.aucs)))
                manAUROCRow.append((np.mean(obj.manual_aurocs)))
                manAUPRCRow.append((np.mean(obj.manual_auprcs)))
                indetCountRow.append(( '%0.2f' % (100*obj.indeterminate_count/total)))
            table_width += max(len(f1Row[oc]),len(sensRow[oc]), len(specsRow[oc]), len(precsRow[oc]), len(accsRow[oc]), len(fnsRow[oc]), len(fpsRow[oc]), len(aurocRow[oc]), len(manAUROCRow[oc]), len(manAUPRCRow[oc])) + 2
           
    
    table = BeautifulTable(max_width=300)#table_width)
    table.column_headers = headers
    table.append_row(f1Row)
    table.append_row(sensRow)
    table.append_row(specsRow)
    table.append_row(precsRow)
    table.append_row(npvsRow)
    table.append_row(accsRow)
    table.append_row(fnsRow)
    table.append_row(fpsRow)
    table.append_row(aurocRow)
    table.append_row(manAUROCRow)
    table.append_row(manAUPRCRow)
    table.append_row(indetCountRow)
    print(table)

def find_misclassifications(dataset, algArray):
    correct_indxs = []
    mis_F01_indxs = []
    mis_F4_indxs = []
    
    for c in range(0, len(dataset.Y_ts)-1):
        for i in range(0, len(dataset.Y_ts[c])-1):
            all_misclassified = True
            for alg in algArray:
                if (alg.name == 'APRI' or alg.name=='ASTALT' or alg.name=='FIB4'):
                    continue
                elif (alg.isUsed == True):
                    if (alg.preds[c][i] == dataset.Y_ts[c][i]):
                        all_misclassified = False
            if (all_misclassified == True):
                if (dataset.Y_ts[c][i] == 4):
                    mis_F4_indxs.append(dataset.ts_indxs[c][i])
                elif (dataset.Y_ts[c][i] == 0):
                    mis_F01_indxs.append(dataset.ts_indxs[c][i])
            else:
                correct_indxs.append(dataset.ts_indxs[c][i])
    dataset.correct_class_indxs = correct_indxs
    dataset.mis_F4_class_indxs = mis_F4_indxs
    dataset.mis_F01_class_indxs = mis_F01_indxs
    print('# of misclassified F4s: ' + str(len(mis_F4_indxs)))
    print('# of misclassified F01s: ' + str(len(mis_F01_indxs)))

def plot_heat_map(dataset, alg1Array):
    heat_map = np.zeros([1, len(dataset.Y)])
    import itertools
    alg1_count = 0    
    alg1_labels=[]
    
    for alg1 in alg1Array:
        if (alg1.name == 'APRI' or alg1.name=='ASTALT' or alg1.name=='FIB4' or alg1.isUsed == False):
            continue
        heat_map = np.resize(heat_map, [np.size(heat_map,0)+1, np.size(heat_map,1)])
        heat_map[alg1_count,:] = list(itertools.chain.from_iterable(alg1.preds))
        alg1_count += 1
        alg1_labels.append(alg1.name)
    alg1_labels.append('Truth')
    print(alg1_labels)
    heat_map[alg1_count, dataset.mis_F01_class_indxs] = 8
    heat_map[alg1_count, dataset.mis_F4_class_indxs] = 8
 
    dataset.heat_map = pd.DataFrame(heat_map)
    # Values from 0-2 will be green and 2-5 will be red
    import matplotlib.colors as mcolors
    cmap, norm = mcolors.from_levels_and_colors([0, 2, 4.1, 9], ['lightblue', 'yellow', 'red'])
    plt.pcolor(dataset.heat_map, cmap=cmap, norm=norm)
    plt.yticks(np.arange(0.5, alg1_count+1 , 1), alg1_labels) #len(heat_map_df.index)
    plt.ylabel('Algorithm')
    plt.xlabel('Record Number')
    plt.show()       
    
        