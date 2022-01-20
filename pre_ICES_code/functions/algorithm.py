import numpy as np
from get_features import rank_feature_importance
from metrics import my_confusion_matrix 
from metrics import calculate_metrics
from metrics import auroc_and_auprc_prob
from metrics import auroc_and_auprc_non_prob
from sklearn.metrics import roc_auc_score # For calculating AUROC
import matplotlib.pyplot as plt
from shapely.geometry import Polygon 

class Alg(object): 
    def __init__(self):
        self.name = ""
        self.params = {};  # Hyperparmaters used to tune the algorithm. 
        self.preds = []; # Predictions vector
        self.probs = []; # Probabilities for each prediction 
        self.tests = []; # Truth vector 
        self.cm    = []; # Confusion matrix
        self.f1s   = []; # F1 scores from each run 
        self.precs = []; # Precisions from each run 
        self.npvs = []; 
        self.sens = []; # Sensitivity/Recall from each run 
        self.specs = []; # Specificity from each run 
        self.accs  = []; # Accuracies from each run
        self.fps   = []; # False positive rate from each run 
        self.fns   = []; # False negative rate from each run 
        self.best_score = []; 
        self.aucs  = []; # AUROCS 
        self.manual_fprs = [];
        self.manual_tprs = [];
        self.manual_precs = []; 
        self.manual_aurocs = []; 
        self.manual_auprcs = [];
        self.features = 0; # Features used by the algorithm 
        self.best_features=[] # Best features, important for RFC and GBC
        self.determinate_indices=[] # Determinante indices, important for comparing performance to APRI & FIB4
        self.isUsed = False
        self.folds = 0
        self.indeterminate_count = 0
    
def reset_algorithm(obj):
    for alg in obj: 
        alg.preds = []; # Predictions vector
        alg.probs = []; # Probabilities for each prediction 
        alg.tests = []; # Truth vector 
        alg.cm    = []; # Confusion matrix
        alg.f1s   = []; # F1 scores from each run 
        alg.precs = []; # Precisions from each run 
        alg.npvs = [];
        alg.sens = []; # Sensitivity/Recall from each run 
        alg.specs = []; # Specificity from each run 
        alg.accs  = []; # Accuracies from each run
        alg.fps   = []; # False positive rate from each run 
        alg.fns   = []; # False negative rate from each run 
        alg.best_score = 0;
        alg.aucs  = []; # Performance metrics 
        alg.manual_fprs = []; 
        alg.manual_tprs = [];
        alg.manual_precs = []; 
        alg.manual_aurocs = []; 
        alg.manual_auprcs = [];
        alg.folds = 0;
        alg.indeterminate_count = 0
        
        if (alg.name=='myApri' or alg.name=='myFIB4' or alg.name=='ASTALT'):
            alg.params['best_threshold'] = []
        
def svm_class(Xp_train, Xp_test, Yp_train, Yp_test, svmObject, count):
    if (svmObject.isUsed == False):
        return
    
    svmObject.name = 'SVM'
    svmObject.folds = count
    
    from sklearn.svm import SVC
    svm = SVC(verbose=False, probability=True, C = svmObject.params['C'],\
              gamma = svmObject.params['gamma'], kernel=svmObject.params['kernel'],\
              degree= svmObject.params['degree'], coef0=svmObject.params['coef0'],\
              shrinking=svmObject.params['shrinking'], tol=svmObject.params['tol'])
    svm.fit(Xp_train, Yp_train)
    Yp_pred = svm.predict(Xp_test)
    probabilities = svm.predict_proba(Xp_test)
    probabilities = probabilities[:,1]
    
    if (svmObject.params['method'] == 'prob'):
        Yp_pred = (probabilities > svmObject.params['threshold'])*4
    cm = my_confusion_matrix(Yp_test, Yp_pred)    
    append_results(Yp_pred, probabilities, Yp_test, roc_auc_score(Yp_test/4, probabilities), cm, count, svmObject)        

def rfc_class(Xp_train, Xp_test, Yp_train, Yp_test, rfcObject, count):
    if (rfcObject.isUsed == False):
        return
    rfcObject.name = 'RFC'
    rfcObject.folds = count
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_jobs=-1, criterion=rfcObject.params['criterion'], bootstrap=rfcObject.params['bootstrap'],\
                                 random_state=0, n_estimators=rfcObject.params['n_estimators'], max_features=rfcObject.params['max_features'],\
                                 max_depth=rfcObject.params['max_depth'], min_samples_split=rfcObject.params['min_samples_split'],\
                                 min_samples_leaf=rfcObject.params['min_samples_leaf'], min_weight_fraction_leaf=rfcObject.params['min_weight_fraction_leaf'],\
                                 max_leaf_nodes=rfcObject.params['max_leaf_nodes'], min_impurity_decrease=rfcObject.params['min_impurity_decrease'], oob_score = rfcObject.params['oob_score'])  
    rfc.fit(Xp_train, Yp_train)  
    Yp_pred = rfc.predict(Xp_test)
    probabilities = rfc.predict_proba(Xp_test)
    probabilities = probabilities[:,1] # to generate AUROC, we only need positive probabilities
    cm = my_confusion_matrix(Yp_test, Yp_pred)
    
    append_results(Yp_pred, probabilities, Yp_test, roc_auc_score(Yp_test/4, probabilities), cm, count, rfcObject)
    rfcObject.best_features = rank_feature_importance("RFC", rfc.feature_importances_, rfcObject.features, False)
      
def gbc_class(Xp_train, Xp_test, Yp_train, Yp_test, gbcObject, count):
    if (gbcObject.isUsed == False):
        return
    gbcObject.name = 'GBC'
    gbcObject.folds = count

    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier(random_state=0, n_estimators=gbcObject.params['n_estimators'], criterion=gbcObject.params['criterion'],\
                                     loss=gbcObject.params['loss'],learning_rate=gbcObject.params['learning_rate'],\
                                     max_depth=gbcObject.params['max_depth'], min_samples_split=gbcObject.params['min_samples_split'],\
                                     min_samples_leaf = gbcObject.params['min_samples_leaf'], min_weight_fraction_leaf= gbcObject.params['min_weight_fraction_leaf'],\
                                     subsample = gbcObject.params['subsample'], max_features=gbcObject.params['max_features'],\
                                     max_leaf_nodes=gbcObject.params['max_leaf_nodes'], min_impurity_decrease=gbcObject.params['min_impurity_decrease'])
    gbc.fit(Xp_train, Yp_train)
    Yp_pred = gbc.predict(Xp_test)
    probabilities = gbc.predict_proba(Xp_test)
    probabilities = probabilities[:,1] # to generate AUROC, we only need positive probabilities  
    cm = my_confusion_matrix(Yp_test, Yp_pred)

    append_results(Yp_pred, probabilities, Yp_test, roc_auc_score(Yp_test/4, probabilities), cm, count, gbcObject)
    gbcObject.best_features = rank_feature_importance("GBC", gbc.feature_importances_, gbcObject.features, False)


def log_class(Xp_train, Xp_test, Yp_train, Yp_test, logObject, count):
    if (logObject.isUsed == False):
        return   
    logObject.name='LOG'
    logObject.folds = count
    from sklearn.linear_model import LogisticRegression
    
    log = LogisticRegression(random_state = 0, max_iter =logObject.params['max_iter'], solver=logObject.params['solver'],\
                             C=logObject.params['C'], tol=logObject.params['tol'], fit_intercept=logObject.params['fit_intercept'],\
                             penalty='l2', intercept_scaling=logObject.params['intercept_scaling'], dual=logObject.params['dual'],\
                             multi_class=logObject.params['multi_class'])
    
    log.fit(Xp_train, Yp_train/4)
    Yp_pred = log.predict(Xp_test)*4        
    probabilities = log.predict_proba(Xp_test)
    probabilities = probabilities[:,1] # to generate AUROC, we only need positive probabilities
    cm = my_confusion_matrix(Yp_test, Yp_pred)

    append_results(Yp_pred, probabilities, Yp_test, roc_auc_score(Yp_test/4, probabilities), cm, count, logObject)
    
def knn_class(Xp_train, Xp_test, Yp_train, Yp_test, knnObject, count):
    if (knnObject.isUsed == False):
        return   
    knnObject.name='KNN'
    knnObject.folds = count

    from sklearn.neighbors import KNeighborsClassifier
       
    KNN = KNeighborsClassifier(n_neighbors=knnObject.params['n_neighbors'], weights=knnObject.params['weights'],\
                               algorithm=knnObject.params['algorithm'], leaf_size=knnObject.params['leaf_size'],\
                               p=knnObject.params['p'], metric=knnObject.params['metric'])

    KNN.fit(Xp_train, Yp_train)     
    Yp_pred = KNN.predict(Xp_test)
    probabilities = KNN.predict_proba(Xp_test)
    probabilities = probabilities[:,1]        
    cm = my_confusion_matrix(Yp_test, Yp_pred)

    append_results(Yp_pred, probabilities, Yp_test, roc_auc_score(Yp_test/4, probabilities), cm, count, knnObject)

def mlp_class(Xp_train, Xp_test, Yp_train, Yp_test, mlpObject, count):
    if (mlpObject.isUsed == False):
        return       
    mlpObject.name='MLP'
    mlpObject.folds = count
    from sklearn.neural_network import MLPClassifier
    
    MLP = MLPClassifier(activation=mlpObject.params['activation'],  solver=mlpObject.params['solver'],\
                        tol= mlpObject.params['tol'], hidden_layer_sizes = mlpObject.params['hidden_layer_sizes'],\
                        max_iter=mlpObject.params['max_iter'], learning_rate=mlpObject.params['learning_rate'],\
                        alpha=mlpObject.params['alpha'], batch_size=mlpObject.params['batch_size'],\
                        power_t = mlpObject.params['power_t'], shuffle=mlpObject.params['shuffle'],\
                        momentum=mlpObject.params['momentum'], nesterovs_momentum=mlpObject.params['nesterovs_momentum'],\
                        early_stopping=mlpObject.params['early_stopping'], validation_fraction = mlpObject.params['validation_fraction'],\
                        beta_1=mlpObject.params['beta_1'], beta_2=mlpObject.params['beta_2'], epsilon=mlpObject.params['epsilon'], random_state = 0)
    MLP.fit(Xp_train, Yp_train)
    Yp_pred = MLP.predict(Xp_test)
    probabilities = MLP.predict_proba(Xp_test)
    probabilities = probabilities[:,1]
    cm = my_confusion_matrix(Yp_test, Yp_pred)
    
    append_results(Yp_pred, probabilities, Yp_test, roc_auc_score(Yp_test/4, probabilities), cm, count, mlpObject) 
    
def gnb_class(Xp_train, Xp_test, Yp_train, Yp_test, gnbObject, count):
    if (gnbObject.isUsed == False):
        return       
    gnbObject.name='GNB'
    gnbObject.folds = count
    from sklearn.naive_bayes import GaussianNB
    GNB = GaussianNB()
    GNB.fit(Xp_train, Yp_train)
    Yp_pred = GNB.predict(Xp_test)
    probabilities = GNB.predict_proba(Xp_test)
    probabilities = probabilities[:,1]
    cm = my_confusion_matrix(Yp_test, Yp_pred)
    
    append_results(Yp_pred, probabilities, Yp_test, roc_auc_score(Yp_test/4, probabilities), cm, count, gnbObject)

def ens_class (s, r, g, l, k, a, b, Yp_test, ensObject, count):
    if (ensObject.isUsed == False):
        return    
    ensObject.name = 'ENS'
    ensObject.folds = count
    probabilities=[]
    Yp_pred=[]
    
    for i in range(0,np.size(Yp_test)):
        prob_sum = 0
        terms = 0
        if (s.isUsed == True):
            prob_sum += float(str(round(s.probs[count][i],8)))
            terms += 1
        if (r.isUsed == True):
            prob_sum += r.probs[count][i]
            terms += 1
        if (g.isUsed == True):
            prob_sum += g.probs[count][i]
            terms += 1
        if (l.isUsed == True):
            prob_sum += l.probs[count][i]
            terms += 1
        if (k.isUsed == True):
            prob_sum += k.probs[count][i]
            terms += 1
        if (a.isUsed == True):
            prob_sum += a.probs[count][i]
            terms += 1
        if (b.isUsed == True):
            prob_sum += b.probs[count][i]
            terms += 1
        probabilities.append(prob_sum/terms) 
#        print('Prob sum: ' + str(prob_sum))
#        print('Terms: ' + str(terms))
#        print(probabilities[i])
#        input('Batman')
        if (probabilities[i] >= ensObject.params['threshold']):
            Yp_pred.append(4);
        else:
            Yp_pred.append(0);

    cm = my_confusion_matrix(Yp_test,Yp_pred)
    
#    print('Yp_test length: ' + str(len(Yp_test)))
#    print('Probabilities length: ' + str(len(probabilities)))
#    input('Enter')
    
    float_probs = np.zeros(len(probabilities))
    for i in range(0, len(probabilities)-1):
        float_probs[i] = probabilities[i]
    
    append_results(Yp_pred, float_probs, Yp_test, roc_auc_score(Yp_test/4, probabilities), cm, count, ensObject)
    
def astalt_class (Xp_test, Yp_test, astaltObject, count):
    if (astaltObject.isUsed == False):
        return   
    astaltObject.name='ASTALT'
    astaltObject.folds = count
    Yp_pred= [];
    Yp_prob= [];
    
    astalt_values = [];

    for i in range(0, len(Xp_test)):
        astalt_values.append(Xp_test[i,1]/Xp_test[i,0])
        Yp_pred.append(4 if (Xp_test[i,1]/Xp_test[i,0] >= 1) else 0)
        Yp_prob.append(np.nan)

    auroc_and_auprc_non_prob(astalt_values, Yp_test, astaltObject)     
    cm = my_confusion_matrix(Yp_test, Yp_pred)
    append_results(Yp_pred, np.nan, Yp_test, np.nan, cm, count, astaltObject)     
    
def apri_class (Xp_test, Yp_test, apriObject, count):
    if (apriObject.isUsed == False):
        return   
    apriObject.name='APRI'
    apriObject.folds = count
        
    Yp_pred_new = [];
    Yp_test_new = [];
    Yp_prob_new = [];
    det_indices = [];
    apri_values = [];
    
    for i in range(0, len(Xp_test)):
            AST_upper = 31 if Xp_test[i,0] == 0 else 19 # Upper limit is 31 for men (0) and 19 for women
            AST = Xp_test[i,1]
            Plt = Xp_test[i,2]
            APRI = (100*AST/AST_upper)/(Plt)
            
            if (APRI >= 2):
                Yp_pred_new.append(4)
                Yp_test_new.append(Yp_test[i])
                Yp_prob_new.append(np.nan)
                det_indices.append([i])
                apri_values.append(APRI)
            elif (APRI <= 0.5):
                Yp_pred_new.append(0)
                Yp_test_new.append(Yp_test[i])
                Yp_prob_new.append(np.nan)
                det_indices.append([i])
                apri_values.append(APRI)
            else:
                apriObject.indeterminate_count += 1
    
    auroc_and_auprc_non_prob(apri_values, Yp_test_new, apriObject) 
    cm = my_confusion_matrix(Yp_test_new,Yp_pred_new)
    
    # The best threshold is stored in index.     
    append_results(Yp_pred_new, Yp_prob_new, Yp_test_new, np.nan, cm, count, apriObject)
    apriObject.determinate_indices.append(det_indices)
            
def fib4_class (Xp_test, Yp_test, fib4Object, count):
    if (fib4Object.isUsed == False):
        return   
    fib4Object.name='FIB4'
    fib4Object.folds = count
    Yp_pred_new = [];
    Yp_test_new = [];
    Yp_prob_new= [];
    det_indices = [];
    fib4_values = [];
    
    for i in range(0, len(Xp_test)):
        age = Xp_test[i,0]
        ALT = Xp_test[i,1]
        AST = Xp_test[i,2]
        Plt = Xp_test[i,3]
        fib4 = age*AST/(Plt*(ALT)**0.5)
        
        if (fib4 <= 1.45):
            Yp_pred_new.append(0)
            Yp_test_new.append(Yp_test[i])
            Yp_prob_new.append(np.nan)
            det_indices.append([i])
            fib4_values.append(fib4)
        elif (fib4 >= 3.25):
            Yp_pred_new.append(4)
            Yp_test_new.append(Yp_test[i])
            Yp_prob_new.append(np.nan)
            det_indices.append([i])
            fib4_values.append(fib4)
        else:
            fib4Object.indeterminate_count += 1
            
    auroc_and_auprc_non_prob(fib4_values, Yp_test_new, fib4Object) 
    cm = my_confusion_matrix(Yp_test_new,Yp_pred_new)
    
    # The best threshold is stored in index.     
    append_results(Yp_pred_new, Yp_prob_new, Yp_test_new, np.nan, cm, count, fib4Object)
    fib4Object.determinate_indices.append(det_indices)

def append_results(pred, prob, test, aucs, cm, count, obj):
    obj.preds.append(pred)
    obj.probs.append(prob)
    obj.tests.append(test)
    obj.aucs.append(aucs)
    auroc_and_auprc_prob(prob, test, obj)
    obj.cm.append(cm)
    calculate_metrics(count, obj)
    
    