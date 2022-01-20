import os
import sys
import importlib
os.chdir("C:\\Users\\Soren\\Desktop\\Thesis\\Data Analysis\\Code")
sys.path.insert(0, "C:\\Users\\Soren\\Desktop\\Thesis\\Data Analysis\\Code\\functions")

# Python libraries 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import math
#from fancyimpute import MICE
import numpy.core.defchararray as nd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
import itertools

import keras
from keras.models import Sequential
from keras.layers import Dense

# my libraries
import algorithm as alg
from metrics import calculate_metrics
from metrics import print_results
from metrics import print_table
from metrics import find_misclassifications
from metrics import plot_heat_map
from metrics import my_confusion_matrix
from data import dataset_class
from data import preprocessing
from data import reset_dataset
from algorithm import append_results
from get_features import get_features

toronto = dataset_class()
test1 = dataset_class()
toronto.description = 'Toronto Liver Clinic Dataset. Used to train the algorithms'
svmObj = alg.Alg(); rfcObj = alg.Alg(); gbcObj=alg.Alg(); logObj=alg.Alg(); knnObj = alg.Alg(); mlpObj = alg.Alg(); gnbObj = alg.Alg(); ensObj = alg.Alg(); apriObj = alg.Alg(); astaltObj = alg.Alg(); fib4Obj = alg.Alg(); annObj = alg.Alg() 
algorithmArray = [svmObj, rfcObj, gbcObj, logObj, knnObj, mlpObj, gnbObj, ensObj, apriObj, astaltObj, fib4Obj, annObj]

toronto.df = pd.read_excel('C:/Users/Soren/Desktop/Thesis/Data Analysis/toronto_dataset.xlsx', parse_cols="A:BK")
#toronto.df = toronto.df.drop_duplicates(subset='MRN', keep='first')
#toronto.df = toronto.df.loc[(toronto.df['TotalMissingnessWithSodiumGGTPlatelets'] < 0) & (toronto.df['DecompensatedCirrhosis'] == 0)] # & (toronto.df['Platelets'] > 0)  &  ) 
toronto.df = toronto.df.loc[(toronto.df['DecompensatedCirrhosis'] == 0)  &  (toronto.df['Albumin'] > 0) & (toronto.df['ALP'] > 0) & (toronto.df['ALT'] > 0) \
                           & (toronto.df['AST'] > 0) & (toronto.df['Bilirubin'] > 0) & (toronto.df['Creatinine'] > 0) & (toronto.df['INR'] > 0)\
                             & (toronto.df['Platelets'] > 0) & (toronto.df['BMI'] > 0) ]
toronto.df = toronto.df.sample(frac=1).reset_index(drop=True)
toronto.X = toronto.df.iloc[:,0:49].values
toronto.Y = toronto.df.iloc[:,49].values 
toronto.Y = nd.replace(nd.replace(nd.replace(toronto.Y.astype(str), 'F 4', '4'), 'F 1', '0'), 'F 0', '0').astype(int)
toronto.MRNs = toronto.df.iloc[:,51]
toronto.entryDates = toronto.df.iloc[:,52]
toronto.split = 'groupKFold' # KFold # groupKFold
dft = toronto.df

from sklearn.model_selection import GroupKFold 
from sklearn.model_selection import KFold 

kf = GroupKFold(n_splits=10)
normalKF = KFold(n_splits =10, shuffle=True, random_state=0)
kf.get_n_splits(toronto.X, toronto.Y, toronto.MRNs.astype(int))

svmObj.params={'method': 'label', 'threshold': 0.40, 'C':0.5, 'gamma':'auto', 'kernel':'rbf', 'degree':3, 'coef0':0.5, 'shrinking':True, 'tol':0.001}
rfcObj.params={'n_estimators': 100, 'criterion':'entropy', 'max_depth': None, 'bootstrap':True, 'max_features':'auto', 'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf':0, 'max_leaf_nodes':None, 'min_impurity_decrease': 0, 'oob_score': False}
gbcObj.params={'max_depth':3, 'learning_rate':0.25, 'n_estimators': 100, 'loss':'exponential',  'criterion':'friedman_mse',  'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf':0, 'subsample':1, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease':0}
logObj.params={'solver':'liblinear', 'C':1.0, 'tol': 1e-4, 'fit_intercept': True, 'max_iter':100, 'penalty': 'l2', 'intercept_scaling': 1, 'dual': False, 'multi_class': 'ovr'}
knnObj.params={'algorithm': 'kd_tree', 'n_neighbors': 6, 'weights': 'distance',  'leaf_size': 30, 'p': 2, 'metric': 'minkowski'}
mlpObj.params={ 'learning_rate_init': 0.00001, 'learning_rate': 'constant', 'batch_size': 160, 'alpha': 0.0001, 'solver': 'adam', 'hidden_layer_sizes': (50,5), 'activation': 'tanh',     'power_t': 0.5, 'max_iter': 200, 'shuffle': True, 'tol': 1e-4, 'momentum': 0.9, 'nesterovs_momentum': True, 'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8}
gnbObj.params={}
ensObj.params={'threshold': 0.5}
astaltObj.params={}

apriObj.features = [0,5,10]; # Sex, AST, and Platelets, and Hepatitis 
astaltObj.features = [4,5];
fib4Obj.features = [1,4,5,10] # Age, ALT, AST, Platelets
annObj.features = [0]

def auto_feature_select(ds, obj):
    obj.features = [] # Reset the features 
    best_score = 0
    for i in range(0, np.size(ds.X,1)):
        print('Current features: ')
        get_features(obj.features)
        print('Now trying feature: ')
        get_features([i])
        obj.features.append(i)
        cross_validation(ds)
        print_results([obj])
        if (obj.best_score > best_score):
            improvement = obj.best_score - best_score
            print('Kept feature; score improved by %0.2f%%: ' % improvement)
            best_score = obj.best_score
        else:
            print('Did not keep feature')
            obj.features.pop()

def cross_validation(ds): 
    alg.reset_algorithm([svmObj, rfcObj, gbcObj, logObj, knnObj, mlpObj, gnbObj, ensObj, apriObj, astaltObj, fib4Obj, annObj])
    reset_dataset(ds)
    c = 0;
    
    if (ds.split == 'groupKFold'):
        splitMethod = kf.split(ds.X, ds.Y, ds.MRNs.astype(int))
    elif (ds.split == 'KFold'):
        splitMethod = normalKF.split(ds.X, ds.Y)
    
    for train_index, test_index in splitMethod:
        print(c)
        preprocessing(ds, train_index, test_index, c)
        
        alg.svm_class(ds.X_tr_imp_scl[c][:,svmObj.features], ds.X_ts_imp_scl[c][:,svmObj.features], ds.Y_tr[c], ds.Y_ts[c], svmObj, c)
        alg.rfc_class(ds.X_tr_imp_scl[c][:,rfcObj.features], ds.X_ts_imp_scl[c][:,rfcObj.features], ds.Y_tr[c], ds.Y_ts[c], rfcObj, c)
        alg.gbc_class(ds.X_tr_imp_scl[c][:,gbcObj.features], ds.X_ts_imp_scl[c][:,gbcObj.features], ds.Y_tr[c], ds.Y_ts[c], gbcObj, c)        
        alg.log_class(ds.X_tr_imp_scl[c][:,logObj.features], ds.X_ts_imp_scl[c][:,logObj.features], ds.Y_tr[c], ds.Y_ts[c], logObj, c)        
        alg.knn_class(ds.X_tr_imp_scl[c][:,knnObj.features], ds.X_ts_imp_scl[c][:,knnObj.features], ds.Y_tr[c], ds.Y_ts[c], knnObj, c)        
        alg.mlp_class(ds.X_tr_imp_scl[c][:,mlpObj.features], ds.X_ts_imp_scl[c][:,mlpObj.features], ds.Y_tr[c], ds.Y_ts[c], mlpObj, c)   
        alg.gnb_class(ds.X_tr_imp_scl[c][:,gnbObj.features], ds.X_ts_imp_scl[c][:,gnbObj.features], ds.Y_tr[c], ds.Y_ts[c], gnbObj, c)        
        ann_class(ds.X_tr_imp_scl[c][:,annObj.features], ds.X_ts_imp_scl[c][:,annObj.features], ds.Y_tr[c]/4, ds.Y_ts[c], annObj, c)        
        alg.ens_class(svmObj,rfcObj,gbcObj,logObj,knnObj,mlpObj, gnbObj, ds.Y_ts[c], ensObj, c)    
        alg.apri_class(ds.X_ts_imp[c][:,apriObj.features], ds.Y_ts[c], apriObj, c)
        alg.astalt_class(ds.X_ts_imp[c][:,astaltObj.features], ds.Y_ts[c], astaltObj, c)
        alg.fib4_class(ds.X_ts_imp[c][:,fib4Obj.features], ds.Y_ts[c], fib4Obj, c)        
        c = c + 1;
    
    # This is the point where I need to get all the AUROCs  
    print_results(algorithmArray)
    print_table(algorithmArray, len(list(itertools.chain.from_iterable(ds.ts_indxs))), True) # The second argument indicates whether uncertanties should be printed here

def ann_class(Xp_train, Xp_test, Yp_train, Yp_test, annObject, count):
    from sklearn.metrics import roc_auc_score
    if (annObject.isUsed == False):
        return
    
    annObject.name = 'ANN'
    annObject.folds = count
    
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(16, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(annObj.features)))
    classifier.add(Dense(8, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier.fit(Xp_train, Yp_train, batch_size = 10, epochs = 1000)[0, 1, 3, 4, 7, 8, 9, 10, 31, 33]
        
    probabilities = classifier.predict(Xp_test)
    print(probabilities)
    Yp_pred = (probabilities > 0.5)*4
    
    cm = my_confusion_matrix(Yp_test, Yp_pred)    
    append_results(Yp_pred, probabilities, Yp_test, roc_auc_score(Yp_test/4, probabilities), cm, count, annObject)        



svmObj.isUsed = True
auto_feature_select(toronto, svmObj)

svmObj.isUsed = False
rfcObj.isUsed = True
auto_feature_select(toronto, rfcObj)

rfcObj.isUsed = False
gbcObj.isUsed = True
auto_feature_select(toronto, gbcObj)

gbcObj.isUsed = False
logObj.isUsed = True
auto_feature_select(toronto, logObj)

logObj.isUsed = False
knnObj.isUsed = True
auto_feature_select(toronto, knnObj)

knnObj.isUsed = False
mlpObj.isUsed = True
auto_feature_select(toronto, mlpObj)

mlpObj.isUsed = False
gnbObj.isUsed = True
auto_feature_select(toronto, gnbObj)

#svmObj.features = [0,1,2,3,4,5,6,7,8,9,10,14,15,19]
#rfcObj.features = svmObj.features
#gbcObj.features = svmObj.features
#logObj.features = svmObj.features
#knnObj.features = svmObj.features
#mlpObj.features = svmObj.features
#gnbObj.features = svmObj.features
#annObj.features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]

svmObj.isUsed = True
rfcObj.isUsed = True
gbcObj.isUsed = True
logObj.isUsed = True
knnObj.isUsed = True
mlpObj.isUsed = True
gnbObj.isUsed = True
annObj.isUsed = False
ensObj.isUsed = True
apriObj.isUsed = True
astaltObj.isUsed = True
fib4Obj.isUsed = True

cross_validation(toronto)

#sens: 63.53
#acc: 68.96
#auc: 0.75

find_misclassifications(toronto, algorithmArray)
plot_heat_map(toronto, algorithmArray)
get_features(gbcObj.features)

#######################  External Validation from Montreal 

# Import the validation set 
montreal = dataset_class()
montreal.description = 'McGill Liver Clinic Dataset. External test set for validation'
montreal.df = pd.read_excel('C:/Users/Soren/Desktop/Thesis/Data Analysis/reformatted_mcgill_dataset.xlsx', parse_cols="A:BD")
montreal.df = montreal.df.loc[(montreal.df['Fibrosis'] >= 0)\
                              & (montreal.df['Fibrosis'] != 2) & (montreal.df['Fibrosis'] != 3)\
                              & (montreal.df['NAFL'] == 1)]
montreal.df = montreal.df.sample(frac=1).reset_index(drop=True)
montreal.X = montreal.df.iloc[:,0:49].values
montreal.Y = (montreal.df.iloc[:,49].values > 1)*4
montreal.MRNs = montreal.df.iloc[:,51]
montreal.entryDates = montreal.df.iloc[:,52]
dfm = montreal.df

def hold_out_validate(tor, mon):
    alg.reset_algorithm([svmObj, rfcObj, gbcObj, logObj, knnObj, mlpObj, gnbObj, ensObj, apriObj, astaltObj, fib4Obj])
    reset_dataset(tor)
    reset_dataset(mon)
    
    tor.X_tr_unimp.append(tor.X)
    mon.X_ts_unimp.append(mon.X)
    
    tor.Y_tr.append(tor.Y.astype(int))
    mon.Y_ts.append(mon.Y.astype(int))
    
    # Assume mean imputation for now, add the other imputer later. 
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    tor.X_tr_imp.append(imp.fit_transform(tor.X_tr_unimp[0]))
    mon.X_ts_imp.append(imp.transform(mon.X_ts_unimp[0]))
    
    # Apply feature scaling for the relevant algorithms. 
    sc_p = StandardScaler()
    tor.X_tr_imp_scl.append(sc_p.fit_transform(tor.X_tr_imp[0]))
    mon.X_ts_imp_scl.append(sc_p.transform(mon.X_ts_imp[0]))
    
    alg.svm_class(tor.X_tr_imp_scl[0][:,svmObj.features], mon.X_ts_imp_scl[0][:,svmObj.features], tor.Y_tr[0], mon.Y_ts[0], svmObj, 0)
    alg.rfc_class(tor.X_tr_imp_scl[0][:,rfcObj.features], mon.X_ts_imp_scl[0][:,rfcObj.features], tor.Y_tr[0], mon.Y_ts[0], rfcObj, 0)
    alg.gbc_class(tor.X_tr_imp_scl[0][:,gbcObj.features], mon.X_ts_imp_scl[0][:,gbcObj.features], tor.Y_tr[0], mon.Y_ts[0], gbcObj, 0)
    alg.log_class(tor.X_tr_imp_scl[0][:,logObj.features], mon.X_ts_imp_scl[0][:,logObj.features], tor.Y_tr[0], mon.Y_ts[0], logObj, 0)
    alg.knn_class(tor.X_tr_imp_scl[0][:,knnObj.features], mon.X_ts_imp_scl[0][:,knnObj.features], tor.Y_tr[0], mon.Y_ts[0], knnObj, 0)
    alg.mlp_class(tor.X_tr_imp_scl[0][:,mlpObj.features], mon.X_ts_imp_scl[0][:,mlpObj.features], tor.Y_tr[0], mon.Y_ts[0], mlpObj, 0)
    alg.gnb_class(tor.X_tr_imp_scl[0][:,gnbObj.features], mon.X_ts_imp_scl[0][:,gnbObj.features], tor.Y_tr[0], mon.Y_ts[0], gnbObj, 0)
    alg.ens_class(svmObj,rfcObj,gbcObj,logObj,knnObj,mlpObj, gnbObj, mon.Y_ts[0], ensObj, 0)    
    alg.apri_class(mon.X_ts_imp[0][:,apriObj.features], mon.Y_ts[0], apriObj, 0)
    alg.astalt_class(mon.X_ts_imp[0][:,astaltObj.features], mon.Y_ts[0], astaltObj, 0)
    alg.fib4_class(mon.X_ts_imp[0][:,fib4Obj.features], mon.Y_ts[0], fib4Obj, 0)        
    
    print_results(algorithmArray)
    print_table(algorithmArray,np.size(mon.X_ts_imp_scl[0],0), True)
        
hold_out_validate(toronto, montreal)
