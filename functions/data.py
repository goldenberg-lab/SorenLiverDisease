import numpy as np 
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold     

class dataset_class(object): 
    def __init__(self):
        self.name = ""
        self.description = ""
        self.df = [];
        
        # Global
        self.X = []; 
        self.Y = [];
        self.cv_indices  = [];
        self.MRNs = [];
        self.entryDates = [];
        
        self.X_hold_out = [];
        self.Y_hold_out = [];
        self.hold_out_MRNs = [];
        self.hold_out_entryDates = [];
        self.hold_out_indices = [];
        
        # Cross validation set     
        self.X_tr_unimp = [];
        self.X_ts_unimp = [];
        
        self.X_tr_imp = [];
        self.X_ts_imp = [];
        
        self.X_tr_imp_scl = [];
        self.X_ts_imp_scl = [];
        
        self.Y_tr = [];
        self.Y_ts = [];
        
        self.tr_indxs = [];
        self.ts_indxs = [];
        
        self.correct_class_indxs = []
        self.mis_F4_class_indxs = []
        self.mis_F01_class_indxs = []
        
        self.heat_map = []
        
def reset_dataset(obj):
    obj.X_tr_unimp = [];
    obj.X_ts_unimp = [];
    
    obj.X_tr_imp = [];
    obj.X_ts_imp = [];
    
    obj.X_tr_imp_scl = [];
    obj.X_ts_imp_scl = [];
    
    obj.Y_tr = [];
    obj.Y_ts = [];
    
    obj.tr_indxs = [];
    obj.ts_indxs = [];
    
    obj.correct_class_indxs = []
    obj.mis_F4_class_indxs = []
    obj.mis_F01_class_indxs = []
    
    obj.heat_map = []
    
def train_test_split_KFold(obj):
    from sklearn.model_selection import GroupKFold
    kf1 = GroupKFold(n_splits = 5)
    kf1.get_n_splits(obj.X, obj.Y, obj.MRNs.astype(int))

    hold_out_count = 0
    
    for main_index, hold_out_index in kf1.split(obj.X, obj.Y, obj.MRNs.astype(int)):
        print(main_index)
        if (hold_out_count == 4):
            obj.X_hold_out = obj.X[hold_out_index, :]
            obj.Y_hold_out = obj.Y[hold_out_index]
            obj.hold_out_MRNs = obj.MRNs[hold_out_index]
            obj.hold_out_entryDates = obj.entryDates[hold_out_index]
            obj.hold_out_indices = hold_out_index;
            
            obj.X = obj.X[main_index, :]
            obj.Y = obj.Y[main_index]
            obj.MRNs = obj.MRNs[main_index]
            obj.entryDates = obj.entryDates[main_index]
            obj.cv_indices = main_index;
        hold_out_count += 1
    

def preprocessing(obj, train_index, test_index, count):
    obj.tr_indxs.append(train_index)
    obj.ts_indxs.append(test_index)
    
    obj.X_tr_unimp.append(obj.X[train_index])
    obj.X_ts_unimp.append(obj.X[test_index])
    
    obj.Y_tr.append(obj.Y[train_index].astype(int))
    obj.Y_ts.append(obj.Y[test_index].astype(int))
    
    #imputeAndCalculate(obj.X_tr_unimp[count], obj.X_ts_unimp[count], obj)

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    obj.X_tr_imp.append(imp.fit_transform(obj.X_tr_unimp[count]))
    obj.X_ts_imp.append(imp.transform(obj.X_ts_unimp[count]))
        
    # Apply feature scaling for relevant algorithms
    sc_p = StandardScaler()
    obj.X_tr_imp_scl.append(sc_p.fit_transform(obj.X_tr_imp[count]))
    obj.X_ts_imp_scl.append(sc_p.transform(obj.X_ts_imp[count]))
    
def imputeAndCalculate(train_unimp, test_unimp, obj):
    from copy import copy
    from fancyimpute import MICE
    train = copy(train_unimp)
    test = copy(test_unimp)
    # Assume mean imputation for now. These are the numerical features to be imputed.  
#    for i in range(2,11):
#        train[:,i][np.isnan(train[:,i])] = np.nanmean(train[:,i])
#        test[:,i][np.isnan(test[:,i])] = np.nanmean(train[:,i])
        
    train[:,2:12] = MICE().complete(train[:, 2:12])
    test[:,2:12] = MICE().complete(test[:,2:12])        

    # Calculated Features    
    train = updateCalculatedFeatures(train)
    test = updateCalculatedFeatures(test)
    obj.X_tr_imp.append(train)
    obj.X_ts_imp.append(test)    

def updateCalculatedFeatures(data):
    ulp = (data[:,0] == 0)*12 + 19 # Upper limit normal platelets, 31 for men, 19 for women
    data[:,11] = data[:,1]*data[:,5]/(data[:,10]*np.sqrt(data[:,4])) # FIB4
    data[:,12] = (100*data[:,5]/ulp)/data[:,10] # APRI
    data[:,13] = data[:,5]/data[:,4]; #  AST/ALT
    data[:,14] = data[:,8]*data[:,8]; # INR ^2
    data[:,15] = data[:,6]/26; # Bilirubin / 26
    data[:,16] = data[:,10]/140; # Platelets / 140
    data[:,17] = data[:,2]/35; # Albumin / 35
    data[:,18] = data[:,7]/90; # Creatinine / 90
    data[:,19] = data[:,9]/30; # BMI/30
    data[:,30] = (data[:,5] > data[:,4])*1; # AST > ALT
    data[:,31] = (data[:,8] > 1)*1 # INR > 1
    data[:,32] = (data[:,6] >= 26)*1 # Bilirubin >= 26
    data[:,33] = (data[:,2] < 35)*1 # Albumin < 35
    data[:,34] = (data[:,7] > 90)*1 # Creatinine > 90
    data[:,35] = (data[:,10] < 140)*1 # Platelets < 140
    data[:,36] = (data[:,9] > 30)*1 # BMI > 30 
    return data
    