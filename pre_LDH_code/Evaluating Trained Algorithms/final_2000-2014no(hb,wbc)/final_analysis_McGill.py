import numpy as np
import pandas as pd 
import pyodbc as db

from sklearn.externals import joblib
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

# To do list: 
# 1. Assemble McGill dataset in correct format 

# Step 1. Import the data 
data = pd.read_excel(r'C:\Users\Soren\Desktop\final_analysis_results\data\mcgill_data.xlsx')
# Mean impute missing ages based on McGill dataset 

### Step 2. Add indicator variables for missing features that will be imputed (imputed value will be put in position of imputed value)
variables = ['Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'Creatinine', 'INR', 'BMI', 'Platelets', 'Hb', 'WBC']
for col in variables: 
    data['missing_' + col] = np.where(data[col].isnull(), 1, 0)

## Step 3. Import imputers and scalers 
imputer = joblib.load('imputer.joblib')
scaler = joblib.load('scaler.joblib')

## Step 3a Impute missing ages with mean of McGill dataset 
data['Age'].fillna(data['Age'].mean(), inplace=True)

### Step 4. Impute missing values and keep this table (for appending predictions later on)
all_features = ['Sex', 'Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'Creatinine', 'INR', 'Platelets', 'BMI', 'Diabetes', 'Hb', 'WBC', 'Fibrosis', 'patientID', 'bx_date', 'reckey_enc']
ml_features = ['Sex', 'Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'Creatinine', 'INR', 'Platelets', 'BMI', 'Diabetes', 'Hb', 'WBC']

X_test_imp = pd.DataFrame(data=imputer.transform(data[ml_features]), columns=ml_features)
X_test_imp_scl = scaler.transform(X_test_imp[ml_features])
Y_test = data['Fibrosis']

## Step 5. Replace features in dataset with imputed features (to allow for APRI and FIB-4 calculations later)
data = data.drop(labels=ml_features, axis='columns')
data = data.merge(X_test_imp, left_index=True, right_index=True, how='left')

## Step 5. Load the ML algorithms (make sure the working directory is set to the folder that the algorithms are stored)
SVM_model = joblib.load('SVM.joblib')
RFC_model = joblib.load('RFC.joblib')
GBC_model = joblib.load('GBC.joblib')
LOG_model = joblib.load('LOG.joblib')
KNN_model = joblib.load('KNN.joblib')
MLP_model = joblib.load('MLP.joblib')
#
## Step 6. Make predictions using the ML Models 
## Note, do not change the features used or the order, since these were what the algorithms were trained on
## Numbers inside the brackets (e.g. [0, 1, 2, 3, ...]) correspond to features in the ml_features list
ml_models_in_ens = ['SVM_prob', 'RFC_prob', 'GBC_prob', 'LOG_prob', 'MLP_prob']
data['SVM_prob'] = SVM_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]])[:,1]
data['RFC_prob'] = RFC_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])[:,1]
data['GBC_prob'] = GBC_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]])[:,1]
data['LOG_prob'] = LOG_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]])[:,1]
data['KNN_prob'] = KNN_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])[:,1]
data['MLP_prob'] = MLP_model.predict_proba(X_test_imp_scl[:,[0, 1, 2, 3, 4, 5, 8, 9, 10, 11]])[:,1]
data['ENS1_prob'] = data[ml_models_in_ens].mean(axis=1)
data['APRI'] = (100/35)*data['AST']/data['Platelets']
data['FIB4'] = data['Age']*data['AST']/(data['Platelets']*(data['ALT']**0.5))
data['dataset'] = 'McGill'


data = data[['dataset', 'reckey_enc', 'bx_date', 'Fibrosis', 'SVM_prob', 'RFC_prob',
       'GBC_prob', 'LOG_prob', 'KNN_prob', 'MLP_prob', 'ENS1_prob', 'APRI',
       'FIB4', 'Sex', 'Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'Creatinine', 'INR',
       'Platelets', 'BMI', 'Diabetes', 'Hb', 'WBC', 'missing_Age', 'missing_Albumin', 'missing_ALP', 'missing_ALT', 'missing_AST',
       'missing_Bilirubin', 'missing_Creatinine', 'missing_INR', 'missing_BMI',
       'missing_Platelets', 'missing_Hb', 'missing_WBC', 'Etiology', 'Date Received', 'DOB']]

data.to_csv('C:\Users\Soren\Desktop\final_analysis_results\McGill_results.csv')


