import pandas as pd
import numpy as np
import pyodbc as db
import matplotlib.pyplot as plt

def breakdown(datasets, names):
    feature_string = {}
    feature_string['Features'] = ('Features').ljust(25)
    
    col_space = ('').center(5)
    
    n_count = 0
    for n in names:    
        feature_string['Features'] += (names[n_count]).center(20)
        n_count += 1
    print('')
    print(feature_string['Features'])

    ds_count = 0    
    for ds in datasets:
        ds = ds[['Sex', 'Age', 'Albumin', 'ALP', 'AST', 'ALT', 'Bilirubin', 'INR', 'Creatinine', 'Platelets', 'BMI', 'Diabetes', 'Hepatitis B', 'Hepatitis C', 'Autoimmune Hepatitis', 'Cholestasis', 'DrugInducedLiverInjury', 'Alcohol', 'NAFL', 'MixedEnzymeAbnormality', 'BiliaryCirrhosis', 'SclerosingCholangitis', 'Neoplasm', 'Other']]
        columns = ds.columns.tolist()
        
        ds_count += 1
        
        for col in columns:
            if(col == 'reckey_enc' or col=='bx_date' or col=='patientID' or col=='missingness'):
                continue
            if ((col in feature_string) == False):
                feature_string[col] = ''
                
            if (ds[col].nunique() > 2 or col=='BMI'):
                feature_string[col] = feature_string[col] + ('%0.1f' % np.nanmean(ds[col])).rjust(5) + ' +/- ' + ('%0.1f' % np.nanstd(ds[col])).rjust(5) + col_space #('%5.1f +/- %5.1f\t' % (np.nanmean(ds[col]), np.nanstd(ds[col])))
            else:
                zeros_num = np.size(ds.loc[ds[col] == 0],0)
                ones_num = np.size(ds.loc[ds[col] == 1],0)
                
                zeros_string = str(zeros_num)
                ones_string = str(ones_num)
                
                if (col == 'Sex'):
                    feature_string[col] = feature_string[col] + ('%3s F' % ones_string).rjust(5) + '  /  ' + ('%s M' % zeros_string).rjust(5) + col_space
                    #feature_string[col] = feature_string[col] + ( '%3s F / %3s M  \t' % (ones_string, zeros_string))
                else:
                    feature_string[col] = feature_string[col] + ('%3s Y' % ones_string).rjust(5) + '  /  ' + ('%s N' % zeros_string).rjust(5) + col_space
                    #feature_string[col] = feature_string[col] + (' %3s Y / %3s N  \t' % (ones_string, zeros_string))
     
    for key, value in feature_string.items():
        if (key == 'Features'):
            continue
        print(key.ljust(25) + ' ' + value)
    return feature_string
        
conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Soren\Desktop\Thesis\Fibrosis.accdb;')
cnxn = db.connect(conn_str)
cursor = cnxn.cursor()

#etiologies = pd.read_sql("SELECT * FROM [Etiology of Liver Disease]", cnxn)
#print(etiologies['Etiology'].unique())

datasets = []
names = []

ds1 = pd.read_sql("SELECT * FROM _McGillData WHERE NAFL=1", cnxn)
ds1 = ds1.loc[ds1['missingness'] <= 3]
ds1 = ds1.loc[(ds1['Fibrosis'] == 0) | (ds1['Fibrosis'] == 1) | (ds1['Fibrosis'] == 4)]
ds1['Fibrosis'] = np.where(ds1['Fibrosis'] == 4, 4, 0)
ds1 = ds1.reset_index(drop=True)

#ds2 = pd.read_sql("SELECT * FROM _TorontoHoldOut90", cnxn)
#ds2 = ds2.loc[ds2['missingness'] <= 3]
#ds2['Fibrosis'] = np.where(ds2['Fibrosis'] == 4, 4, 0)
#ds2 = ds2.reset_index(drop=True)
#ds2_F01 = ds2.loc[ds2['Fibrosis'] == 0]
#ds2_F4 = ds2.loc[ds2['Fibrosis'] == 4]

ds3 = pd.read_sql("SELECT * FROM _McGillData WHERE NAFL <> 1", cnxn) #
ds3 = ds3.loc[ds3['missingness'] <= 3]
ds3 = ds3.loc[(ds3['Fibrosis'] == 0) | (ds3['Fibrosis'] == 1) | (ds3['Fibrosis'] == 4)]
ds3['Fibrosis'] = np.where(ds3['Fibrosis'] == 4, 4, 0)
ds3 = ds3.reset_index(drop=True)

#ds1 = ds1.loc[ds1['Age'] < 60]
#ds1 = ds1.loc[ds1['Albumin'] >= 30]
#ds3 = ds3.loc[ds3['Age'] < 60]
###ds3 = ds3.loc[ds3['Albumin'] >= 30]
ds1 = ds1.loc[ds1['INR'] < 2]
ds3 = ds3.loc[ds3['INR'] < 2]

ds1_F01 = ds1.loc[ds1['Fibrosis'] == 0] # NAFL F01 
ds1_F4 = ds1.loc[ds1['Fibrosis'] == 4]  # NAFL F4 

ds3_F01 = ds3.loc[ds3['Fibrosis'] == 0] # non-NAFL F01
ds3_F4 = ds3.loc[ds3['Fibrosis'] == 4]  # non-NAFL F4 

datasets.append(ds1_F01)
names.append('NAFL F01')
datasets.append(ds3_F01)
names.append('non-NAFL F01')

datasets.append(ds1_F4)
names.append('NAFL F4')
datasets.append(ds3_F4)
names.append('non-NAFL F4')

#datasets.append(ds2_F01)
#names.append('Tor90 F01')
#datasets.append(ds2_F4)
#names.append('Tor90 F4')

feature_strings = breakdown(datasets, names)

col = 'BMI'

plt.rcParams['figure.figsize'] = (10,3)
plt.subplot(1,2,1)
#plt.hist([ds1_F01[col].dropna(), ds3_F01[col].dropna()], 30, alpha=0.75, label=['Tor30 F01', 'McGill F01'], edgecolor='black')
xmin = min(min(ds1_F01[col].dropna()), min(ds3_F01[col].dropna()))
xmax = max(max(ds1_F01[col].dropna()), max(ds3_F01[col].dropna()))
plt.hist(ds1_F01[col].dropna(), 30, alpha=0.75, label=['NAFL F01'], edgecolor='black', color='red', range=(xmin,xmax))
plt.hist(ds3_F01[col].dropna(), 30, alpha=0.75, label=['nonNAFL F01'], edgecolor='black', color='green', range=(xmin,xmax))
plt.legend(loc='upper right')
#plt.ylim(0,50)
plt.xlabel(col + ' values')
plt.ylabel('Frequency')
plt.title(col + ' F01 Histogram')

plt.subplot(1, 2, 2)
#plt.hist([ds1_F4[col].dropna(), ds3_F4[col].dropna()], 30, alpha=0.75, label=['Tor30 F4', 'McGill F4'],edgecolor='black')
xmin = min(min(ds1_F4[col].dropna()), min(ds3_F4[col].dropna()))
xmax = max(max(ds1_F4[col].dropna()), max(ds3_F4[col].dropna()))
plt.hist(ds1_F4[col].dropna(), 30, alpha=0.75, label=['NAFL F4'], edgecolor='black', color='red', range=(xmin,xmax))
plt.hist(ds3_F4[col].dropna(), 30, alpha=0.75, label=['nonNAFL F4'], edgecolor='black', color='green', range=(xmin,xmax))
plt.legend(loc='upper right')
#plt.ylim(0,50)
plt.xlabel(col + ' values')
plt.ylabel('Frequency')
plt.title(col + ' F4 Histogram')
plt.show()

print('')
print(feature_strings['Features'])
print(col.ljust(25) + feature_strings[col])

# Take them head on! Sabet Enterprises does not retreat! They want to test my will agaionst theirs?
# Let them come. Who are they, and what great tests have they faced, that I should fear him? 
# Who are they, and what great battles have they won, that I should fear him? 


# Okay. I need plots of sensitivity, specificity, and AUROC for Toronto Training, Toronto Test, and McGill Test 
