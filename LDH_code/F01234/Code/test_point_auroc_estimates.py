import numpy as np
import pandas as pd 
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score 


dk = 'McGill'
pred_path = "C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\F01234\\Predictions\\predictions\\"
df = pd.read_csv(pred_path + dk + '.csv', index_col=0) 

df = df.loc[df['orig_fibrosis'].isin([0,4])]
df['label'] = np.where(df['orig_fibrosis'] >= 1, 1, 0)

df['APRI'] = df['APRI_vals']/max(df['APRI_vals'])
df['FIB4'] = df['FIB4_vals']/max(df['FIB4_vals'])

print('APRI', roc_auc_score(df['label'], df['APRI']))
print('FIB4', roc_auc_score(df['label'], df['FIB4']))