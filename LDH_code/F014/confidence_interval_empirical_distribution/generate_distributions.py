import os
import copy 
import numpy as np
import pandas as pd 
import pyodbc as db 
import matplotlib.pyplot as plt 

from sklearn.metrics import auc 
from sklearn.externals import joblib
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import precision_recall_curve 

from beautifultable import BeautifulTable

pd.options.mode.chained_assignment = None  # default='warn'


class MLA():
    def __init__(self, name, threshold, probs, label): 
        self.name = name 
        self.preds = (probs >= threshold)*4
        self.probs = probs
        self.label = label
        self.bin_label = (label/4).astype(int)
        self.cm = get_confusion_matrix(self.label, self.preds)
        self.my_auroc, self.my_auprc, self.df_auroc, self.df_auprc,  = my_auc_prob(label, probs)
        self.pm = get_performance_metrics(self.cm)
        self.percent_det = 1
        self.uthresh = threshold
        self.lthresh = threshold
        
class BIO(): 
    def __init__(self, name, lower, upper, df):

        print('Initializing: ' + name)
        self.probs = np.where(df[name + '_vals'] >= upper, 1, 0.5)
        self.probs = np.where(df[name + '_vals'] <= lower, 0, self.probs)
        
        df['indet'] = np.where(df[name + '_vals'].between(lower, upper, inclusive=False), 1, 0)
        self.percent_det = (len(df) - df['indet'].sum())/len(df)
        df = df.loc[df['indet'] == 0][[name + '_vals', 'Fibrosis']].reset_index(drop=True)
        df['preds'] = np.where(df[name + '_vals'] >= upper, 4, 0) # This is the line that causes problems for TE

        self.df = df
        self.name = name
        self.preds = df['preds']
        self.label = df['Fibrosis']
        self.values = df[name + '_vals']
        self.bin_label = (self.label/4).astype(int)
        self.cm = get_confusion_matrix(self.label, self.preds)
        self.my_auroc, self.my_auprc, self.df_auroc, self.df_auprc = my_auc_non_prob(self.label, self.values)  
        self.pm = get_performance_metrics(self.cm)
        self.uthresh = upper
        self.lthresh = lower
        
class ENS(): 
    def __init__(self, name, lower, upper, df, apri_prob, fib4_prob):
        
        df['af'] = apri_prob + fib4_prob
        df['indet'] = np.where(df[name + '_probs'].between(lower, upper), 1, 0)
        df['indet'] = np.where(((df['af'] == 0) | (df['af'] == 2)), 0, df['indet'])
        
        
        df['all_probs'] = np.where((df[name + '_probs'] >= upper) | ((df['af'] == 2) & (df[name + '_probs'].between(lower, upper))), 1, np.nan)
        df['all_probs'] = np.where((df[name + '_probs'] <= lower) | ((df['af'] == 0) & (df[name + '_probs'].between(lower, upper))), 0, df['all_probs'])
        df['all_probs'] = np.where(df['indet'] == 1, 0.5, df['all_probs'])

        
        self.all_probs_df = df
        
        self.percent_det = (len(df) - df['indet'].sum())/len(df)
        df = df.loc[df['indet'] == 0][[name + '_probs', 'Fibrosis']].reset_index(drop=True)
        df['preds'] = np.where(df[name + '_probs'] >= upper, 4, 0)
        
        self.df = df
        self.name = name 
        self.preds = df['preds']
        self.probs = df[name + '_probs']
        self.label = df['Fibrosis']
        self.bin_label = (self.label/4).astype(int)
        self.cm = get_confusion_matrix(self.label, self.preds)
        self.my_auroc, self.my_auprc, self.df_auroc, self.df_auprc,  = my_auc_prob(self.label, self.probs)
        self.pm = get_performance_metrics(self.cm)
        self.uthresh = upper
        self.lthresh = lower

def get_confusion_matrix(label, preds):
    cm = pd.DataFrame({'pred': preds, 'label': label})
    cm['TN'] = np.where((cm['pred'] == 0) & (cm['label'] == 0), 1, 0)
    cm['FP'] = np.where((cm['pred'] == 4) & (cm['label'] == 0), 1, 0)
    cm['FN'] = np.where((cm['pred'] == 0) & (cm['label'] == 4), 1, 0)
    cm['TP'] = np.where((cm['pred'] == 4) & (cm['label'] == 4), 1, 0)
    cm['null'] = np.where(cm['pred'].isnull(), 1, 0)
    return cm.sum()
    
def get_performance_metrics(cm): 
    TP = cm['TP']
    TN = cm['TN']
    FP = cm['FP']
    FN = cm['FN']
    
    sens = TP/(TP + FN)
    spec = TN/(TN + FP)
    ppv = TP/(TP + FP)
    npv = TN/(TN + FN)
    acc = (TP + TN)/(TP + TN + FP + FN)
    
    pm = {'sens': sens, 'spec': spec, 'ppv': ppv, 
          'npv': npv, 'acc': acc}
    
    return pm 

def my_auc_prob(label, probs): 
    results = []
    
    new_probs = np.unique(probs.to_numpy(copy=True))
    mid_probs = new_probs[:-1] + np.diff(new_probs)/2
    mid_probs = np.unique(np.append(mid_probs, [0,1]))
    
    for t in mid_probs: 
        cm = get_confusion_matrix(label, (probs >= t)*4)
        results.append([t, cm['TP'], cm['FP'], cm['FN'], cm['TN']])
        
    df = pd.DataFrame.from_records(results, columns=['thresh', 'TP', 'FP', 'FN', 'TN'])
    df['fpr'] = df['FP']/(df['FP'] + df['TN'])
    df['tpr'] = df['TP']/(df['TP'] + df['FN'])
    df['prc'] = df['TP']/(df['TP'] + df['FP'])
    
    auroc_df = df[['thresh', 'fpr', 'tpr']]
    auroc_tempf = auroc_df.drop_duplicates(subset=['tpr'], keep='first')
    auroc_templ = auroc_df.drop_duplicates(subset=['tpr'], keep='last')
    auroc_df_2 = pd.concat([auroc_tempf, auroc_templ])
    auroc_df_2.sort_values(by=['thresh'], ascending = [True], inplace=True)  
    auroc_tempf2 = auroc_df_2.drop_duplicates(subset=['fpr'], keep='first')
    auroc_templ2 = auroc_df_2.drop_duplicates(subset=['fpr'], keep='last')
    auroc_df_2 = pd.concat([auroc_tempf2, auroc_templ2])
    auroc_df_2.sort_values(by=['thresh'], ascending = [True], inplace=True)
    auroc_df_2.reset_index(drop=True, inplace=True)
    
    auprc_df = df[['thresh', 'tpr', 'prc']]
    auprc_tempf = auprc_df.drop_duplicates(subset=['tpr'], keep='first')
    auprc_templ = auprc_df.drop_duplicates(subset=['tpr'], keep='last')
    auprc_df_2 = pd.concat([auprc_tempf, auprc_templ])
    auprc_df_2 = auprc_df_2.loc[~auprc_df_2['prc'].isnull()]
    auprc_df_2.sort_values(by=['thresh', 'tpr', 'prc'], inplace=True)
    auprc_df_2.reset_index(drop=True, inplace=True)
    l2 = len(auprc_df_2)
    auprc_df_2.loc[l2] = [1, 0, auprc_df_2.iloc[l2-1]['prc']]
    
    auroc = auc(auroc_df_2['fpr'], auroc_df_2['tpr'])
    auprc = auc(auprc_df_2['tpr'], auprc_df_2['prc'])
    
    return auroc, auprc, auroc_df_2, auprc_df_2

def my_auc_non_prob(label, values): 
    
    results = []
    new_vals = np.unique(values.to_numpy(copy=True))
    mid_vals = new_vals[:-1] + np.diff(new_vals)/2
    mid_vals = np.unique(np.append(mid_vals, [-1000,1000]))

    for t in mid_vals: 
        cm = get_confusion_matrix(label, (values >= t)*4)
        results.append([t, cm['TP'], cm['FP'], cm['FN'], cm['TN']])
    
    
    df = pd.DataFrame.from_records(results, columns=['thresh', 'TP', 'FP', 'FN', 'TN'])
    df['fpr'] = df['FP']/(df['FP'] + df['TN'])
    df['tpr'] = df['TP']/(df['TP'] + df['FN'])
    df['prc'] = df['TP']/(df['TP'] + df['FP'])
        
    auroc_df = df[['thresh', 'fpr', 'tpr']]
    auroc_tempf = auroc_df.drop_duplicates(subset=['tpr'], keep='first')
    auroc_templ = auroc_df.drop_duplicates(subset=['tpr'], keep='last')
    auroc_df_2 = pd.concat([auroc_tempf, auroc_templ])
    auroc_df_2.sort_values(by=['thresh'], ascending = [True], inplace=True)  
    auroc_tempf2 = auroc_df_2.drop_duplicates(subset=['fpr'], keep='first')
    auroc_templ2 = auroc_df_2.drop_duplicates(subset=['fpr'], keep='last')
    auroc_df_2 = pd.concat([auroc_tempf2, auroc_templ2])
    auroc_df_2.sort_values(by=['thresh'], ascending = [True], inplace=True)
    auroc_df_2.reset_index(drop=True, inplace=True)
    
    auprc_df = df[['thresh', 'tpr', 'prc']]
    auprc_tempf = auprc_df.drop_duplicates(subset=['tpr'], keep='first')
    auprc_templ = auprc_df.drop_duplicates(subset=['tpr'], keep='last')
    auprc_df_2 = pd.concat([auprc_tempf, auprc_templ])
    auprc_df_2 = auprc_df_2.loc[~auprc_df_2['prc'].isnull()]
    auprc_df_2.sort_values(by=['thresh', 'tpr', 'prc'], inplace=True)
    auprc_df_2.reset_index(drop=True, inplace=True)
    l2 = len(auprc_df_2)
    auprc_df_2.loc[l2] = [1, 0, auprc_df_2.iloc[l2-1]['prc']]
    
    auroc = auc(auroc_df_2['fpr'], auroc_df_2['tpr'])
    auprc = auc(auprc_df_2['tpr'], auprc_df_2['prc'])
    
    return auroc, auprc, auroc_df_2, auprc_df_2

def write_performance_metrics(algs, key, sql_dict):
    col_heads = []
    sens_row =  []
    spec_row = []
    ppv_row = []
    npv_row = []
    acc_row = []
    AUROC_row = []
    AUPRC_row = []
    det_row = []
    
    for alg in algs: 
        col_heads.append('%s\n%0.3f - %0.3f' % (alg.name, alg.lthresh, alg.uthresh))
        sens_row.append('%0.1f' % (100*alg.pm['sens']))
        spec_row.append('%0.1f' % (100*alg.pm['spec']))
        ppv_row.append('%0.1f' % (100*alg.pm['ppv']))
        npv_row.append('%0.1f' % (100*alg.pm['npv']))
        acc_row.append('%0.1f' % (100*alg.pm['acc']))
        AUROC_row.append('%0.1f' % (100*alg.my_auroc))
        AUPRC_row.append('%0.1f' % (100*alg.my_auprc))
        det_row.append('%0.1f' % (100*alg.percent_det))
    
    table = BeautifulTable(maxwidth=200)
    table.columns.header = col_heads
    table.rows.header = ['sens', 'spec', 'ppv', 'npv', 'acc', 'AUROC', 'AUPRC', '%det']    
    table.rows[0] = sens_row
    table.rows[1] = spec_row
    table.rows[2] = ppv_row
    table.rows[3] = npv_row
    table.rows[4] = acc_row
    table.rows[5] = AUROC_row
    table.rows[6] = AUPRC_row
    table.rows[7] = det_row
    print(table)


datakey = 'McGill'
num_trials = 1000

path = 'C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\ConfInterval_Distribution_Modelling\\Predictions\\'
path_dict = {'Toronto': path + 'Toronto.xlsx',
             'Expert': path + 'Expert.xlsx',
             'McGill': path + 'McGill.xlsx',
             'NAFL': path + 'NAFL.xlsx', 
             'TE': path + 'TE.xlsx'} # Experienced a strange error on 9/4/2020, workaround is loading data from excel

# Ensemble threshold dictionary
etd = {'Toronto': [(0.675, 0.75),(0.15,0.775)],
       'Expert': [(0.35,0.80),(0.675,0.675)],
       'McGill': [(0.80, 0.85),(0.45,0.875)],
       'NAFL': [(0.675,0.75 ),(0.225,0.775)],
       'TE': [(0.725, 0.775),(0.775,0.775)]} 


# Master dataset which will get resammpled
main_dataset = pd.read_excel(path_dict[datakey], index_col=0)
num_records = len(main_dataset)
results = []

for i in range(0,num_trials):
    
    print('Trial: %d' % (i+1))
    
    dataset = main_dataset.sample(n=num_records, replace=True, random_state=i)
    
    # Create objects which calculate the relevant performance metrics
    APRI = BIO('APRI', 1, 2, dataset[['APRI_vals', 'Fibrosis']])
    FIB4 = BIO('FIB4', 1.45, 3.25, dataset[['FIB4_vals', 'Fibrosis']])
    NFLD = BIO('NAFL', -1.455, 0.675, dataset[['NAFL_vals', 'Fibrosis']])    
    ENS3 = ENS('ENS3', 0.25, 0.45, dataset[['ENS3_probs', 'Fibrosis']], APRI.probs, FIB4.probs)
    ENS3b = ENS('ENS3', etd[datakey][0][0], etd[datakey][0][1], dataset[['ENS3_probs', 'Fibrosis']], APRI.probs, FIB4.probs)
    ENS3c = ENS('ENS3', etd[datakey][1][0], etd[datakey][1][1], dataset[['ENS3_probs', 'Fibrosis']], APRI.probs, FIB4.probs)
    algs = [APRI, FIB4, NFLD, ENS3, ENS3b, ENS3c]
    
    if (datakey == 'TE'):
        TE_1 = BIO('TE', 8, 8, dataset[['TE_vals', 'Fibrosis']])
        algs.append(TE_1)
    if (datakey == 'Expert'):
        EXP = MLA('EXP', 0.5, dataset['EXP_probs'], dataset['Fibrosis'])
        algs.append(EXP)
        
    for alg in algs: 
        results.append({
            'trial': i, 
            'algorithm': alg.name + '(' + str(alg.lthresh) + ',' + str(alg.uthresh) + ')',
            'sens': alg.pm['sens'],
            'spec': alg.pm['spec'],
            'ppv': alg.pm['ppv'], 
            'npv': alg.pm['npv'], 
            'acc': alg.pm['acc'], 
            'auroc': alg.my_auroc, 
            'auprc': alg.my_auprc, 
            'per_det': alg.percent_det})

dist = pd.DataFrame.from_records(results)
dist.sort_values(by=['algorithm'], inplace=True)
outpath = 'C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\ConfInterval_Distribution_Modelling\\Distributions\\'
dist.to_excel(outpath + datakey + '.xlsx')

