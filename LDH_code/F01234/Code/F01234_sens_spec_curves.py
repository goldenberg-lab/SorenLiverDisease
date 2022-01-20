# Generate a sensitivity/specificity curve of each algorithm for our new model 
# Step 1. (DONE) Generate predictions of each algorithm on the new datasets and save to disk 
# Step 2. (DONE) Create new file for making performance curves 
# Step 3. Load model performance for dataset of interest 
# Step 4. Calculate sensitivity and specificity of model at each threshold 
# Step 5. Calculate competitor sens/spec and plot them at each algorithms respective thresholds 

import os
import sys
import copy 
import pickle 
import numpy as np
import pandas as pd 
import pyodbc as db 
import matplotlib.pyplot as plt 

from sklearn.metrics import auc 
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
        self.pm['auroc'] = self.my_auroc
        self.pm['auprc'] = self.my_auprc
        self.pm['per_det'] = self.percent_det
        self.pm['name'] = name + '(' + str(threshold) + ' , ' + str(threshold) + ')'
        
        self.color = None

        
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
        
        self.pm['auroc'] = self.my_auroc
        self.pm['auprc'] = self.my_auprc
        self.pm['per_det'] = self.percent_det
        self.pm['name'] = name + '(' + str(lower) + ' , ' + str(upper) + ')'
        
        self.color = None

        
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
        
        self.pm['auroc'] = self.my_auroc
        self.pm['auprc'] = self.my_auprc
        self.pm['per_det'] = self.percent_det
        self.pm['name'] = name + '(' + str(lower) + ' , ' + str(upper) + ')'

        self.color = None
        
class ENS_NO_AFD():
    def __init__(self, name, thresh, df):
        
        df['all_probs'] = np.where(df[name + '_probs'] >= thresh, 1, 0)
        
        self.all_probs_df = df        
        df['preds'] = np.where(df['all_probs'] == 1, 4, 0)
        
        self.df = df
        self.name = name 
        self.preds = df['preds']
        self.probs = df[name + '_probs']
        self.label = df['Fibrosis']
        self.bin_label = (self.label/4).astype(int)
        self.cm = get_confusion_matrix(self.label, self.preds)
        self.my_auroc, self.my_auprc, self.df_auroc, self.df_auprc,  = my_auc_prob(self.label, self.probs)
        self.pm = get_performance_metrics(self.cm)
        self.uthresh = thresh
        self.lthresh = thresh
        self.percent_det = 1
        
        self.pm['auroc'] = self.my_auroc
        self.pm['auprc'] = self.my_auprc
        self.pm['per_det'] = self.percent_det
        self.pm['name'] = name + '(' + str(thresh) + ' , ' + str(thresh) + ')'

        self.color = None



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

def plot_rocs(algs):
    plt.figure(figsize=(6,4))

    plt.plot([0,1], [0,1], '--', color='k', linewidth=1)
    AUROC_info = []
    for alg in algs: 
        lab = ('%0.2f - %s' % (alg.my_auroc, alg.name))
        plt.plot(alg.df_auroc['fpr'], alg.df_auroc['tpr'], '-',  color=alg.color)
        AUROC_info.append('%0.1f' % (100*alg.my_auroc))
    
    cell_text = [AUROC_info]
    rows = ['100*AUROC']
    colors = [al.color for al in algs]
    columns = []
    for al in algs: 
        text = al.name.partition('(')
        columns.append(text[0] + '\n' + '(' + text[-1])
    
    plt.grid(True)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('Recall (Sensitivity)')
    #plt.title('ROC Curve for Algorithms')
    
    ts_x =0
    te_x = 1-ts_x
    
    ts_y = -0.375
    te_y = 0.2
    
    the_table = plt.table(cellText=cell_text,
                  rowLabels=rows,
                  colLabels=columns,
                  colColours=colors,
                  cellLoc='center',
                  bbox = [ts_x,ts_y,te_x,te_y], # (left-x, bottom-y, length-x, length-y)
                  #bbox = [0.4,-0.4,0.4,0.3],
                  loc='top')    
    the_table.auto_set_font_size(False)
    if (len(algs) == 4):
        the_table.set_fontsize(10)
    else:
        the_table.set_fontsize(9)

    return None
    

def plot_prcs(algs, nskill):
    plt.figure(figsize=(6,4))

    plt.plot([0,1], [nskill,nskill], '--', color='k', linewidth=1)
    AUPRC_info = []

    for alg in algs: 
        lab = ('%0.2f - %s' % (alg.my_auprc, alg.name))
        plt.plot(alg.df_auprc['tpr'], alg.df_auprc['prc'], '-', label=lab, color=alg.color)
        AUPRC_info.append('%0.1f' % (100*alg.my_auprc))
    
    cell_text = [AUPRC_info]
    rows = ['100*AUPRC']
    colors = [al.color for al in algs]
    columns = []
    for al in algs: 
        text = al.name.partition('(')
        columns.append(text[0] + '\n' + '(' + text[-1])
        
    plt.grid(True)
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    #plt.title('PR Curve for Algorithms')

    ts_x =0
    te_x = 1-ts_x
    
    ts_y = -0.375
    te_y = 0.2
    
    the_table = plt.table(cellText=cell_text,
                  rowLabels=rows,
                  colLabels=columns,
                  colColours=colors,
                  cellLoc='center',
                  bbox = [ts_x,ts_y,te_x,te_y], # (left-x, bottom-y, length-x, length-y)
                  #bbox = [0.4,-0.4,0.4,0.3],
                  loc='top')    
    the_table.auto_set_font_size(False)
    if (len(algs) == 4):
        the_table.set_fontsize(10)
    else:
        the_table.set_fontsize(9)

    return None


def get_threshold(df, metric, val): 
       
    # Get the closest thresholds above and below 
    A_df_lo = df[['thresh', metric]].loc[df[metric] < val].sort_values(by=[metric], ascending=False)
    A_df_hi = df[['thresh', metric]].loc[df[metric] > val].sort_values(by=[metric], ascending=True)
    
    t1 = A_df_lo['thresh'].iloc[0]
    t2 = A_df_hi['thresh'].iloc[0]
    s1 = A_df_lo[metric].iloc[0]
    s2 = A_df_hi[metric].iloc[0]
    
    m = (s2 - s1)/(t2 - t1)
    tVal =(val - s1)/m + t1

    return tVal, val  


datakey = 'TE'
thesis_path = '/Volumes/Chamber of Secrets/Thesis'
datapath = thesis_path + '/Code/Lancet Digital Health Code/Predictions/F01234_models/' + datakey +'.csv'
dataset = pd.read_csv(datapath, index_col=0)
dataset['Fibrosis'] = dataset['orig_fibrosis']
dataset['Fibrosis'] = np.where(dataset['Fibrosis'] >= 3, 4, 0)

# Step 1. Load preprocessed curve to plot 
ss_df = pd.read_csv(thesis_path + '/Code/Lancet Digital Health Code/Predictions/F01234_models/' + datakey +'_sens_spec_curve.csv')

# Step 3. Plot the Sens/Spec Curves with relevant competitors 
plt.figure()
plt.plot(ss_df['thresh'], ss_df['sens'], '--', color='red', label='ENS sensitivity')
plt.plot(ss_df['thresh'], ss_df['spec'], color='red', label='ENS specificity')


# Plot APRI specificity star at point at which it intersects the specificity line 
# Plot APRI sensitivity star at point at which it intersects the sensitivity line 

# Step 2. Get point estimates for algorithms of interest 
if (datakey == 'Toronto' or datakey == 'McGill'):
    APRI = BIO('APRI', 1, 2, dataset[['APRI_vals', 'Fibrosis']])
    FIB4 = BIO('FIB4', 1.45, 3.25, dataset[['FIB4_vals', 'Fibrosis']])
    
    t_A_sens, A_sens = get_threshold(ss_df, 'sens', APRI.pm['sens'])
    t_A_spec, A_spec = get_threshold(ss_df, 'spec', APRI.pm['spec'])
    aSensLabel = 'APRI Sensitivity (%0.1f%%)' % (100*A_sens)
    aSpecLabel = 'APRI Specificity (%0.1f%%)' % (100*A_spec)
    
    t_F_sens, F_sens = get_threshold(ss_df, 'sens', FIB4.pm['sens'])
    t_F_spec, F_spec = get_threshold(ss_df, 'spec', FIB4.pm['spec'])
    fSensLabel = 'FIB4 Sensitivity (%0.1f%%)' % (100*F_sens)
    fSpecLabel = 'FIB4 Specificity (%0.1f%%)' % (100*F_spec)
    
    plt.scatter(t_A_sens, A_sens, marker='*', s=150, color='cyan', label=aSensLabel, edgecolors='black')
    plt.scatter(t_A_spec, A_spec, marker='s', s=150, color='cyan', label=aSpecLabel, edgecolors='black')

    # Plot horizontal lines to indicate where sensitivity and specificity are fixed 
    plt.plot([t_A_sens, t_A_spec], [A_sens, A_sens], '--', color='cyan', markersize=10)    
    plt.plot([t_A_sens, t_A_spec], [A_spec, A_spec], '--', color='cyan', markersize=10)    

    plt.scatter(t_F_sens, F_sens, marker='*', s=150, color='lightblue', label=fSensLabel, edgecolors='black')
    plt.scatter(t_F_spec, F_spec, marker='s', s=150, color='lightblue', label=fSpecLabel, edgecolors='black')
    
    # Plot horizontal lines to indicate where sensitivity and specificity are fixed 
    plt.plot([t_F_sens, t_F_spec], [F_sens, F_sens], '--', color='lightblue', markersize=10)    
    plt.plot([t_F_sens, t_F_spec], [F_spec, F_spec], '--', color='lightblue', markersize=10)    
    
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5,-0.35))

elif (datakey == 'Expert'):
    #expert_dataset = temp2
    EXP = MLA('EXP', 0.5, dataset['EXP_probs'], dataset['Fibrosis'])

    t_E_sens, E_sens = get_threshold(ss_df, 'sens', EXP.pm['sens'])
    t_E_spec, E_spec = get_threshold(ss_df, 'spec', EXP.pm['spec'])
    
    eSensLabel = 'EXP Sensitivity (%0.1f%%)' % (100*E_sens)
    eSpecLabel = 'EXP Specificity (%0.1f%%)' % (100*E_spec)
    
    plt.scatter(t_E_sens, E_sens, marker='*', s=150, color='lightgreen', label=eSensLabel, edgecolors='black')
    plt.scatter(t_E_spec, E_spec, marker='s', s=150, color='lightgreen', label=eSpecLabel, edgecolors='black')
    
    plt.plot([t_E_sens, t_E_spec], [E_sens, E_sens], '--', color='lightgreen', markersize=10)    
    plt.plot([t_E_sens, t_E_spec], [E_spec, E_spec], '--', color='lightgreen', markersize=10)    
    
    plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.35))
elif (datakey == 'NAFL'):
    NFLD = BIO('NAFL', -1.455, 0.675, dataset[['NAFL_vals', 'Fibrosis']])    

    t_N_sens, N_sens = get_threshold(ss_df, 'sens', NFLD.pm['sens'])
    t_N_spec, N_spec = get_threshold(ss_df, 'spec', NFLD.pm['spec'])
    
    nSensLabel = 'NFS Sensitivity (%0.1f%%)' % (100*N_sens)
    nSpecLabel = 'NFS Specificity (%0.1f%%)' % (100*N_spec)
    
    plt.scatter(t_N_sens, N_sens, marker='*', s=150, color='yellow', label=nSensLabel, edgecolors='black')
    plt.scatter(t_N_spec, N_spec, marker='s', s=150, color='yellow', label=nSpecLabel, edgecolors='black')
    
    plt.plot([t_N_sens, t_N_spec], [N_sens, N_sens], '--', color='yellow', markersize=10)    
    plt.plot([t_N_sens, t_N_spec], [N_spec, N_spec], '--', color='yellow', markersize=10)    
    
    plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.35))
elif (datakey == 'TE'):
    TE_1 = BIO('TE', 8, 8, dataset[['TE_vals', 'Fibrosis']])

    t_T_sens, T_sens = get_threshold(ss_df, 'sens', TE_1.pm['sens'])
    t_T_spec, T_spec = get_threshold(ss_df, 'spec', TE_1.pm['spec'])
    
    tSensLabel = 'TE Sensitivity (%0.1f%%)' % (100*T_sens)
    tSpecLabel = 'TE Specificity (%0.1f%%)' % (100*T_spec)
    
    plt.scatter(t_T_sens, T_sens, marker='*', s=150, color='plum', label=tSensLabel, edgecolors='black')
    plt.scatter(t_T_spec, T_spec, marker='s', s=150, color='plum', label=tSpecLabel, edgecolors='black')
    
    plt.plot([t_T_sens, t_T_spec], [T_sens, T_sens], '--', color='plum', markersize=10)    
    plt.plot([t_T_sens, t_T_spec], [T_spec, T_spec], '--', color='plum', markersize=10)    
    
    plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.35))

    
plt.grid(True)
plt.xlabel('Threshold')
plt.ylabel('Performance (%)')
plt.xlim([0,1.01])
plt.ylim([0,1.01])
plt.title(datakey + ' Test Set Sensitivity-Specificity Tradeoff')
    
    
