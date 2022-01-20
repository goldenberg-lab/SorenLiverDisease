# Calculate relevant performance metrics 
# Plot results and add tables 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from beautifultable import BeautifulTable
import matplotlib.style as style
import scipy.stats as st
style.use('seaborn-dark')
np.set_printoptions(precision=5)

class alg:
    def __init__(self):
        self.name = ""
        self.sens = []
        self.spec = [] 
        self.ppv = []
        self.npv = []
        self.acc = []
        self.AUROC = []
        self.AUPRC = []
        self.det = []
        
    def split(self, name, d, num):
        self.name = name
        self.sens = d['sens'][:num].dropna().tolist()
        self.spec = d['spec'][:num].dropna().tolist()
        self.ppv = d['ppv'][:num].dropna().tolist()
        self.npv = d['npv'][:num].dropna().tolist()
        self.acc = d['acc'][:num].dropna().tolist()
        self.AUROC = d['AUROC'][:num].dropna().tolist()
        self.AUPRC = d['AUPRC'][:num].dropna().tolist()
        self.det = d['det'][:num].dropna().tolist()

def read(path):
    data = pd.read_excel(path)
    data = data.transpose()
    return data

def plot_dist(metric_a, metric_f, metric_e, metric_text, metric_expert=None):
    plt.figure(figsize=(6,3))
    plt.hold(True)
    plt.title('Empirical ' + metric_text + ' Distribution')
    if (metric_expert == None):
        plt.hist(np.array(metric_e), bins=np.arange(0, 100,1).tolist(), label='ENS2', alpha=0.5, linewidth=1, edgecolor='black', color='red')
        plt.hist(np.array(metric_a), bins=np.arange(0,100,1).tolist(), label='APRI', alpha=0.5, linewidth=1, edgecolor='black', color='cyan')
        plt.hist(np.array(metric_f), bins=np.arange(0,100,1).tolist(), label='FIB4', alpha=0.75, linewidth=1, edgecolor='black', color='lightblue')
    else:
        plt.hist(np.array(metric_e), bins=np.arange(0, 100,1).tolist(), label='ENS2', alpha=0.5, linewidth=1, edgecolor='black', color='red')
        plt.hist(np.array(metric_a), bins=np.arange(0,100,1).tolist(), label='APRI', alpha=0.5, linewidth=1, edgecolor='black', color='cyan')
        plt.hist(np.array(metric_f), bins=np.arange(0,100,1).tolist(), label='FIB4', alpha=0.75, linewidth=1, edgecolor='black', color='lightblue')        
        plt.hist(np.array(metric_expert), bins=np.arange(0,100,1).tolist(), label='Expert', alpha=0.5, linewidth=1, edgecolor='black', color='purple')
   
    plt.legend(loc='upper left')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    APRI_mean = np.mean(metric_a)
    APRI_emp_ci_low = np.percentile(metric_a, 2.5)
    APRI_emp_ci_high = np.percentile(metric_a, 97.5)

    FIB4_mean = np.mean(metric_f)
    FIB4_emp_ci_low = np.percentile(metric_f, 2.5)
    FIB4_emp_ci_high = np.percentile(metric_f,  97.5)
    
    ENS3_mean = np.mean(metric_e)
    ENS3_emp_ci_low = np.percentile(metric_e, 2.5)
    ENS3_emp_ci_high = np.percentile(metric_e,  97.5)

    if (metric_expert != None):
        EXP_mean = np.mean(metric_expert)
        EXP_emp_ci_low = np.percentile(metric_expert, 2.5)
        EXP_emp_ci_high = np.percentile(metric_expert,  97.5)      
    
    cell_text = []
    
    if (metric_expert == None):
        columns = ('APRI', 'FIB-4', 'ENS2')
        rows = ['Mean', '2.5th percentile', '97.5th percentile']
    else:
        columns = ('APRI', 'FIB-4', 'ENS2', 'Experts')
        rows = ['Mean', '2.5th percentile', '97.5th percentile']        
#        APRI_EXP = st.wilcoxon(metric_a, metric_expert)
#        FIB4_EXP = st.wilcoxon(metric_f, metric_expert)
#        ENS3_EXP = st.wilcoxon(metric_e, metric_expert)
#        EXP_EXP = st.wilcoxon(metric_expert, metric_expert)    

    if (metric_expert == None):
        row1= [('%0.2f' % APRI_mean), ('%0.2f' % FIB4_mean), ('%0.2f' % ENS3_mean)] 
        row2= [('%0.2f' % APRI_emp_ci_low), ('%0.2f' % FIB4_emp_ci_low),  ('%0.2f' % ENS3_emp_ci_low)] 
        row3= [('%0.2f' % APRI_emp_ci_high), ('%0.2f' % FIB4_emp_ci_high), ('%0.2f' % ENS3_emp_ci_high)] 

    else:
        row1= [('%0.2f' % APRI_mean), ('%0.2f' % FIB4_mean),  ('%0.2f' % ENS3_mean),('%0.2f' % EXP_mean) ] 
        row2= [('%0.2f' % APRI_emp_ci_low), ('%0.2f' % FIB4_emp_ci_low), ('%0.2f' % ENS3_emp_ci_low),  ('%0.2f' % EXP_emp_ci_low)] 
        row3= [('%0.2f' % APRI_emp_ci_high), ('%0.2f' % FIB4_emp_ci_high), ('%0.2f' % ENS3_emp_ci_low),  ('%0.2f' % EXP_emp_ci_high)] 

    cell_text.append(row1)
    cell_text.append(row2)
    cell_text.append(row3)

    the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      cellLoc='center',
                      bbox = [0,-0.4,1,0.3],
                      loc='top')    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    plt.show()    
    return metric_a, metric_f, metric_e

#    if (metric_expert == None):
#        columns = ('APRI-mean', 'APRI-95%CI', 'FIB4-mean', 'FIB4-95%CI', 'ENS2-mean', 'ENS3-95%CI')
#        rows = ['Empirical CI', 'APRI p-val', 'FIB4 p-val', 'ENS3 p-val']
#    else:
#        columns = ('APRI-mean', 'APRI-95%CI', 'FIB4-mean', 'FIB4-95%CI', 'ENS2-mean', 'ENS3-95%CI', 'EXP-mean', 'EXP-95% CI')
#        rows = ['Empirical CI', 'APRI p-val', 'FIB4 p-val', 'ENS3 p-val', 'EXP p-val'] 
#        APRI_EXP = st.wilcoxon(metric_a, metric_expert)
#        FIB4_EXP = st.wilcoxon(metric_f, metric_expert)
#        ENS3_EXP = st.wilcoxon(metric_e, metric_expert)
#        EXP_EXP = st.wilcoxon(metric_expert, metric_expert)

#    APRI_APRI = st.wilcoxon(metric_a, metric_a)
#    APRI_FIB4 = st.wilcoxon(metric_a, metric_f)
#    APRI_ENS3 = st.wilcoxon(metric_a, metric_e)
#    FIB4_FIB4 = st.wilcoxon(metric_f, metric_f)
#    FIB4_ENS3 = st.wilcoxon(metric_f, metric_e)
#    ENS3_ENS3 = st.wilcoxon(metric_e, metric_e)
#    
#    if (metric_expert == None):
#        row1= [('%0.2f' % APRI_mean), ('%0.2f' % APRI_emp_ci_low) + '-' + ('%0.2f' % APRI_emp_ci_high), ('%0.2f' % FIB4_mean), ('%0.2f' % FIB4_emp_ci_low) + '-' + ('%0.2f' % FIB4_emp_ci_high), ('%0.2f' % ENS3_mean), ('%0.2f' % ENS3_emp_ci_low) + '-' + ('%0.2f' % ENS3_emp_ci_high)] 
#        row2= [('%0.5f' % APRI_APRI.pvalue), ' ', ('%0.5f' % APRI_FIB4.pvalue), ' ', ('%0.5f' % APRI_ENS3.pvalue), ' '] 
#        row3= [('%0.5f' % APRI_FIB4.pvalue), ' ', ('%0.5f' % FIB4_FIB4.pvalue), ' ', ('%0.5f' % FIB4_ENS3.pvalue), ' '] 
#        row4= [('%0.5f' % APRI_ENS3.pvalue), ' ', ('%0.5f' % FIB4_ENS3.pvalue), ' ', ('%0.5f' % ENS3_ENS3.pvalue), ' '] 
#    else:
#        row1= [('%0.2f' % APRI_mean), ('%0.2f' % APRI_emp_ci_low) + '-' + ('%0.2f' % APRI_emp_ci_high), ('%0.2f' % FIB4_mean), ('%0.2f' % FIB4_emp_ci_low) + '-' + ('%0.2f' % FIB4_emp_ci_high), ('%0.2f' % ENS3_mean), ('%0.2f' % ENS3_emp_ci_low) + '-' + ('%0.2f' % ENS3_emp_ci_high), ('%0.2f' % EXP_mean), ('%0.2f' % EXP_emp_ci_low) + '-' + ('%0.2f' % EXP_emp_ci_high) ] 
#        row2= [('%0.5f' % APRI_APRI.pvalue), ' ', ('%0.5f' % APRI_FIB4.pvalue), ' ', ('%0.5f' % APRI_ENS3.pvalue), ' ', ('%0.5f' % APRI_EXP.pvalue), ' '] 
#        row3= [('%0.5f' % APRI_FIB4.pvalue), ' ', ('%0.5f' % FIB4_FIB4.pvalue), ' ', ('%0.5f' % FIB4_ENS3.pvalue), ' ', ('%0.5f' % FIB4_EXP.pvalue), ' ']  
#        row4= [('%0.5f' % APRI_ENS3.pvalue), ' ', ('%0.5f' % FIB4_ENS3.pvalue), ' ', ('%0.5f' % ENS3_ENS3.pvalue), ' ', ('%0.5f' % ENS3_EXP.pvalue), ' ']
#        row5= [('%0.5f' % APRI_EXP.pvalue), ' ', ('%0.5f' % FIB4_EXP.pvalue), ' ', ('%0.5f' % ENS3_EXP.pvalue), ' ', ('%0.5f' % EXP_EXP.pvalue), ' ']
#
#    if (metric_expert == None):
#        cell_text.append(row1)
#        cell_text.append(row2)
#        cell_text.append(row3)
#        cell_text.append(row4)
#    else:
#        cell_text.append(row1)
#        cell_text.append(row2)
#        cell_text.append(row3)
#        cell_text.append(row4)        
#        cell_text.append(row5)        
             
#    the_table = plt.table(cellText=cell_text,
#                      rowLabels=rows,
#                      colLabels=columns,
#                      cellLoc='center',
#                      bbox = [0,-0.4,1,0.3],
#                      loc='top')    
#    the_table.auto_set_font_size(False)
#    the_table.set_fontsize(12)
#    plt.show()    
#    return metric_a, metric_f, metric_e

# 1. Calculate empirical confidence intervals 
# 2. Use KS test to get a p-value to quantify if one distribution is different than another 

ds = 'McGill'
APRI = alg()
FIB4 = alg()
ENS3 = alg()
EXP = alg()

if (ds == 'Toronto'):
    ENS3_ds = read('Toronto_ENS3_bootstrap_low.xlsx')
    APRI_ds = read('Toronto_APRI_bootstrap.xlsx')
    FIB4_ds = read('Toronto_FIB4_bootstrap.xlsx')
elif (ds == 'McGill'):
    ENS3_ds = read('McGill_ENS3_bootstrap_0.45_nonNAFLonly.xlsx')
    APRI_ds = read('McGill_APRI_bootstrap_nonNAFLonly.xlsx')
    FIB4_ds = read('McGill_FIB4_bootstrap_nonNAFLonly.xlsx')
elif (ds == 'Expert'):
    ENS3_ds = read('Expert_ENS3_bootstrap_0.665.xlsx')
    APRI_ds = read('Expert_APRI_bootstrap.xlsx')
    FIB4_ds = read('Expert_FIB4_bootstrap.xlsx')
    EXP_ds = read('EXP_averaged.xlsx')

APRI.split('APRI', APRI_ds, 1000)
FIB4.split('FIB4', FIB4_ds, 1000)
ENS3.split('ENS3', ENS3_ds, 1000)
if (ds=='Expert'):
    EXP.split('EXP_avg', EXP_ds, 1000)


# Okay. Task 1. Plot distributions   
if (ds !='Expert'):
    plot_dist(APRI.sens, FIB4.sens, ENS3.sens, 'Sensitivity')
    plot_dist(APRI.spec, FIB4.spec, ENS3.spec, 'Specificity')
    plot_dist(APRI.ppv, FIB4.ppv, ENS3.ppv, 'PPV')
    plot_dist(APRI.npv, FIB4.npv, ENS3.npv, 'NPV')
    plot_dist(APRI.acc, FIB4.acc, ENS3.acc, 'Accuracy')
    plot_dist(APRI.AUROC, FIB4.AUROC, ENS3.AUROC, 'AUROC')
    plot_dist(APRI.AUPRC, FIB4.AUPRC, ENS3.AUPRC, 'AUPRC')
    plot_dist(APRI.det, FIB4.det, ENS3.det, 'Percentage of Determinate Cases')
else:
    plot_dist(APRI.sens, FIB4.sens, ENS3.sens, 'Sensitivity', EXP.sens)
    plot_dist(APRI.spec, FIB4.spec, ENS3.spec, 'Specificity', EXP.spec)
    plot_dist(APRI.ppv, FIB4.ppv, ENS3.ppv, 'PPV', EXP.ppv)
    plot_dist(APRI.npv, FIB4.npv, ENS3.npv, 'NPV', EXP.npv)
    plot_dist(APRI.acc, FIB4.acc, ENS3.acc, 'Accuracy', EXP.acc)
    plot_dist(APRI.AUROC, FIB4.AUROC, ENS3.AUROC, 'AUROC')
    plot_dist(APRI.AUPRC, FIB4.AUPRC, ENS3.AUPRC, 'AUPRC')
    plot_dist(APRI.det, FIB4.det, ENS3.det, 'Percentage of Determinate Cases', EXP.det)
# Test the KS distribution tool 

