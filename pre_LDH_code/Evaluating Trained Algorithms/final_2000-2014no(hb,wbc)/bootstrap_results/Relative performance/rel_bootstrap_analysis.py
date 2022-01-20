# Calculate relevant performance metrics 
# Plot results and add tables 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from beautifultable import BeautifulTable
import matplotlib.style as style
style.use('seaborn-dark')
np.set_printoptions(precision=3)

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
        
    def split(self, name, d):
        self.name = name
        self.sens = d['sens'].dropna().tolist()
        self.spec = d['spec'].dropna().tolist()
        self.ppv = d['ppv'].dropna().tolist()
        self.npv = d['npv'].dropna().tolist()
        self.acc = d['acc'].dropna().tolist()
        self.AUROC = d['AUROC'].dropna().tolist()
        self.AUPRC = d['AUPRC'].dropna().tolist()
        self.det = d['det'].dropna().tolist()

def read(path):
    data = pd.read_excel(path)
    data = data.transpose()
    return data

def plot_dist(metric_a, metric_b, metric_text):
    plt.figure(figsize=(10,4))
    plt.hold(True)
    plt.title(metric_text)
    plt.hist(np.array(metric_a), bins=np.arange(-100,100,1).tolist(), label='ENS3-FIB4', alpha=0.5, linewidth=1, edgecolor='black')
    plt.hist(np.array(metric_b), bins=np.arange(-100,100,1).tolist(), label='ENS3-APRI', alpha=0.5, linewidth=1, edgecolor='black')
    plt.legend(loc='upper left')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    A_mean = np.mean(metric_a)
    A_emp_ci_low = np.percentile(metric_a, 2.5)
    A_emp_ci_high = np.percentile(metric_a, 97.5)
    A_nor_ci_low = A_mean - 1.96*np.std(metric_a)/np.sqrt(len(metric_a))
    A_nor_ci_high = A_mean + 1.96*np.std(metric_a)/np.sqrt(len(metric_a))

    B_mean = np.mean(metric_b)
    B_emp_ci_low = np.percentile(metric_b, 2.5)
    B_emp_ci_high = np.percentile(metric_b, 97.5)
    B_nor_ci_low = B_mean - 1.96*np.std(metric_b)/np.sqrt(len(metric_b))
    B_nor_ci_high = B_mean + 1.96*np.std(metric_b)/np.sqrt(len(metric_b))
    
    
    cell_text = []
    columns = ('ENS3-FIB4-mean', 'ENS3-FIB4-95%CI', 'ENS3-APRI-mean', 'ENS3-APRI-95% CI')
    rows = ['Empirical', 'Normal']
    
    row1= [('%0.2f' % A_mean), ('%0.2f' % A_emp_ci_low) + ' | ' + ('%0.2f' % A_emp_ci_high), ('%0.2f' % B_mean), ('%0.2f' % B_emp_ci_low) + ' | ' + ('%0.2f' % B_emp_ci_high)] 
    row2= [('%0.2f' % A_mean), ('%0.2f' % A_nor_ci_low) + ' | ' + ('%0.2f' % A_nor_ci_high), ('%0.2f' % B_mean), ('%0.2f' % B_nor_ci_low) + ' | ' + ('%0.2f' % B_nor_ci_high)] 
    cell_text.append(row1)
    cell_text.append(row2)
                     
    the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      cellLoc='center',
                      bbox = [0,-0.4,1,0.3],
                      loc='top')    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    plt.show()    

# Okay. What is my goal? 
# I want to calculate the mean, as well as the percentile confidence intervals and standard normal confidence intervals to compare. 
# For the first one, I need     

ds = 'Expert'
ENS3_FIB4 = alg()
ENS3_APRI = alg()

if (ds == 'Toronto'):
    ENS3_FIB4_ds = read('Toronto_FIB4_ENS3_bootstrap.xlsx')
    ENS3_APRI_ds = read('Toronto_APRI_ENS3_bootstrap.xlsx')
elif (ds == 'McGill'):
    ENS3_FIB4_ds = read('McGill_FIB4_ENS3_bootstrap.xlsx')
    ENS3_APRI_ds = read('McGill_APRI_ENS3_bootstrap.xlsx')    
elif (ds == 'Expert'):
    ENS3_FIB4_ds = read('Expert_FIB4_ENS3_bootstrap.xlsx')
    ENS3_APRI_ds = read('Expert_APRI_ENS3_bootstrap.xlsx')    

ENS3_FIB4.split('ENS3_FIB4', ENS3_FIB4_ds)
ENS3_APRI.split('ENS3_APRI', ENS3_APRI_ds)

# Okay. Task 1. Plot distributions   
plot_dist(ENS3_FIB4.sens, ENS3_APRI.sens, 'sensitivity')
plot_dist(ENS3_FIB4.spec, ENS3_APRI.spec, 'specificity')
plot_dist(ENS3_FIB4.ppv,  ENS3_APRI.ppv, 'ppv')
plot_dist(ENS3_FIB4.npv,  ENS3_APRI.npv, 'npv')
plot_dist(ENS3_FIB4.acc,  ENS3_APRI.acc, 'acc')
plot_dist(ENS3_FIB4.AUROC, ENS3_APRI.AUROC, 'AUROC')
plot_dist(ENS3_FIB4.AUPRC, ENS3_APRI.AUPRC, 'AUPRC')
plot_dist(ENS3_FIB4.det, ENS3_APRI.det, 'det')
