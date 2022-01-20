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

def plot_dist(low, high, metric_text):
    plt.figure(figsize=(10,4))
    plt.hold(True)
    plt.title(metric_text)
    plt.hist(np.array(low), bins=np.arange(0, 100,1).tolist(), label='ENS3(45%)', alpha=0.5, linewidth=1, edgecolor='black')
    plt.hist(np.array(high), bins=np.arange(0,100,1).tolist(), label='ENS3(66.5%)', alpha=0.5, linewidth=1, edgecolor='black')

    plt.legend(loc='upper left')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    low_mean = np.mean(low)
    low_emp_ci_low = low_mean - np.percentile(low, 2.5)
    low_emp_ci_high = np.percentile(low,  97.5) - low_mean

    high_mean = np.mean(high)
    high_emp_ci_low = high_mean - np.percentile(high, 2.5)
    high_emp_ci_high = np.percentile(high,  97.5) - high_mean  
    
    cell_text = []
    
    columns = ('ENS45-mean', 'ENS45-95%CI', 'ENS66-mean', 'ENS66-95%CI')
    rows = ['Empirical CI', 'ENS45 p-val', 'ENS66 p-val']

    E45_E45 = st.ttest_rel(low, low)
    E45_E66 = st.ttest_rel(low, high)
    E66_E66 = st.ttest_rel(high, high) 
    
    row1= [('%0.2f' % low_mean), ('%0.2f' %low_emp_ci_low) + '-' + ('%0.2f' % low_emp_ci_high), ('%0.2f' % high_mean), ('%0.2f' % high_emp_ci_low) + '-' + ('%0.2f' % high_emp_ci_high)]
    row2= [('%0.5f' % E45_E45.pvalue), ' ', ('%0.5f' % E45_E66.pvalue), ' ']
    row3= [('%0.5f' % E45_E66.pvalue), ' ', ('%0.5f' % E66_E66.pvalue), ' ' ]
  
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
    

# 1. Calculate empirical confidence intervals 
# 2. Use KS test to get a p-value to quantify if one distribution is different than another 

ENS45 = alg()
ENS66 = alg()

ENS45_ds = read('Toronto_ENS3_low.xlsx')
ENS66_ds = read('Toronto_ENS3_high.xlsx')

ENS45.split('ENS3_45', ENS45_ds, 100)
ENS66.split('ENS3_665', ENS66_ds, 100)

plot_dist(ENS45.sens, ENS66.sens, 'sensitivity')
plot_dist(ENS45.spec, ENS66.spec,  'specificity')
plot_dist(ENS45.ppv, ENS66.ppv,  'ppv')
plot_dist(ENS45.npv, ENS66.npv, 'npv')
plot_dist(ENS45.acc, ENS66.acc, 'acc')
plot_dist(ENS45.AUROC, ENS66.AUROC,'AUROC')
plot_dist(ENS45.AUPRC, ENS66.AUPRC,  'AUPRC')
plot_dist(ENS45.det, ENS66.det, 'det')

