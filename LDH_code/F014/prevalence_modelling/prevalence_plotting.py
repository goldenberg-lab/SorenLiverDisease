import ast 
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats as st
import matplotlib.pyplot as plt 
import math
from matplotlib.font_manager import FontProperties


plt.close('all')

# Okay. Let's step back. What is the problem I am trying to sovle?
# Error bars are not informative because the sensitivities are focused at extreme values due to the small number of examples. 
# Box, violin, or scatter plots are more informative. 

# Set boundaries. This project is not a priority for me
# This isn't going to help my career or be useful for anyone. 
# I need to have the courage to say no to it if I can't get anything past this week. 
# IT is already distracting from valuable study time. 
# function for calculating the t-test for two independent samples

def low_percentiles(series): 
    s = e(series)
    p25 = np.mean(s) - np.percentile(s, 2.5)
    return 100*p25

def high_percentiles(series): 
    s = e(series)
    p975 = np.percentile(s, 97.5) - np.mean(s)
    return 100*p975    

def e(a):
    return ast.literal_eval(a)

def plot(A,F,E2e,E2, metric, metric_name):
    
    A_met_err = np.zeros([2, len(A['prev'])])
    A_met_err[0,:] = A[metric + '_2.5']
    A_met_err[1,:] = A[metric + '_97.5']
                        
    F_met_err = np.zeros([2, len(F['prev'])])
    F_met_err[0,:] = F[metric + '_2.5']
    F_met_err[1,:] = F[metric + '_97.5']
    
    E2e_met_err = np.zeros([2, len(E2e['prev'])])
    E2e_met_err[0,:] = E2e[metric + '_2.5']
    E2e_met_err[1,:] = E2e[metric + '_97.5']
    
    E2_met_err = np.zeros([2, len(E2['prev'])])
    E2_met_err[0,:] = E2[metric + '_2.5']
    E2_met_err[1,:] = E2[metric + '_97.5']
    
    plt.figure(figsize=(12,2))
    plt.scatter(A['prev'] - 0.0135, A[metric], color='cyan', marker='o', label='APRI', s=20)
    plt.errorbar(A['prev'] - 0.0135,A[metric], yerr=A_met_err, ecolor='cyan', capsize=3, barsabove=True,  ls='none', linewidth=1)
    
    plt.scatter(F['prev'] - 0.0045, F[metric], color='lightblue', marker='o', label='FIB4', s=20)
    plt.errorbar(F['prev'] - 0.0045, F[metric], yerr=F_met_err, ecolor='lightblue', capsize=3, barsabove=True,  ls='none', linewidth=1)
    
    plt.scatter(E2e['prev'] + 0.0045, E2e[metric], color='chocolate', marker='o', label='ENS2e', s=20)
    plt.errorbar(E2e['prev'] + 0.0045, E2e[metric], yerr=E2e_met_err, ecolor='chocolate', capsize=3, barsabove=True,  ls='none', linewidth=1)
    
    plt.scatter(E2['prev'] + 0.0135, E2[metric], color='red', marker='o', label='ENS2', s=20)
    plt.errorbar(E2['prev'] + 0.0135, E2[metric], yerr=E2_met_err, ecolor='red', capsize=3, barsabove=True,  ls='none', linewidth=1)
    
    # Okay. I now need to add the table underneath the plot. 
    # First I need column names. 
    
    
    
    cell_text =  []
    colors = ['cyan', 'lightblue', 'chocolate', 'red', 'white']
    columns = ['0.1%', '0.2%', '0.3%', '0.4%', '0.5%']
    rows = ['APRI (1 , 2)', 'FIB-4 (1.45 , 3.25)', 'ENS2e (0.525 , 0.7)', 'ENS2 (0.25, 0.45)', 'FIB-4 vs ENS2e p-value']

    algs = [A, F, E2e, E2]
    for alg in algs:
        cell_text.append(['%0.1f (%0.1f - %0.1f)' % ((x,y,z)) for (x,y,z) in zip(alg[metric], alg[metric] - alg[metric + '_2.5'], alg[metric] + alg[metric + '_97.5'])])


    print('\n' + metric)
    p_val_list = []
    for i in range(0,len(A['prev'])):
        
        # Okay. I need to determine the p-values to show in the table.
        f_data = e(F[metric + '_data'][i])
        e_data = e(E2e[metric + '_data'][i])         
        
        # Step 1. Check if both are normal:
        _, f_norm_p = st.shapiro(f_data)
        _, e_norm_p = st.shapiro(e_data)
        
        f_mean = np.mean(f_data)
        e_mean = np.mean(e_data)
        
        if ((f_norm_p < 0.05) or (e_norm_p < 0.05)): # Not normal, use mann-whitney U-statistic
            stat, p = st.mannwhitneyu(f_data, e_data)
            
            print('i:%d, NON-NORMAL - f-mean: %0.3f, e-mean: %0.3f, test_stat: %0.3f, p-val: %0.3f'\
                  %(i, f_mean, e_mean, stat, p))
            
            if (p < 0.001):
                p_val_string = '<0.001'
            else:
                p_val_string = ('%0.3f' % (p))
            
            p_val_list.append(p_val_string)
        else:   # both normal, use 1 sided t-test
            # My goal is to test if one is bigger than the other 
            if (e_mean > f_mean): 
                big = e_data
                small = f_data
            else:
                big = f_data
                small = e_data
        
            stat, p = st.ttest_ind(big, small, equal_var=False)
            
            print('i:%d, NORMAL - f-mean: %0.3f, e-mean: %0.3f, test_stat: %0.3f, p-val: %0.10f'\
                  %(i, f_mean, e_mean, stat, p))
            
            s1sq = np.var(big)    
            n1 = len(big)
            v1 = n1-1
            
            s2sq = np.var(small)
            n2 = len(small)
            v2 = n2-1
            
            my_dof = ((s1sq/n1 + s2sq/n2)**2)/((s1sq**2)/((n1**2)*v1) + (s2sq**2)/((n2**2)*v2))
            
            man_p_val = 1 - st.t.cdf(stat, my_dof)
            print('Manually calculated p-val: %0.10f' % (man_p_val))
            
            if (p < 0.001):
                p_val_string = '<0.001'
            else:
                p_val_string = ('%0.3f' % (p))
        
            p_val_list.append(p_val_string)
            pass
        
    cell_text.append(p_val_list)


    plt.grid(True)
    plt.xlim([0.05,0.55])
    plt.xlabel('Prevalence (%)')
    plt.ylabel(metric_name)
    plt.title('Prevalence vs. ' + metric_name)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(True)

    frame1.set_facecolor('white')
    

    
    ts_x =0
    te_x = 1-ts_x
    
    ts_y = -0.75
    te_y = 0.75
    
    the_table = plt.table(cellText=cell_text,
                  rowLabels=rows,
                  rowColours=colors,
                  colLabels=columns,
                  cellLoc='center',
                  bbox = [ts_x,ts_y,te_x,te_y], # (left-x, bottom-y, length-x, length-y)
                  loc='bottom')
    the_table.auto_set_font_size(False)
    for (row, col), cell in the_table.get_celld().items():
       if (row == 0):
           cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    if (metric == 'sens'):    
        plt.ylim([-5,105])
    else:
        plt.ylim([55,100])
    plt.figure()

folder_path = 'C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\Prevalence Modelling\\'

APRI_df = pd.read_excel(folder_path + 'APRI_prev.xlsx', index_col=0)
FIB4_df = pd.read_excel(folder_path + 'FIB4_prev.xlsx', index_col=0)
ENS2e_df = pd.read_excel(folder_path + 'ENS2e_prev.xlsx', index_col=0)
ENS2_df = pd.read_excel(folder_path + 'ENS2_prev.xlsx', index_col=0)

dfs = [APRI_df, FIB4_df, ENS2e_df, ENS2_df]
for df in dfs: 
    df['sens'] = df['sens']*100
    df['spec'] = df['spec']*100
    df['det'] = df['det']*100
#dfs = [EA_df, EF_df]

for df in dfs: 
    df['sens_2.5'] = df['sens_data'].apply(low_percentiles) - 0.000001
    df['sens_97.5'] = df['sens_data'].apply(high_percentiles)
    df['spec_2.5'] = df['spec_data'].apply(low_percentiles) - 0.000001
    df['spec_97.5'] = df['spec_data'].apply(high_percentiles)
    df['det_2.5'] = df['spec_data'].apply(low_percentiles) - 0.000001
    df['det_97.5'] = df['spec_data'].apply(high_percentiles)
    
metrics = {'Sensitivity': 'sens', 'Specificity': 'spec', '% Determinate': 'det'}

for met in metrics.keys(): 
    plot(APRI_df, FIB4_df, ENS2e_df, ENS2_df, metrics[met], met)

for met in metrics.keys(): 
    # if (met == '% Determinate'):
    break
    for i in range(0,5): 
        A_data = e(APRI_df.iloc[i][metrics[met] + '_data'])
        F_data = e(FIB4_df.iloc[i][metrics[met] + '_data'])
        E_data = e(ENS2_df.iloc[i][metrics[met] + '_data'])
        E2e_data = e(ENS2e_df.iloc[i][metrics[met] + '_data'])
        
        

        
        # print('%s - %d: FIB4 normality: %0.4f ENS2e normality: %0.4f' % (met, i, fib_p_val, ens2e_p_val))

        # E_mean = np.mean(E_data)
        # E_std = np.std(E_data)
        # E_var = np.var(E_data)
        # E_N = len(E_data)
        
        # norm1 = np.random.normal(loc=E_mean, scale=E_std, size=100)
        # norm2 = np.random.normal(loc=E_mean + 0.002, scale=E_std, size=100)
        
        # F_mean = np.mean(F_data)
        # F_std = np.std(F_data)
        # F_var = np.var(F_data)
        # F_N = len(F_data)
        
        # t = (E_mean - F_mean)/np.sqrt((E_var/E_N) + (F_var/F_N))
        # dof = math.floor((((E_var/E_N) + (F_var/F_N))**2)/(((E_var**2)/((E_N**3)-(E_N**2))) + ((F_var**2)/((F_N**3)-(F_N**2)))))
        # p_value = 1 - st.t.cdf(t, dof)

        # print('T: %0.2f, dof: %d, p-val: %0.3f' % (t, dof, p_value))
        
        
        # t_score, p_value = st.ttest_ind(E_data, F_data, equal_var=False)
        # t_score2, p_value2 = st.ttest_ind(norm1, norm2, equal_var=False)
         
        plt.title('%s for prevalence: %0.1f%%' % (met, (i+1)/10))
        #plt.hist(x=A_data, bins=30, alpha=0.65, label='APRI', color='r', linewidth=0.5, edgecolor='black')
        plt.hist(x=F_data, bins=30, alpha=0.65, label='FIB4', color='g', linewidth=0.5, edgecolor='black')
        #plt.hist(x=E_data, bins=30, alpha=0.65, label='ENSe', color='b', linewidth=0.5, edgecolor='black')
        plt.hist(x=E2e_data, bins=30, alpha=0.65, label='ENS2e', color='b', linewidth=0.5, edgecolor='black')
        #plt.hist(x=norm, bins=30, alpha=0.65, label='Normal', color='y', linewidth=0.5, edgecolor='black')
        
        #print('met: %s, prev: %d, p-val: %0.20f' % (met, i, p_value))
        #print('met: %s, prev: %d, test - p-val: %0.20f' % (met, i, p_value2))
        
        plt.legend()
        plt.grid(True)
        #plt.xlabel('%s ENS3-FIB4 T-test p-val: %0.3f' % (met, p_value))
        plt.ylabel('Frequency')
        plt.figure()
    
# Okay. I want to check the normality assumption at each prevalence. 
# What do I need to do? 

# For each algorithm 
# I can also compare the ensemble to a normal distribution by 
# calculating mean and variance, then generating random numbers with that 
# distribution and plotting them. 



