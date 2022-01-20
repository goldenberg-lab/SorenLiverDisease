import ast 
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats as st
import matplotlib.pyplot as plt 
import math
from matplotlib.font_manager import FontProperties

pd.options.mode.chained_assignment = None  # default='warn'


plt.close('all')

# Okay. Let's step back. What is the problem I am trying to sovle?
# Error bars are not informative because the sensitivities are focused at extreme values due to the small number of examples. 
# Box, violin, or scatter plots are more informative. 

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

def plot_alg(df, data_df, mk, mn, colors, ylims, p_val_dict, mode): 
    """
    mk: Metric Key
    mn: Metric Name 
    """
    if mode != 'ALL':
        plt.figure(figsize=(12,2))
    else:
        plt.figure(figsize=(12,3))
    
    #plt.ylim([0,100])
    plt.grid(True)
    plt.xlabel('Prevalence (%)')
    plt.ylabel(mn + ' (%)')
    plt.title('Prevalence vs. ' + mn)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(True)

    frame1.set_facecolor('white')
    if len(df['alg'].unique()) == 3:
        pos = [-0.015, 0, 0.015]
    elif (len(df['alg'].unique()) == 5): 
        pos = [-0.01, -0.005,0,0.005, 0.01]
    
    cell_text =  []
    table_colors = []
    table_rows = []
    table_columns = ['0.1%', '0.2%', '0.3%', '0.4%', '0.5%']

    for c, alg in enumerate(df['alg'].unique()):
        print(mn, alg, c, colors[alg])
        tdf = df.loc[df['alg'] == alg][['prevalence', mk, mk + '_025', mk + '_975']].sort_values(by=['prevalence'])
        plt.scatter(tdf['prevalence'] + pos[c], tdf[mk], color=colors[alg], marker='o', label=alg, s=20)
        plt.errorbar(tdf['prevalence'] + pos[c], tdf[mk], yerr=[tdf[mk + '_025'], tdf[mk + '_975']], ecolor=colors[alg], capsize=3, barsabove=True, ls='none', linewidth=1)
        plt.ylim(ylims[mk])
        table_colors.append(colors[alg])
        table_rows.append(alg)
        
        # print(tdf[mk])
        # print(tdf[mk + '_025'])
        # print(tdf[mk + '_975'])
        
        if (mk != 'npv'):
            if (mk in ['sens', 'spec', 'npv', 'per_det']):
                cell_text.append(['%0.1f (%0.1f - %0.1f)' % ((x,y,z)) for (x,y,z) in zip(tdf[mk], tdf[mk] - tdf[mk + '_025'], tdf[mk] + tdf[mk + '_975'])])
            else:
                cell_text.append(['%0.2f (%0.2f - %0.2f)' % ((x,y,z)) for (x,y,z) in zip(tdf[mk], tdf[mk] - tdf[mk + '_025'], tdf[mk] + tdf[mk + '_975'])])
        else:
            cell_text.append(['%0.2f (%0.2f - %0.2f)' % ((x,y,z)) for (x,y,z) in zip(tdf[mk], tdf[mk] - tdf[mk + '_025'], tdf[mk] + tdf[mk + '_975'])])

        
    for key in p_val_dict.keys(): 
        
        p_val_list = []
        table_rows.append(key)
        table_colors.append('white')

        for prev in tdf['prevalence'].unique(): 
            
            
        
            pvd = p_val_dict[key]
            tdf2 = data_df.loc[(data_df['prevalence'] == prev) & (data_df['alg'].isin(pvd))]
        
            alg1 = tdf2.loc[tdf2['alg'] == pvd[0]][mk]
            alg2 = tdf2.loc[tdf2['alg'] == pvd[1]][mk]

            # Step 1. Check normality 
            _, norm_p_1 = st.shapiro(alg1)
            _, norm_p_2 = st.shapiro(alg2)
            
            
            mean_1 = np.mean(alg1)
            mean_2 = np.mean(alg2)
            
            print(mn)
            
            if ((norm_p_1 < 0.05) or (norm_p_2 < 0.05)): # Not normal, use mann-whitney U statistic 
                stat, p = st.mannwhitneyu(alg1, alg2)
            
                print('NON-NORMAL - mean_1: %0.5f, mean_2: %0.4f, test_stat: %0.4f, p-val: %0.4f'\
                  %(mean_1, mean_2, stat, p))
            
                if (p < 0.0001):
                    p_val_string = '<0.0001'
                else:
                    p_val_string = ('%0.4f' % (p))
                p_val_list.append(p_val_string)
            else: # Both normal, use 1 sided t-test 
                big = None 
                small = None 
                
                if (mean_1 > mean_2):  
                    big = alg1
                    small = alg2
                elif (mean_1 <= mean_2):
                    big = alg2 
                    small = alg1
            
                stat, p = st.ttest_ind(big, small, equal_var=False)
                print('NORMAL - mean_1: %0.4f, mean_2: %0.4f, test_stat: %0.4f, p-val: %0.40f'\
                      %(np.mean(big), np.mean(small), stat, p))  

                s1sq = np.var(big)    
                n1 = len(big)
                v1 = n1-1
                
                s2sq = np.var(small)
                n2 = len(small)
                v2 = n2-1
                
                my_dof = ((s1sq/n1 + s2sq/n2)**2)/((s1sq**2)/((n1**2)*v1) + (s2sq**2)/((n2**2)*v2))
                
                man_p_val = 1 - st.t.cdf(stat, my_dof)
                print('Manually calculated p-val: %0.10f' % (man_p_val))
                
                if (p < 0.0001):
                    p_val_string = '<0.0001'
                else:
                    p_val_string = ('%0.4f' % (p))
            
                p_val_list.append(p_val_string)
                pass

        input('Batman')
        cell_text.append(p_val_list)

        # print('\n\n\nTable Rows:')
        # print(table_rows)   
        # print('\n\n\nCell Text:')         
        # print(cell_text)
        
        # print('\n')
        
        
            # table_rows.append(key)

    # print('NOW PRINTING TABLE ROWS!')
    # print(table_rows)
    # print('Now printing cell text!')
    # print(cell_text)

    # for key in p_val_dict.keys(): # Iterate through p-values, update row labels and calculate p-values for each prevalence 
    #     print(key)
    #     # table_rows.append(key)
    #     # cell_text.append(['0' for x in range(1,6)])
        
    # print(table_rows)
    # print(table_columns)
    # print(cell_text)
    
    # print('\n' + metric)
    # p_val_list = []
    # for i in range(0,len(A['prev'])):
        
    #     # Okay. I need to determine the p-values to show in the table.
    #     f_data = e(F[metric + '_data'][i])
    #     e_data = e(E2e[metric + '_data'][i])         
        
    #     # Step 1. Check if both are normal:
    #     _, f_norm_p = st.shapiro(f_data)
    #     _, e_norm_p = st.shapiro(e_data)
        
    #     f_mean = np.mean(f_data)
    #     e_mean = np.mean(e_data)
        
    #     if ((f_norm_p < 0.05) or (e_norm_p < 0.05)): # Not normal, use mann-whitney U-statistic
    #         stat, p = st.mannwhitneyu(f_data, e_data)
            
    #         print('i:%d, NON-NORMAL - f-mean: %0.3f, e-mean: %0.3f, test_stat: %0.3f, p-val: %0.3f'\
    #               %(i, f_mean, e_mean, stat, p))
            
    #         if (p < 0.001):
    #             p_val_string = '<0.001'
    #         else:
    #             p_val_string = ('%0.3f' % (p))
            
    #         p_val_list.append(p_val_string)
    #     else:   # both normal, use 1 sided t-test
    #         # My goal is to test if one is bigger than the other 
    #         if (e_mean > f_mean): 
    #             big = e_data
    #             small = f_data
    #         else:
    #             big = f_data
    #             small = e_data
        
    #         stat, p = st.ttest_ind(big, small, equal_var=False)
            
    #         print('i:%d, NORMAL - f-mean: %0.3f, e-mean: %0.3f, test_stat: %0.3f, p-val: %0.10f'\
    #               %(i, f_mean, e_mean, stat, p))
            
    #         s1sq = np.var(big)    
    #         n1 = len(big)
    #         v1 = n1-1
            
    #         s2sq = np.var(small)
    #         n2 = len(small)
    #         v2 = n2-1
            
    #         my_dof = ((s1sq/n1 + s2sq/n2)**2)/((s1sq**2)/((n1**2)*v1) + (s2sq**2)/((n2**2)*v2))
            
    #         man_p_val = 1 - st.t.cdf(stat, my_dof)
    #         print('Manually calculated p-val: %0.10f' % (man_p_val))
            
    #         if (p < 0.001):
    #             p_val_string = '<0.001'
    #         else:
    #             p_val_string = ('%0.3f' % (p))
        
    #         p_val_list.append(p_val_string)
    #         pass
        
    # cell_text.append(p_val_list)
    
    #table_rows.append('APRI vs. ENS_APRI p-value')
    ts_x =0
    te_x = 1-ts_x
    
    ts_y = -0.75
    te_y = 0.75
    
    the_table = plt.table(cellText=cell_text,
                  rowLabels=table_rows,
                  rowColours=table_colors,
                  colLabels=table_columns,
                  cellLoc='center',
                  bbox = [ts_x,ts_y,te_x,te_y], # (left-x, bottom-y, length-x, length-y)
                  loc='bottom')
    the_table.auto_set_font_size(False)
    for (row, col), cell in the_table.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
   
    plt.show()

    return None 

folder_path = 'C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\Prevalence Modelling\\'
data = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/prevalence modelling/pm_gaussian_noise(0.5).csv', index_col=0)
metrics = {'sens': 'Sensitivity', 
           'spec': 'Specificity', 
           'ppv': 'PPV', 
           'npv': 'NPV', 
           'per_det': '% of Dataset'}

color_dict = {'APRI(1,2)': 'cyan', 
              'FIB4(1.45,3.25)': 'lightblue', 
              'ENS_TE(0.465)': 'orangered', 
              'ENS_EXP(0.6)': 'red', 
              'ENS_APRI_ALL(0.736)': 'pink',
              'ENS_APRI_det(0.736)': 'red', 
              'ENS_APRI_indet(0.736)': 'salmon', 
              'ENS_FIB4_det(0.6)': 'red', 
              'ENS_FIB4_indet(0.6)': 'salmon'}

ylims = {'APRI': {'sens': [0,100], 
                  'spec': [0,100], 
                  'ppv': [0,5], 
                  'npv': [99,100], 
                  'per_det': [0,100]},
         'FIB4': {'sens': [0,100], 
                  'spec': [0,100], 
                  'ppv': [0,5], 
                  'npv': [99,100], 
                  'per_det': [0,100]},
             'ALL': {'sens': [0,100], 
                     'spec': [0,100], 
                     'ppv': [0,5], 
                     'npv': [99,100], 
                     'per_det': [0,100]}}

order = {'APRI': ['APRI(1,2)', 'ENS_APRI_det(0.736)', 'ENS_APRI_indet(0.736)'],
         'FIB4': {'FIB4(1.45,3.25)': 0, 'ENS_FIB4_det(0.6)': 1, 'ENS_FIB4_indet(0.6)': 2}, 
         'ALL': {'APRI(1,2)': 0, 'ENS_APRI_ALL(0.736)': 1, 'FIB4(1.45,3.25)': 2, 'ENS_FIB4_ALL/EXP(0.6)': 3, 'ENS_TE(0.465)': 4}}

p_vals = {'APRI': {'APRI vs. ENS_APRI_det p-val': ['APRI(1,2)', 'ENS_APRI_det(0.736)']},
          'FIB4': {'FIB4 vs. ENS_FIB4_det p-val': ['FIB4(1.45,3.25)', 'ENS_FIB4_det(0.6)']},
          'ALL': {'APRI vs. ENS_APRI_all p-val': ['APRI(1,2)', 'ENS_APRI_ALL(0.736)'], 
                  'APRI vs. ENS_TE p-val': ['APRI(1,2)', 'ENS_TE(0.465)'], 
                  'APRI vs. ENS_EXP p-val': ['APRI(1,2)', 'ENS_EXP(0.6)'],
                  'FIB4 vs. ENS_TE p-val': ['FIB4(1.45,3.25)', 'ENS_TE(0.465)'], 
                  'FIB4 vs. ENS_FIB4/EXP p-val': ['FIB4(1.45,3.25)', 'ENS_EXP(0.6)']}}
          
data.drop(columns={'trial'}, inplace=True)
for met in metrics.keys(): 
    data[met] *= 100
data['prevalence'] *= 100

# I need to fix percentage of determinates and indeterminates for APRI, ENS_APRI_det, ENS_APRI_indet, FIB4, ENS_FIB4_det, ENS_FIB4_indet
# Don't complain, just do the work and it will get done faster. 

mean = data.groupby(['prevalence', 'alg']).mean()
lo_q = data.groupby(['prevalence', 'alg']).quantile(0.025)
hi_q = data.groupby(['prevalence', 'alg']).quantile(0.975)

for col in mean.columns.tolist():
    lo_q.rename(columns={col: col + "_lo"}, inplace=True)
    hi_q.rename(columns={col: col + "_hi"}, inplace=True)
       
adata = mean.merge(lo_q, left_index=True, right_index=True, how='left')
adata = adata.merge(hi_q, left_index=True, right_index=True, how='left')    
adata.reset_index(inplace=True)

for met in metrics.keys(): 
    adata[met + '_025'] = adata[met] - adata[met + '_lo']
    adata[met + '_975'] = adata[met + '_hi'] - adata[met]

# Case 1: APRI det/indet performance 
# APRI(1,2), ENS_APRI_det(0.736), ENS_APRI_indet(0.736)
# APRI_data = adata.loc[]
APRI_df = adata.loc[(adata['alg'].str.contains('APRI')) & (adata['alg'] != 'ENS_APRI_ALL(0.736)')]
#APRI_df = APRI_df[['prevalence', 'alg', 'per_det', 'per_det_lo', 'per_det_hi', 'per_det_025', 'per_det_975']]

oAPRI_df = APRI_df.copy()

for prev in APRI_df['prevalence'].unique():
    ptdf = APRI_df.loc[(APRI_df['prevalence'] == prev) & (APRI_df['alg']=='APRI(1,2)')]
    per_det = ptdf['per_det'].iloc[0]
    per_det_lo = ptdf['per_det_lo'].iloc[0]
    per_det_hi = ptdf['per_det_hi'].iloc[0]
    per_det_025 = ptdf['per_det_025'].iloc[0]
    per_det_975 = ptdf['per_det_975'].iloc[0]
    
    APRI_df.loc[(APRI_df['prevalence'] == prev) & (APRI_df['alg'] == 'ENS_APRI_det(0.736)'), 'per_det'] = per_det
    APRI_df.loc[(APRI_df['prevalence'] == prev) & (APRI_df['alg'] == 'ENS_APRI_det(0.736)'), 'per_det_lo'] = per_det_lo
    APRI_df.loc[(APRI_df['prevalence'] == prev) & (APRI_df['alg'] == 'ENS_APRI_det(0.736)'), 'per_det_hi'] = per_det_hi
    APRI_df.loc[(APRI_df['prevalence'] == prev) & (APRI_df['alg'] == 'ENS_APRI_det(0.736)'), 'per_det_025'] = per_det_025
    APRI_df.loc[(APRI_df['prevalence'] == prev) & (APRI_df['alg'] == 'ENS_APRI_det(0.736)'), 'per_det_975'] = per_det_975

    APRI_df.loc[(APRI_df['prevalence'] == prev) & (APRI_df['alg'] == 'ENS_APRI_indet(0.736)'), 'per_det'] = 100-per_det
    APRI_df.loc[(APRI_df['prevalence'] == prev) & (APRI_df['alg'] == 'ENS_APRI_indet(0.736)'), 'per_det_lo'] = 100-per_det_hi
    APRI_df.loc[(APRI_df['prevalence'] == prev) & (APRI_df['alg'] == 'ENS_APRI_indet(0.736)'), 'per_det_hi'] = 100-per_det_lo

APRI_df['per_det_025'] = np.where(APRI_df['alg'] == 'ENS_APRI_indet(0.736)', APRI_df['per_det'] - APRI_df['per_det_lo'], APRI_df['per_det_025'])
APRI_df['per_det_975'] = np.where(APRI_df['alg'] == 'ENS_APRI_indet(0.736)', APRI_df['per_det_hi'] - APRI_df['per_det'], APRI_df['per_det_975'])
APRI_data = data.loc[data['alg'].isin(APRI_df['alg'])]


# Case 2: FIB4 det/indet performance 
# FIB4(1.45,3.25), ENS_FIB4_DET(0.6), ENS_FIB4_indet(0.6)
FIB4_df = adata.loc[(adata['alg'].str.contains('FIB4')) & (adata['alg'] != 'ENS_FIB4_ALL(0.6)')]
#FIB4_df = FIB4_df[['prevalence', 'alg', 'per_det', 'per_det_lo', 'per_det_hi', 'per_det_025', 'per_det_975']]

oFIB4_df = FIB4_df.copy()

for prev in FIB4_df['prevalence'].unique():
    ptdf = FIB4_df.loc[(FIB4_df['prevalence'] == prev) & (FIB4_df['alg']=='FIB4(1.45,3.25)')]
    per_det = ptdf['per_det'].iloc[0]
    per_det_lo = ptdf['per_det_lo'].iloc[0]
    per_det_hi = ptdf['per_det_hi'].iloc[0]
    per_det_025 = ptdf['per_det_025'].iloc[0]
    per_det_975 = ptdf['per_det_975'].iloc[0]
    
    FIB4_df.loc[(FIB4_df['prevalence'] == prev) & (FIB4_df['alg'] == 'ENS_FIB4_det(0.6)'), 'per_det'] = per_det
    FIB4_df.loc[(FIB4_df['prevalence'] == prev) & (FIB4_df['alg'] == 'ENS_FIB4_det(0.6)'), 'per_det_lo'] = per_det_lo
    FIB4_df.loc[(FIB4_df['prevalence'] == prev) & (FIB4_df['alg'] == 'ENS_FIB4_det(0.6)'), 'per_det_hi'] = per_det_hi
    FIB4_df.loc[(FIB4_df['prevalence'] == prev) & (FIB4_df['alg'] == 'ENS_FIB4_det(0.6)'), 'per_det_025'] = per_det_025
    FIB4_df.loc[(FIB4_df['prevalence'] == prev) & (FIB4_df['alg'] == 'ENS_FIB4_det(0.6)'), 'per_det_975'] = per_det_975

    FIB4_df.loc[(FIB4_df['prevalence'] == prev) & (FIB4_df['alg'] == 'ENS_FIB4_indet(0.6)'), 'per_det'] = 100-per_det
    FIB4_df.loc[(FIB4_df['prevalence'] == prev) & (FIB4_df['alg'] == 'ENS_FIB4_indet(0.6)'), 'per_det_lo'] = 100-per_det_hi
    FIB4_df.loc[(FIB4_df['prevalence'] == prev) & (FIB4_df['alg'] == 'ENS_FIB4_indet(0.6)'), 'per_det_hi'] = 100-per_det_lo

FIB4_df['per_det_025'] = np.where(FIB4_df['alg'] == 'ENS_FIB4_indet(0.6)', FIB4_df['per_det'] - FIB4_df['per_det_lo'], FIB4_df['per_det_025'])
FIB4_df['per_det_975'] = np.where(FIB4_df['alg'] == 'ENS_FIB4_indet(0.6)', FIB4_df['per_det_hi'] - FIB4_df['per_det'], FIB4_df['per_det_975'])
FIB4_df['order'] = FIB4_df['alg']
FIB4_df.replace({'order': order['FIB4']}, inplace=True)
FIB4_df.sort_values(by=['prevalence', 'order'], inplace=True)
FIB4_data = data.loc[data['alg'].isin(FIB4_df['alg'])]

# Case 3: APRI, FIB4, ENS_EXP, ENS_TE performance 
# APRI, FIB4, ENS_APRI_ALL, ENS_FIB4/ENS_EXP, ENS_TE 
all_algs = ['APRI(1,2)', 'ENS_APRI_ALL(0.736)', 
            'FIB4(1.45,3.25)', 'ENS_FIB4_ALL(0.736)', 
            'ENS_TE(0.465)', 'ENS_EXP(0.6)']
ALL_df = adata.loc[(adata['alg'].isin(all_algs))]
ALL_df['order'] = ALL_df['alg']
ALL_df.replace({'order': order['ALL']}, inplace=True)
ALL_df.sort_values(by=['prevalence', 'order'], inplace=True)
ALL_data = data.loc[data['alg'].isin(ALL_df['alg'])]
    

for met in metrics.keys(): 
    #plot_alg(APRI_df, APRI_data, met, metrics[met], color_dict, ylims['APRI'], p_vals['APRI'], 'APRI')
    #plot_alg(FIB4_df, FIB4_data, met, metrics[met], color_dict, ylims['FIB4'], p_vals['FIB4'], 'FIB4')
    plot_alg(ALL_df, ALL_data, met, metrics[met], color_dict, ylims['ALL'], p_vals['ALL'], 'ALL')
   



