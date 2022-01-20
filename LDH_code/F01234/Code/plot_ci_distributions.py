import os
import shutil
import numpy as np
import pandas as pd 
import scipy.stats as st
import matplotlib.pyplot as plt 
import matplotlib.style as style
from matplotlib.font_manager import FontProperties


class alg:
    def __init__(self, name, df, metrics, tot_recs=None):
        print(name)
        self.name = name
        self.df = process(df.loc[df['algorithm'] == name], metrics)
        
        sens_res = get_results(self.df['sens'])
        spec_res = get_results(self.df['spec'])
        ppv_res = get_results(self.df['ppv'])
        npv_res = get_results(self.df['npv'])
        acc_res = get_results(self.df['acc'])
        auroc_res = get_results(self.df['auroc'])
        auprc_res = get_results(self.df['auprc'])
        per_det_res = get_results(self.df['per_det'])
        
        self.results = {
            'sens_vals': self.df['sens'],
            'sens_mean': sens_res[0], 
            'sens_low': sens_res[1],
            'sens_high': sens_res[2],
            
            'spec_vals': self.df['spec'],
            'spec_mean': spec_res[0], 
            'spec_low': spec_res[1],
            'spec_high': spec_res[2], 
            
            'ppv_vals': self.df['ppv'],
            'ppv_mean': ppv_res[0], 
            'ppv_low': ppv_res[1],
            'ppv_high': ppv_res[2], 
            
            'npv_vals': self.df['npv'],
            'npv_mean': npv_res[0], 
            'npv_low': npv_res[1],
            'npv_high': npv_res[2],   
            
            'acc_vals': self.df['acc'],
            'acc_mean': acc_res[0], 
            'acc_low': acc_res[1],
            'acc_high': acc_res[2], 
            
            'auroc_vals': self.df['auroc'],
            'auroc_mean': auroc_res[0], 
            'auroc_low': auroc_res[1],
            'auroc_high': auroc_res[2], 
                    
            'auprc_vals': self.df['auprc'],
            'auprc_mean': auprc_res[0], 
            'auprc_low': auprc_res[1],
            'auprc_high': auprc_res[2], 
            
            'per_det_vals': self.df['per_det'],
            'per_det_mean': per_det_res[0], 
            'per_det_low': per_det_res[1],
            'per_det_high': per_det_res[2], 
            }

def process(df, mets):
    return df.sort_values(by=['trial']).reset_index(drop=True)[mets]

def get_results(s):
    avg = np.mean(s)
    low = avg - np.percentile(s, 2.5)
    high = np.percentile(s,97.5) - avg
    
    return (avg, low, high)

def plot_distributions(algs_dict, algs_show, alg_colours, mets, key, mode):
    
    '''
    algs_dict: A dictionary containing all the algorithm objects 
    algs_show: A list of the algorithms to be shown in the plot 
    mets:      A list of performance metrics to 

    '''
    
    style.use('seaborn-dark')
    
    met_dict = {'sens': 'Sensitivity',
                'spec': 'Specificity', 
                'ppv': 'PPV', 
                'npv': 'NPV', 
                'acc': 'Accuracy',
                'auroc': 'AUROC',
                'auprc': 'AUPRC',
                'per_det': '% Determinate'}
    
    dataset_plot_path ='/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/CI_bootstrap/' + key + '/plots/distributions/'
    # old_files = os.listdir(dataset_plot_path)
    # for f in old_files: 
    #     os.remove(f)
    
    # Okay. What are the steps I need to take? 
    # I need histograms of each algorithm
    # I need p-values for each algorithm 
    # I need the histograms to be formatted in a certain way. 
    
    # Okay. The next step is to generate a table underneath each graph that has the mean, 2.5th, and 97.5th percentile
    # The bottom of the table should then include p-value calculations using wilcoxon signed rank test for each algorithm in a grid 
    
    
    for met in mets: # Need a seperate plot for each metric in the dataset
        # if (met == 'num_recs'):
        #     continue
    
        print('\n\n\nCurrent metric: %s ' % (met) )
    
        # if (met != 'per_det'):
        #     continue
    
        if ((met == 'auroc') or (met == 'auprc')):
            xlab = str('100*') + met_dict[met]
        else:
            xlab = met_dict[met] + ' (%)'
    
        plt.figure(figsize=(9,3))
        plt.title('Empirical ' + met_dict[met] + ' Distribution', fontsize=15)
        plt.ylabel('Frequency', fontsize=14)
        plt.xlabel(xlab, fontsize=14)
        plt.grid(True)

        # Get column names: 
        columns = []
        for al in algs_show: 
            text = al.partition('(')
            columns.append(text[0] + '\n' + '(' + text[-1])

        cell_text = []
        #rows = ['2.5th percentile', 'mean', '97.5th percentile'] + ['vs. ' + al + ' p-val' for al in algs_show]
        rows = ['Mean (2.5th - 97.5th)'] + [al.split('(')[0] + ' p-value' for al in algs_show]

        colors = [alg_colours[al] for al in algs_show]

        dist_row = []
        
        for alg in algs_show: 
            # print(algs_dict[alg].df[met]*100)
            # print(np.arange(0,100,1).tolist())
            plt.hist(algs_dict[alg].df[met]*100, bins=np.arange(0,101,1).tolist(), label=alg, alpha=0.5, linewidth=1, edgecolor='black', color=alg_colours[alg])
            low = 100*(algs_dict[alg].results[met + '_mean'] - algs_dict[alg].results[met + '_low'])
            mean = 100*algs_dict[alg].results[met + '_mean']
            high = 100*(algs_dict[alg].results[met + '_high'] + algs_dict[alg].results[met + '_mean'])
            
            low_str = str('%0.1f' % low)
            mean_str = str('%0.1f' % mean)
            high_str = str('%0.1f' % high)
            
            dist_row.append(('%s (%s - %s)' % (mean_str.rjust(5, ' '), low_str.rjust(5, ' '), high_str.rjust(5, ' '))))            
            
           
        cell_text.append(dist_row)
        
        for alg1 in algs_show: 
            p_val_list = []
            for alg2 in algs_show: 
                if (alg2 == alg1): 
                    p_val_string = '-'
                else:
                    
                        
                    print('\nAlg1: %s, Alg2: %s ' % (alg1, alg2))
                    alg1_met_vals = algs_dict[alg1].results[met + '_vals']
                    alg2_met_vals = algs_dict[alg2].results[met + '_vals']
                    

                    
                    p_val_string = get_p_value(alg1_met_vals, alg2_met_vals, met)
                    
                p_val_list.append(('%s' % (p_val_string)))
            cell_text.append(p_val_list)
        
        ts_x =0
        te_x = 1-ts_x
        
        ts_y = -1.1
        te_y = 0.9
        
        the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      colColours=colors,
                      rowColours=['white'] + colors,
                      cellLoc='center',
                      bbox = [ts_x,ts_y,te_x,te_y], # (left-x, bottom-y, length-x, length-y)
                      #bbox = [0.4,-0.4,0.4,0.3],
                      loc='top')    
        the_table.auto_set_font_size(False)
        if (len(algs_show) == 4):
            the_table.set_fontsize(14)
        else:
            the_table.set_fontsize(10.5)
        cellDict = the_table.get_celld()
        # Change the size of the first row of cells
        for i in range(0, len(columns)):
            cellDict[(0,i)].set_height(0.1)
            #cellDict[(0,i)].set_fontsize(20)
        
        #break
        
        
        plt.savefig(dataset_plot_path + mode + '_' +  met + '_Distribution', bbox_inches='tight')
    
    return None
    
def get_p_value(data1, data2, met):
    # Okay. I need to determine the p-values to be shown in the table. 
    
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    
    if ((mean1 == mean2) and (np.var(mean1) == np.var(mean2)) and met=='per_det'):
        return '-'
    
    print('Alg1 mean: %0.4f, Alg2 mean: %0.4f' % (mean1, mean2))
    
    if (mean1 > mean2): 
        big = data1
        small = data2
    else:
        big = data2
        small = data1
    
    # Step 1. Check if both are normal: 
    _, big_norm_p = st.shapiro(big)
    _, small_norm_p = st.shapiro(small)
    
    if ((big_norm_p <= 0.05) or (small_norm_p <= 0.05)):
       stat, p = st.mannwhitneyu(big, small)
       print('NON-NORMAL - big-mean: %0.3f, small-mean: %0.3f, test_stat: %0.3f, p-val: %0.4f'\
              %(np.mean(big), np.mean(small), stat, p))

       if (p < 0.0001):
           p_val_string = '< 0.0001'
       else:
           p_val_string = ('%0.4f' % (p))

    else:
       stat, p = st.ttest_ind(big, small, equal_var=False)
       print('NORMAL - big-mean: %0.3f, small-mean: %0.3f, test_stat: %0.3f, p-val: %0.40f'\
              %(np.mean(big), np.mean(small), stat, p))
        
       s1sq = np.var(big)    
       n1 = len(big)
       v1 = n1-1
        
       s2sq = np.var(small)
       n2 = len(small)
       v2 = n2-1
        
       my_dof = ((s1sq/n1 + s2sq/n2)**2)/((s1sq**2)/((n1**2)*v1) + (s2sq**2)/((n2**2)*v2))
        
       man_p_val = 1 - st.t.cdf(stat, my_dof)
       print('Manually calculated p-val: %0.40f' % (man_p_val))
        
       if (p < 0.0001):
           p_val_string = '< 0.0001'
       else:
           p_val_string = ('%0.4f' % (p))

    return p_val_string


# Step 1. Load in relevant data based on condition (DONE)
# Step 2. Generate empirical bootstrapped distributions and use these to calculate p-values 
# Step 3. Generate bar plots with appropriate colors and descriptions (for main paper) 
# Step 4. Write the code so that I can generate the appropriate plot by changing a few parameters 

dk = 'NAFL' # Toronto, Expert, McGill, NAFL, TE
mode = 'all' # APRI_det, APRI_indet, FIB4_det, FIB4_indet, all, NFS_det, NFS_indet, Expert, TE_det
 
e_co = {'Toronto': {'APRI_det': 0.55, 'APRI_indet': 0.55, 'FIB4_det': 0.5, 'FIB4_indet': 0.5, 'all': 0.6},
        'McGill': {'APRI_det': 0.7525, 'APRI_indet': 0.7525, 'FIB4_det': 0.5875, 'FIB4_indet': 0.5875, 'all': 0.6},
        'Expert': {'all': 0.6},
        'TE': {'all': 0.6}, 
        'NAFL': {'NFS_det': 0.585, 'NFS_indet': 0.585, 'all': 0.6}}

# Load in the relevant dataset based on dk 
ds_path = '/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/CI_bootstrap/' 
metrics = {'sens': 'Sensitivity', 'spec': 'Specificity', 'ppv': 'Positive Predictive Value', 'npv': 'Negative Predictive Value', 
           'acc': 'Accuracy', 'auroc': 'AUROC', 'auprc': 'AUPRC', 'per_det': '% of Dataset'}
num_records = {'Toronto': 104, 'Expert': 45, 'McGill': 404, 'NAFL': 333, 'TE': 172}

# For each dataset, I need to get separate dfs for sens, spec, ppv, npv, accuracy, AUROC, AUPRC, and % of dataset
# For each metric, I need to generate the empirical distributions and plot them 
# In the same code, I need to do the appropriate statistical tests and plot the appropriate bar graph and title 

# Colors: Red for expert level and medium violet red for te level 

et = 'ENS(' +str(e_co[dk][mode]) + ')'
et2 = 'ENS(' +str(e_co[dk][mode]) + ',' +  str(e_co[dk][mode]) + ')'    
dk_path = ds_path + dk + '/data/1k_pms/' 

def get_ci(algs): 
    res = []
    for alg in algs: 
        ad_lo = {'name': alg.name + '_low'}
        ad_hi = {'name': alg.name + '_high'}
            
        for met in metrics.keys():    
            ad_lo[met] = round(alg.results[met + '_low']*100,1)
            ad_hi[met] = round(alg.results[met + '_high']*100,1)
        res.append(ad_lo)
        res.append(ad_hi)
    res_df = pd.DataFrame.from_records(res)
    return res_df 
        

# Plotting distributions
if dk == 'Toronto' or dk == 'McGill':     
    # Case 1: APRI_det vs. ENS                      # One CI plot, fig1
    if mode == 'APRI_det':
        APRI_det_df = pd.read_csv(dk_path + 'APRI_det.csv', index_col=0)
        Adet_APRI = alg('APRI(1,2)', APRI_det_df,metrics.keys())
        Adet_ENS1 = alg(et2, APRI_det_df, metrics.keys())
        Adet_dict = {'APRI(1,2)': Adet_APRI, et: Adet_ENS1}
        Adet_colours = {'APRI(1,2)': 'cyan', et: 'red'}
        plot_distributions(Adet_dict, Adet_dict.keys(), Adet_colours, metrics.keys(), dk, mode)
        algs = [Adet_APRI, Adet_ENS1]
        
    # Case 2: APRI_indet ENS                        # One CI plot, fig1
    elif mode == 'APRI_indet':
        APRI_indet_df = pd.read_csv(dk_path + 'APRI_indet.csv', index_col=0)
        Aindet_ENS1 = alg(et2, APRI_indet_df, metrics.keys())
        Aindet_dict = {et: Aindet_ENS1}
        Aindet_colours = {et: 'salmon'}
        plot_distributions(Aindet_dict, Aindet_dict.keys(), Aindet_colours, metrics.keys(), dk, mode)
        algs = [Aindet_ENS1]
    
    # Case 3: FIB4_det vs. ENS                      # One CI plot, fig2
    elif mode == 'FIB4_det':
        FIB4_det_df = pd.read_csv(dk_path + 'FIB4_det.csv', index_col=0)
        Fdet_FIB4 = alg('FIB4(1.45,3.25)', FIB4_det_df,metrics.keys())
        Fdet_ENS1 = alg(et2, FIB4_det_df, metrics.keys())
        Fdet_dict = {'FIB4(1.45,3.25)': Fdet_FIB4, et: Fdet_ENS1}
        Fdet_colours = {'FIB4(1.45,3.25)': 'lightblue', et: 'red'}
        plot_distributions(Fdet_dict, Fdet_dict.keys(), Fdet_colours, metrics.keys(), dk, mode)
        algs = [Fdet_FIB4, Fdet_ENS1]
    
    # Case 4: FIB4_indet ENS                        # One CI plot, fig2
    elif mode == 'FIB4_indet':
        FIB4_indet_df = pd.read_csv(dk_path + 'FIB4_indet.csv', index_col=0)
        Findet_ENS1 = alg(et2, FIB4_indet_df, metrics.keys())
        Findet_dict = {et: Findet_ENS1}
        Findet_colours = {et: 'salmon'}
        plot_distributions(Findet_dict, Findet_dict.keys(), Findet_colours, metrics.keys(), dk, mode)        
        algs = [Findet_ENS1]

    # Case 5: APRI vs. FIB4 vs. ENS_EXP vs. ENS_TE on all patients
    # Did I calculate the two ENSEMBLES at the same time or not? 
    elif mode == 'all': 
        
        et = 'ENS(' +str(e_co[dk][mode]) + ')'
        et2 = 'ENS(' +str(e_co[dk][mode]) + ',' +  str(e_co[dk][mode]) + ')'    
        dk_path = ds_path + dk + '/data/1k_pms/' 

        ALL_df = pd.read_csv(dk_path + mode + '.csv', index_col=0)
        #ALL_df.rename(columns={'num_recs': 'per_det'}, inplace=True)
        ALL_APRI = alg('APRI(1,2)', ALL_df, metrics.keys())
        ALL_FIB4 = alg('FIB4(1.45,3.25)', ALL_df, metrics.keys())
        ALL_ENS_TE = alg('ENS(0.465,0.465)', ALL_df, metrics.keys())
        ALL_ENS_EXP = alg('ENS(0.6,0.6)', ALL_df, metrics.keys())
        ALL_dict = {'APRI(1,2)': ALL_APRI, 'FIB4(1.45,3.25)': ALL_FIB4, 'ENS_TE(0.465)': ALL_ENS_TE, 'ENS_EXP(0.6)': ALL_ENS_EXP}
        ALL_colors = {'ENS_TE(0.465)': 'orangered', 'ENS_EXP(0.6)': 'red', 'APRI(1,2)': 'cyan', 'FIB4(1.45,3.25)': 'lightblue'}
        plot_distributions(ALL_dict, ALL_dict.keys(), ALL_colors, metrics.keys(), dk, mode)
        
        algs = [ALL_APRI, ALL_FIB4, ALL_ENS_TE, ALL_ENS_EXP]

            
elif dk == 'Expert':
    ALL_df = pd.read_csv(dk_path + 'all.csv', index_col=0)
    ALL_EXP = alg('EXP(0.5,0.5)', ALL_df, metrics.keys())
    ALL_ENS_TE = alg('ENS(0.465,0.465)', ALL_df, metrics.keys())
    ALL_ENS_EXP = alg('ENS(0.6,0.6)', ALL_df, metrics.keys())
    
    ALL_dict = {'EXP(0.5)': ALL_EXP, 'ENS_TE(0.465)': ALL_ENS_TE, 'ENS_EXP(0.6)': ALL_ENS_EXP}
    ALL_colors = {'ENS_TE(0.465)': 'orangered', 'ENS_EXP(0.6)': 'red', 'EXP(0.5)': 'lightgreen'}
    plot_distributions(ALL_dict, ALL_dict.keys(), ALL_colors, metrics.keys(), dk, mode)
    algs = [ALL_EXP, ALL_ENS_TE, ALL_ENS_EXP]


elif dk == 'TE':
    ALL_df = pd.read_csv(dk_path + 'all.csv', index_col=0)
    ALL_TE = alg('TE(8,8)', ALL_df, metrics.keys())
    ALL_ENS_TE = alg('ENS(0.465,0.465)', ALL_df, metrics.keys())
    ALL_ENS_EXP = alg('ENS(0.6,0.6)', ALL_df, metrics.keys())

    ALL_dict = {'TE(8)': ALL_TE, 'ENS_TE(0.465)': ALL_ENS_TE, 'ENS_EXP(0.6)': ALL_ENS_EXP}
    ALL_colors = {'ENS_TE(0.465)': 'orangered', 'ENS_EXP(0.6)': 'red', 'TE(8)': 'plum'}
    plot_distributions(ALL_dict, ALL_dict.keys(), ALL_colors, metrics.keys(), dk, mode)
   
    algs = [ALL_TE, ALL_ENS_TE, ALL_ENS_EXP]

    
elif dk == 'NAFL':
    if mode == 'all':
        ALL_df = pd.read_csv(dk_path + 'all.csv', index_col=0)
        ALL_NFS = alg('NAFL(-1.455,0.675)', ALL_df, metrics.keys())
        ALL_ENS_TE = alg('ENS(0.465,0.465)', ALL_df, metrics.keys())
        ALL_ENS_EXP = alg('ENS(0.6,0.6)', ALL_df, metrics.keys())

        ALL_dict = {'NFS(-1.455,0.675)': ALL_NFS, 'ENS_TE(0.465)': ALL_ENS_TE, 'ENS_EXP(0.6)': ALL_ENS_EXP}
        ALL_colors = {'ENS_TE(0.465)': 'orangered', 'ENS_EXP(0.6)': 'red', 'NFS(-1.455,0.675)': 'yellow'}
        
        plot_distributions(ALL_dict, ALL_dict.keys(), ALL_colors, metrics.keys(), dk, mode)
        algs = [ALL_NFS, ALL_ENS_TE, ALL_ENS_EXP]

        
    elif mode == 'NFS_det':
        NFS_det_df = pd.read_csv(dk_path + mode + '.csv', index_col=0)
        Ndet_NFS = alg('NAFL(-1.455,0.675)', NFS_det_df,metrics.keys())
        Ndet_ENS1 = alg(et2, NFS_det_df, metrics.keys())
        Ndet_dict = {'NFS(-1.455,0.675)': Ndet_NFS, et: Ndet_ENS1}
        Ndet_colours = {'NFS(-1.455,0.675)': 'yellow', et: 'red'}
        plot_distributions(Ndet_dict, Ndet_dict.keys(), Ndet_colours, metrics.keys(), dk, mode)
        
        algs = [Ndet_NFS, Ndet_ENS1]

    elif mode == 'NFS_indet':
        NFS_det_df = pd.read_csv(dk_path + mode + '.csv', index_col=0)
        Ndet_ENS1 = alg(et2, NFS_det_df, metrics.keys())
        Ndet_dict = {et: Ndet_ENS1}
        Ndet_colours = {et: 'salmon'}
        plot_distributions(Ndet_dict, Ndet_dict.keys(), Ndet_colours, metrics.keys(), dk, mode)
        
        algs = [Ndet_ENS1]

# Get CI's for plotting bar graph 
CI_df = get_ci(algs)
CI_df.to_csv(ds_path + '/' + dk + '/data/CIs/' + mode + '_CI.csv')
        
