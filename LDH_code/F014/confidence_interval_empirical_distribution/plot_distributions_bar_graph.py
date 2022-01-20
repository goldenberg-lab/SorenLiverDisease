import os
import shutil
import numpy as np
import pandas as pd 
import scipy.stats as st
import matplotlib.pyplot as plt 
import matplotlib.style as style
from matplotlib.font_manager import FontProperties


class alg:
    def __init__(self, name, df, metrics):
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
        
# Step 1, Load in the data 
# Step 2. Split the results into dataframes based on algorithms 
# Step 3. Re-create the bar-plot functionality that I had from last time 
# Step 4. Make it dynamic so that I don't have to manually go through results 

def process(df, mets):
    return df.sort_values(by=['trial']).reset_index(drop=True)[mets]

def get_results(s):
    avg = np.mean(s)
    low = avg - np.percentile(s, 2.5)
    high = np.percentile(s,97.5) - avg
    
    return (avg, low, high)

def plot_distributions(algs_dict, algs_show, alg_colours, mets, key):
    
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
                'per_det': 'Percentage of Determinate Cases',
                'auroc': 'AUROC',
                'auprc': 'AUPRC'}
    
    dataset_plot_path ='C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\ConfInterval_Distribution_Modelling\\Plots\\' + key 
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
                    p_val_string = get_p_value(alg1_met_vals, alg2_met_vals)
                    
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
        plt.savefig(dataset_plot_path + '\\' + key + '_' +  met + '_Distribution', bbox_inches='tight')
    
    return None

def plot_bargraphs(algs_dict, algs_show, df_points, algs_colours, mets, key): 
    style.use('seaborn-dark')
    
    dataset_plot_path ='C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\ConfInterval_Distribution_Modelling\\Plots\\' + key 

    
    # Loop through metrics
    # Loop through algorithms 
    # Get the point estimates as well as the confidence intervals 
    # Plot the results
    
    # df_points is a dataframe with all the metrics for the algorithm.
    columns = ('Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'AUROC*100', 'AUPRC*100', '% Determinate')
    
    not_ens = []
    for alg in algs_show:
        if (('ENS' in alg) == False):
            not_ens.append(alg)
    
    rows =  [al.split('(')[0] + ' (' + al.split('(')[1] for al in algs_show] + [al.split('(')[0] + ' vs. ENS2 p-value' for al in not_ens] # Only compare other algorithms except ENS against all other algorithms 
    
    cell_text = []

    # After I finish these plots, I need to generate plots for prevalence modelling and update the tables, then I can contact Mamatha. 
    print('----')
    
    for alg in algs_show: 
        cell_text.append(['%0.1f' % (100*df_points.loc[alg][met]) for met in mets])
        
    # I also need to calculate p-values for non-ENS algorithms 
    for nens in not_ens:  # This is the name of the non-ensemble algorithm
        # Need the ens2 df -> Get the values for the metric -> first input into wilcoxon 
        # Need the non-ens df -> Get the values for the metric -> 2nd input into wilcoxon 
        # Do the wilcoxon for that metric and append into cell text 
        p_val_list = []
        for met in mets:
            alg1_met_vals = algs_dict['ENS2(0.25 , 0.45)'].results[met + '_vals']
            alg2_met_vals = algs_dict[nens].results[met + '_vals']
            p_val_list.append(get_p_value(alg1_met_vals, alg2_met_vals))    
        cell_text.append(p_val_list)
    
    # Okay. Now, how do I actually plot these results? 
    # I can count the length of df-point, dynamically shift my bar
    # thickness based on the number of elements
    index1 = np.linspace(0,1,9)
    index2 = []
    for i in range(0, len(index1)-1):
        index2.append((index1[i] + index1[i+1])/2)
    index2 = np.array(index2)
    print(index1)
    print(index2) # Contains the midpoints for 8 side by side sets of bars. 
    # If I have an even number of bars, this needs to be the center.
    # If I have an odd number of bars, this needs to be the side
    # I'm going to have 4-5 bars on each plot - I can hardcode for this. 
    
    # bar_width = (18/160)/len(algs_show)
    # min_max = 1.5*bar_width if len(algs_show) == 4 else 2.5*bar_width

    # a = np.linspace(1/16 - min_max, 1/16 + min_max, len(algs_show)) # Position relative to midpoints where each bar should be plotted
    
    if (len(algs_show) == 4):
        plt.figure(figsize=(12,2.5))
    else:
        plt.figure(figsize=(12,2.5))
        
    if (len(algs_show) == 4):
        bar_width = 0.025
        bwa = [-1.5, -0.5, 0.5, 1.5]
    elif (len(algs_show) == 5):
        bar_width = 0.02
        bwa = [-2, -1, 0, 1, 2]

    #print('a: %s ' % (a))

    j = 0
    for alg in algs_show:
        #print('%s - %s' % (alg, index2 + a[j]))
        #print(df_points.loc[alg][mets])
        # I also need to get the error for the current algorithm 
        # algs_dict[alg].results[met + '_mean']
        
        err = np.zeros([2,len(mets)])
        
        e_idx = 0
        for met in mets: 
            l = 100*algs_dict[alg].results[met + '_low']
            h = 100*algs_dict[alg].results[met + '_high']
            err[0][e_idx] = l
            err[1][e_idx] = h
            e_idx += 1
        
        plt.bar(index2 + bwa[j]*bar_width, 100*df_points.loc[alg][mets], bar_width, linewidth=1, color=algs_colours[alg], edgecolor='black', yerr = err, capsize=3)
        j += 1
        
    colors = [algs_colours[al] for al in algs_show] + ['white' for al in not_ens]


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

    if (len(algs_show) == 4):
        the_table.set_fontsize(12)
    else:
        the_table.set_fontsize(10.5)
        
    for (row, col), cell in the_table.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    plt.ylabel('Performance (%)', fontsize=15)
    plt.ylim([0,100])
    plt.xlim([0,1])
    plt.grid(True, axis='y', color='gray', linestyle='--')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.set_facecolor('white')
    
    plt.savefig(dataset_plot_path + '\\' + key + '_' +  met + '_all_metrics', bbox_inches='tight')

    return None
    
def get_p_value(data1, data2):
    # Okay. I need to determine the p-values to be shown in the table. 
    
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    
    if ((mean1 == mean2) and (np.var(mean1) == np.var(mean2))):
        return '-'
    
    print('Alg1 mean: %0.3f, Alg2 mean: %0.3f' % (mean1, mean2))
    
    if (mean1 > mean2): 
        big = data1
        small = data2
    else:
        big = data2
        small = data1
    
    # Step 1. Check if both are normal: 
    _, big_norm_p = st.shapiro(big)
    _, small_norm_p = st.shapiro(small)
    
    if ((big_norm_p < 0.05) or (small_norm_p < 0.05)):
       stat, p = st.mannwhitneyu(big, small)
       print('NON-NORMAL - big-mean: %0.3f, small-mean: %0.3f, test_stat: %0.3f, p-val: %0.3f'\
              %(np.mean(big), np.mean(small), stat, p))

       if (p < 0.001):
           p_val_string = '< 0.001'
       else:
           p_val_string = ('%0.3f' % (p))

    else:
       stat, p = st.ttest_ind(big, small, equal_var=False)
       print('NORMAL - big-mean: %0.3f, small-mean: %0.3f, test_stat: %0.3f, p-val: %0.10f'\
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
        
       if (p < 0.001):
           p_val_string = '< 0.001'
       else:
           p_val_string = ('%0.3f' % (p))

    return p_val_string
    


datakey = 'NAFL' 
etd = {'Toronto': [(0.675, 0.75),(0.15,0.775)],
        'Expert': [(0.35,0.80),(0.675,0.675)],  # Note, second element is not @ Expert level performance but will not be shown
        'McGill': [(0.80, 0.85),(0.45,0.875)],
        'NAFL': [(0.675,0.75 ),(0.225,0.775)],
        'TE': [(0.725, 0.775),(0.775,0.775)]} 

competitor_name = {'Toronto': '',
             'Expert': 'EXPERT(0.5 , 0.5)', 
             'McGill': '', 
             'NAFL': 'NFS(-1.455 , 0.675)',
             'TE': 'TE(8 , 8)'}

dataset_algs = {'Toronto': ['APRI(1 , 2)', 'FIB-4(1.45 , 3.25)', 'ENS2(0.25 , 0.45)', 'ENS2b(0.15 , 0.775)'],
                'Expert': ['APRI(1 , 2)', 'FIB-4(1.45 , 3.25)', 'EXPERT(0.5 , 0.5)', 'ENS2(0.25 , 0.45)', 'ENS2c(0.675 , 0.675)'],
                'McGill': ['APRI(1 , 2)', 'FIB-4(1.45 , 3.25)', 'ENS2(0.25 , 0.45)', 'ENS2b(0.45 , 0.875)'],
                'NAFL': ['APRI(1 , 2)', 'FIB-4(1.45 , 3.25)', 'NFS(-1.455 , 0.675)', 'ENS2(0.25 , 0.45)', 'ENS2b(0.225 , 0.775)'],
                'TE': ['APRI(1 , 2)', 'FIB-4(1.45 , 3.25)', 'TE(8 , 8)', 'ENS2(0.25 , 0.45)', 'ENS2d(0.775 , 0.775)'],
            }

ens2bcol = 'darkorange'

colours = {'APRI(1 , 2)': 'cyan', 'FIB-4(1.45 , 3.25)': 'lightblue', 'ENS2(0.25 , 0.45)': 'red', 
              'EXPERT(0.5 , 0.5)': 'lightgreen', 'NFS(-1.455 , 0.675)': 'yellow', 'TE(8 , 8)': 'plum', 
              'ENS2b(0.15 , 0.775)': ens2bcol, 'ENS2b(0.45 , 0.875)': ens2bcol, 'ENS2b(0.225 , 0.775)': ens2bcol, 
              'ENS2d(0.775 , 0.775)': 'darkgoldenrod', 'ENS2b(0.35 , 0.8)': ens2bcol, 'ENS2c(0.675 , 0.675)': 'magenta'}

metrics = ['sens', 'spec', 'ppv', 'npv', 'acc', 'auroc', 'auprc', 'per_det']

df = pd.read_excel('C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\ConfInterval_Distribution_Modelling\\Distributions\\' + datakey +'.xlsx', index_col=0)
df_point = pd.read_excel('C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\ConfInterval_Distribution_Modelling\\Distributions\\' + datakey +'-point.xlsx', index_col=0)
df_point.index = df_point['name']
df_point = df_point.loc[df_point['name'].isin(dataset_algs[datakey])]
print(df_point)

if (datakey == 'Expert'):
    ENS_90 = 'ENS2c(' + str(etd[datakey][1][0]) + ' , ' + str(etd[datakey][1][1]) + ')'
elif (datakey == 'TE'):
    ENS_90 = 'ENS2d(' + str(etd[datakey][1][0]) + ' , ' + str(etd[datakey][1][1]) + ')'
else:
    ENS_90 = 'ENS2b(' + str(etd[datakey][1][0]) + ' , ' + str(etd[datakey][1][1]) + ')'
competitor = competitor_name[datakey]

APRI_obj = alg('APRI(1 , 2)', df, metrics)
FIB4_obj = alg('FIB-4(1.45 , 3.25)', df, metrics)
ENSbs_obj = alg('ENS2(0.25 , 0.45)', df, metrics)
ENS90_obj = alg(ENS_90, df, metrics)

algs = {'APRI(1 , 2)': APRI_obj, 'FIB-4(1.45 , 3.25)': FIB4_obj, 'ENS2(0.25 , 0.45)': ENSbs_obj, str(ENS_90): ENS90_obj}

if ((datakey != 'Toronto') and (datakey != 'McGill')):
    COMP_obj = alg(competitor, df, metrics)
    algs[competitor] = COMP_obj

plot_distributions(algs, dataset_algs[datakey], colours, metrics, datakey)
plot_bargraphs(algs, dataset_algs[datakey], df_point, colours, metrics, datakey)
        
# Okay. The next step is to generate performance plots 
# What functionality will I need? 
# 1. Plotting bar graphs and distributions 
# 2. Choosing the colors for each algorithm 
# 3. Filling in the table based on the values. 
# 4. Calculting p-values for the different distributions 
# 5. Choosing the order of the algorithms in the plot. 

# Okay. Next step. Figure out why p-value is NAN instead of 

