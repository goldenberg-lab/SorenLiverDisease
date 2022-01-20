import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

pd.options.mode.chained_assignment = None

data = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/feature_importance/feature_importances.csv', index_col=0)
data.drop(columns={'i'}, inplace=True)
data.replace({'dataset': {'Toronto': 'TLC', 'McGill': 'MUHC'}}, inplace=True)
data.replace({'alg': {'MLP': 'ANN'}}, inplace=True)

mean = data.groupby(['dataset', 'feature', 'alg']).mean().rename(columns={'del_auroc': 'auroc'}) 
lo_q = data.groupby(['dataset', 'feature', 'alg']).quantile(0.025).rename(columns={'del_auroc': 'lo_q'})
hi_q = data.groupby(['dataset', 'feature', 'alg']).quantile(0.975).rename(columns={'del_auroc': 'hi_q'})

res = mean.merge(lo_q, left_index=True, right_index=True, how='left')
res = res.merge(hi_q, left_index=True, right_index=True, how='left')
res.reset_index(inplace=True)

res['lo_err'] = res['auroc'] - res['lo_q']
res['hi_err'] = res['hi_q'] - res['auroc']

res['max'] = res['auroc'] + res['hi_err']
res['min'] = res['auroc'] - res['lo_err']

max_val = res['max'].max()*1.1
min_val = res['min'].min()*1.1

ds = ['TLC', 'MUHC', 'Combined']
algs = ['SVM', 'RFC', 'GBC', 'LOG', 'ANN', 'ENS']
features = {'Age': 0, 
            'Sex': 1, 
            'Albumin': 2, 
            'ALP': 3, 
            'ALT': 4, 
            'AST': 5, 
            'Bilirubin': 6, 
            'Creatinine': 7, 
            'INR': 8,
            'BMI': 9,
            'Platelets': 10,
            'Diabetes': 11}

indices = np.arange(1, len(features)+1, 1) - 0.5
width = 0.25
x_pos = [indices - width, indices, indices + width]
params = {'width': width, 'edgecolor': 'black', 'linewidth': 0.5}
colors = ['dodgerblue', 'darkorange', 'forestgreen', 'white', 'white']
rows = ['%s' % x for x in ds]

for alg in algs: 
    plt.figure(figsize=(21,3))
    plt.xlabel('Parameter')
    plt.ylabel('Δ 100*AUROC', fontweight='bold')
    plt.title(alg + ' Univariate Permutation Parameter Importance', fontweight='bold')
    plt.ylim([min_val, max_val])
    plt.xlim([0,12])
    #plt.xticks(indices, feats)
    plt.grid(True, axis='y', color='gray', linestyle='--')
    
    idx_pos = 0
    cell_text = []
    
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.set_facecolor('white')    

    for key in ds: 
        t_ds = res.loc[(res['dataset'] == key) & (res['alg'] == alg)]
        t_ds['feat_count'] = t_ds['feature']
        t_ds.replace({'feat_count': features}, inplace=True)
        t_ds.sort_values(by=['feat_count'], inplace=True)
        t_ds.drop(columns={'feat_count'}, inplace=True)
               
        plt.bar(x_pos[idx_pos], t_ds['auroc'], label=key, yerr=[t_ds['lo_err'], t_ds['hi_err']], color=colors[idx_pos], **params, capsize=3)
        cell_text.append(['%0.3f' % x for x in  t_ds['auroc']])
        idx_pos += 1
    
    
    the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=list(features.keys()),
                      cellLoc='center',
                      bbox = [0,-0.3,1,0.3],
                      loc='bottom', fontsize=20)
    the_table.set_fontsize(20)

    for (row, col), cell in the_table.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    plt.subplots_adjust(left=0.4, bottom=0.2)