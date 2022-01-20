import math
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.font_manager import FontProperties

style.use('seaborn-dark')

cols = ['SVM_diff', 'RFC_diff', 'GBC_diff', 'LOG_diff', 'ANN_diff', 'ENS1_diff', 'ENS2_diff']
TOR_df = pd.read_excel('TOR.xlsx', index_col=0)
MCG_df = pd.read_excel('MCG.xlsx', index_col=0)
TOR_MCG_df = pd.read_excel('TOR-MCG.xlsx', index_col=0)

a = TOR_df[cols]
b = MCG_df[cols]
c = TOR_MCG_df[cols]
max_val = max(max(a.max()),max(b.max()),max(c.max()))
min_val = min(min(a.min()),min(b.min()),min(c.min()))

max_val = math.ceil(max_val*100)/100
min_val = math.floor(min_val*100)/100

algs = ['SVM', 'RFC', 'GBC', 'LOG', 'ANN', 'ENS1', 'ENS2']
dfs = {'TLC': TOR_df, 'MUHC': MCG_df, 'Combined': TOR_MCG_df}
# feats = list(TOR_df['feat'])
feats = ['Sex', 'Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilir.', 'Creat.', 'INR', 'Platelets', 'BMI', 'Diabetes']

indices = np.arange(1, len(feats)+1, 1) - 0.5
width = 0.25
x_pos = [indices - width, indices, indices + width]
params = {'width': width, 'edgecolor': 'black', 'linewidth': 0.5}
colors = ['dodgerblue', 'darkorange', 'forestgreen', 'white', 'white']
rows = ['%s' % x for x in dfs.keys()]

for alg in algs: 
    plt.figure(figsize=(18,3))
    plt.xlabel('Parameter')
    plt.ylabel('Δ AUROC', fontweight='bold')
    plt.title(alg + ' Parameter Importance', fontweight='bold')
    plt.ylim([-1*max_val, -1*min_val])
    plt.xlim([0,12])
    #plt.xticks(indices, feats)
    plt.grid(True, axis='y', color='gray', linestyle='--')
    
    idx_pos = 0
    cell_text = []

    for key in dfs.keys(): 
        plt.bar(x_pos[idx_pos], dfs[key][alg + '_diff']*-1, label=key, color=colors[idx_pos], **params)
        cell_text.append(['%0.3f' % x for x in  dfs[key][alg + '_diff']])
        idx_pos += 1
        
    #plt.legend(loc='upper center', ncol=3)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.set_facecolor('white')
    
    the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=feats,
                      cellLoc='center',
                      bbox = [0,-0.3,1,0.3],
                      loc='bottom', fontsize=20)
    the_table.set_fontsize(20)

    for (row, col), cell in the_table.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    plt.subplots_adjust(left=0.4, bottom=0.2)
