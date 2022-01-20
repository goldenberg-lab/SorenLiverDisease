import numpy as np
import pandas as pd

alg = 'APRI'
dk = 'McGill'
with_indet_path = 'C:\\Users\\Darth\\Desktop\\Thesis\\Code\\Lancet Digital Health Code\\F01234\\Predictions\\AUROC vs. def_advanced_fibrosis_without_indets\\'+dk+'.csv'
#comb = 'F0vF234'

df = pd.read_csv(with_indet_path, index_col=0)[['comb', 'alg', 'auroc']]

combs = {'F0vF1': 0, 
         'F0vF2': 1,
         'F0vF3': 2, 
         'F0vF4': 3,
         'F0vF234': 4,
         'F1vF2': 5,
         'F1vF3': 6,
         'F1vF4': 7,
         'F1vF234': 8,
         'F2vF3': 9,
         'F2vF4': 10,
         'F3vF4': 11,
         'F01vF2': 12,
         'F01vF3': 13,
         'F01vF23': 14,
         'F01vF24': 15,
         'F01vF34': 16,
         'F01vF4': 17,
         'F01vF234': 18,
         'F12vF3': 19,
         'F12vF4': 20,
         'F12vF34': 21,
         'F012vF3': 22,
         'F012vF4': 23,
         'F012vF34': 24,
         'F0123vF4': 25}

df_mean = df.groupby(['comb', 'alg']).mean().reset_index()
df_lo = df.groupby(['comb', 'alg']).quantile(0.025).reset_index().rename(columns={'auroc': 'p25'})
df_hi = df.groupby(['comb', 'alg']).quantile(0.975).reset_index().rename(columns={'auroc': 'p975'})

df_all = df_mean.merge(df_lo[['comb', 'alg', 'p25']], on=['comb', 'alg'], how='left')
df_all = df_all.merge(df_hi[['comb', 'alg', 'p975']], on=['comb', 'alg'], how='right')
df_all = df_all[['comb', 'alg', 'p25', 'auroc', 'p975']]
df_all = df_all.loc[df_all['alg'] == alg]
df_all['num_sort'] = df_all['comb']
df_all['num_sort'] = df_all['num_sort'].replace(combs)
df_all.sort_values(by=['num_sort'], inplace=True)
df_all.drop(columns=['num_sort'], inplace=True)
df_all = df_all.round(3)
df_all['text'] = '(' + df_all['p25'].astype(str) + '-' + df_all['p975'].astype(str) + ')'

print(df_all.round(3))