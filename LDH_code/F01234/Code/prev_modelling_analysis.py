import numpy as np
import pandas as pd 

data = pd.read_csv('/Volumes/Chamber of Secrets/Thesis/Code/Lancet Digital Health Code/F01234/Predictions/prevalence modelling/pm_results_0.1_0.5.csv', index_col=0)
data.drop(columns={'trial'}, inplace=True)
data['prevalence'] = data['prevalence'].astype(str)
data['sens'] *= 100
data['spec'] *= 100
data['ppv'] *= 100
data['npv'] *= 100

data.replace({'prevalence': {'0.1001001001001001': 0.1, '0.2004008016032064': 0.2,
              '0.30090270812437314': 0.3, '0.4016064257028112': 0.4,
              '0.5025125628140703': 0.5}}, inplace=True)
data['prevalence'] = data['prevalence'].astype(float)

means = data.groupby(['prevalence', 'algorithm']).mean()
means.reset_index(inplace=True)
CI_low = data.groupby(['prevalence', 'algorithm']).quantile(0.025)
CI_high = data.groupby(['prevalence', 'algorithm']).quantile(0.975)

data['prevalence'] = np.where(data['prevalence'] == '0.1001001001001001', '0.1', data['prevalence'])
data['prevalence'] = np.where(data['prevalence'] == '0.2004008016032064', '0.2', data['prevalence'])
data['prevalence'] = np.where(data['prevalence'] == '0.30090270812437314', '0.3', data['prevalence'])
data['prevalence'] = np.where(data['prevalence'] == '0.4016064257028112', '0.4', data['prevalence'])
data['prevalence'] = np.where(data['prevalence'] == '0.5025125628140703', '0.5', data['prevalence'])

APRI = means.loc[means['algorithm'] == 'APRI']
FIB4 = means.loc[means['algorithm'] == 'FIB4']
ENS_TE = means.loc[means['algorithm'] == 'ENS_TE']
ENS_EXP = means.loc[means['algorithm'] == 'ENS_EXP']