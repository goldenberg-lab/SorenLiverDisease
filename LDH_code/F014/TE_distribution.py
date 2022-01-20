import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

path = r'C:\Users\Darth\Desktop\Thesis\Data\Sept2020FinalDatasets\TE.xlsx'

data = pd.read_excel(path)[['TE', 'Fibrosis']]
data = data.loc[data['TE'] <= 10]
data_low = data.loc[data['Fibrosis'] <= 2]
data_high = data.loc[data['Fibrosis'] >= 3]

plt.figure()
plt.hist(data_low['TE'], bins=np.arange(0, 15, 1).tolist(), alpha=0.5,linewidth=1, edgecolor='black')
plt.hist(data_high['TE'], bins=np.arange(0, 15, 1).tolist(), alpha=0.5,linewidth=1, edgecolor='black')            
