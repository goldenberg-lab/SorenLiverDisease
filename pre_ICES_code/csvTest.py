import pandas as pd
import numpy as np 

fibrosis = pd.read_csv('C:/Users/Soren/Desktop/Thesis/Data Transfer Tests/Fibrosis.csv')
fibrosis = fibrosis.loc[(fibrosis["Fibrosis"] == 'F 4')]
print(fibrosis)

temp_df = df.loc[(df['MRN'] == MRN) & (df['EventDate'] > startDate) & (df['EventDate'] < endDate)]
pd.to