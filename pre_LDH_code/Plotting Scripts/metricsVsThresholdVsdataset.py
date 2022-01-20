import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

xls = pd.ExcelFile('C:\\Users\\Soren\\Desktop\\Thesis\\Data Analysis\\Hold-out Test Sets\\final_2000-2014no(hb,wbc)\\performance_thresholds.xlsx')
tor = pd.read_excel(xls, sheetname=0)
mcg = pd.read_excel(xls, sheetname=1)
exp = pd.read_excel(xls, sheetname=2)

plt.rcParams['figure.figsize'] = (11,10)
fig, ax = plt.subplots()
