import pyodbc as db
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

class variance(object): 
    def __init__(self):
        self.albumin = []
        self.alp = []
        self.alt = []
        self.ast = []
        self.bilirubin=[]
        self.creatinine=[]
        self.BMI = []
        self.INR = []
        self.Platelets = []
        self.GGT = []
        self.Sodium = []
        
def reset_variance(obj):
        obj.albumin = []
        obj.alp = []
        obj.alt = []
        obj.ast = []
        obj.bilirubin=[]
        obj.creatinine=[]
        obj.BMI = []
        obj.INR = []
        obj.Platelets = []
        obj.GGT = []
        obj.Sodium = []
        
def histogram(obj, variable, desc):
    plt.grid = True
    plt.hist(obj, bins=50)
    plt.ylabel('Variance')
    plt.title(variable + ' Variance (' + desc+ ')')
    
def one80Days(obj, df, MRN, eventDate, startDate, endDate):
    temp_df = df.loc[(df['MRN'] == MRN) & (df['EventDate'] > startDate) & (df['EventDate'] < endDate)]
    if (len(temp_df) > 0):
        obj.append(np.var(temp_df.iloc[:,1].values))
        
def everyDays(obj, df, MRN):
    temp_df = df.loc[(df['MRN'] == MRN)].iloc[:,1].values
    if (len(temp_df) > 0):
        obj.append(np.var(temp_df))
        
conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Soren\Desktop\Thesis\Fibrosis.accdb;')
cnxn = db.connect(conn_str)
cursor = cnxn.cursor()
sql = "SELECT * FROM "

Fibrosis = pd.read_sql(sql + "Fibrosis WHERE Fibrosis.FIBROSIS <> 'F 2' AND Fibrosis.FIBROSIS <> 'F 3'", cnxn).iloc[:,[0,4]]
albumin = pd.read_sql(sql + "Albumin WHERE Albumin_value > 0", cnxn).iloc[:,0:3]
alp = pd.read_sql(sql + "ALP WHERE ALP_Value > 0", cnxn).iloc[:,0:3]
alt = pd.read_sql(sql + "ALT WHERE ALT_VALUE > 0", cnxn).iloc[:,0:3]
ast = pd.read_sql(sql + "AST WHERE AST_VALUE > 0", cnxn).iloc[:,0:3]
bilirubin = pd.read_sql(sql + "Bilirubin WHERE BILIRUBIN_VALUE > 0", cnxn).iloc[:,0:3]
creatinine = pd.read_sql(sql + "Creatinine WHERE Creatinine_Value > 0", cnxn).iloc[:,0:3]
BMI = pd.read_sql(sql + "BMI WHERE BMI_VALUE > 0", cnxn).iloc[:,0:3]
INR = pd.read_sql(sql + "INR WHERE INR_VALUE > 0", cnxn).iloc[:,0:3]
Platelets = pd.read_sql(sql + "Platelets WHERE Platelets_Value > 0", cnxn).iloc[:,0:3]
GGT = pd.read_sql(sql + "GGT WHERE GGT_VALUE > 0", cnxn).iloc[:,0:3]
Sodium = pd.read_sql(sql + "Sodium WHERE SODIUM_VALUE IS NOT NULL", cnxn).iloc[:,0:3]

allDays = variance()
_180Days = variance()
previous_MRNs = set()

# Okay. Focus. What is the most effective way to do this? 
# For each MRN and Event Date pair in biopsies (5 min)
#   # Loop through each blood test, and find results with same MRN and within +/- 180 days of Entry Date 
#   # Calculate the variance for that set of 180 day observations for that patient 
#   # Repeat for all other measurements for that same patient 
#   # Repeat for each blood test 
reset_variance(allDays)
reset_variance(_180Days)

for index, row in Fibrosis.iterrows():
    
    MRN = row['MRN']
    EventDate = pd.to_datetime(row['EventDate'])
    startDate = EventDate - timedelta(days=180)
    endDate = EventDate + timedelta(days=180)
    print(index)
    
    one80Days(_180Days.albumin, albumin, MRN, EventDate, startDate, endDate)
    one80Days(_180Days.alp, alp, MRN, EventDate, startDate, endDate)
    one80Days(_180Days.alt, alt, MRN, EventDate, startDate, endDate)
    one80Days(_180Days.ast, ast, MRN, EventDate, startDate, endDate)
    one80Days(_180Days.bilirubin, bilirubin, MRN, EventDate, startDate, endDate)
    one80Days(_180Days.creatinine, creatinine, MRN, EventDate, startDate, endDate)
    one80Days(_180Days.BMI, BMI, MRN, EventDate, startDate, endDate)
    one80Days(_180Days.INR, INR, MRN, EventDate, startDate, endDate)
    one80Days(_180Days.Platelets, Platelets, MRN, EventDate, startDate, endDate)
    one80Days(_180Days.GGT, GGT, MRN, EventDate, startDate, endDate)
    one80Days(_180Days.Sodium, Sodium, MRN, EventDate, startDate, endDate)

    if (MRN in previous_MRNs):
        continue
    previous_MRNs.add(MRN)
    
    everyDays(allDays.albumin, albumin, MRN)
    everyDays(allDays.alp, alp, MRN)
    everyDays(allDays.alt, alt, MRN)
    everyDays(allDays.ast, ast, MRN)
    everyDays(allDays.bilirubin, bilirubin, MRN)
    everyDays(allDays.creatinine, creatinine, MRN)
    everyDays(allDays.BMI, BMI, MRN)
    everyDays(allDays.INR, INR, MRN)
    everyDays(allDays.Platelets, Platelets, MRN)
    everyDays(allDays.GGT, GGT, MRN)
    everyDays(allDays.Sodium, Sodium, MRN)

histogram(_180Days.albumin, 'Albumin', '180 day intervals')
histogram(allDays.albumin, 'Albumin', 'all measurements')

histogram(_180Days.alp, 'ALP', '180 day intervals')
histogram(allDays.alp, 'ALP', 'all measurements')

histogram(_180Days.alt, 'ALT', '180 day intervals')
histogram(allDays.alt, 'ALT', 'all measurements')

histogram(_180Days.ast, 'AST', '180 day intervals')
histogram(allDays.ast, 'AST', 'all measurements')

histogram(_180Days.bilirubin, 'bilirubin', '180 day intervals')
histogram(allDays.bilirubin, 'bilirubin', 'all measurements')

histogram(_180Days.creatinine, 'creatinine', '180 day intervals')
histogram(allDays.creatinine, 'creatinine', 'all measurements')

histogram(_180Days.BMI, 'BMI', '180 day intervals')
histogram(allDays.BMI, 'BMI', 'all measurements')

histogram(_180Days.INR, 'INR', '180 day intervals')
histogram(allDays.INR, 'INR', 'all measurements')

histogram(_180Days.Platelets, 'Platelets', '180 day intervals')
histogram(allDays.Platelets, 'Platelets', 'all measurements')

histogram(_180Days.GGT, 'GGT', '180 day intervals')
histogram(allDays.GGT, 'GGT', 'all measurements')

histogram(_180Days.Sodium, 'Sodium', '180 day intervals')
histogram(allDays.Sodium, 'Sodium', 'all measurements')