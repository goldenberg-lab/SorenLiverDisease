# Etiology check

import pyodbc as db 
import pandas as pd
import numpy as np 

conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Soren\Desktop\Thesis\Fibrosis.accdb;')
cnxn = db.connect(conn_str)
cursor = cnxn.cursor()

mrns = pd.read_sql('SELECT MRN FROM _ExpertPredsCombined WHERE Fibrosis=0', cnxn)
etiologies = pd.read_sql("SELECT * FROM [Etiology of Liver Disease]", cnxn)
etiologies = etiologies.loc[etiologies['MRN'].isin(mrns['MRN'].tolist())]
etiologies = etiologies.sort_values(by='MRN')

for e in etiologies.iterrows():
    print(e)
    print('')
    input('Press enter to continue')


# Hep B:    14
# Hep C:    6
# Alcohol:  4
# NAFL:     5
# Chol:     5
# PBC:      4
# Wils:     4
# Enzy:     5
# AIH:      4
# PSC:      3
# Drug:     6