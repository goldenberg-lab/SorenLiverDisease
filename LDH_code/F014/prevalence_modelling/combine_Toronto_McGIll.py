import pandas as pd
import pyodbc as db 


columns = ['Sex', 'Age', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 
            'Creatinine', 'INR', 'Platelets', 'BMI', 'Diabetes', 'Hb', 
            'WBC', 'Fibrosis', 'patientID', 'bx_date', 'reckey_enc', 'TE',
            'TE_date', 'missingness', 'Alcohol', 'Autoimmune Hepatitis', 
            'Cholestasis', 'DrugInducedLiverInjury', 'Hepatitis B',
            'Hepatitis C', 'MixedEnzymeAbnormality', 'NAFL', 
            'BiliaryCirrhosis', 'SclerosingCholangitis', 'WilsonsDisease', 
            'Etiology', 'Neoplasm', 'Other']
            
conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Darth\Desktop\Thesis\Data\Fibrosis.accdb;')
filepath = r'C:\Users\Darth\Desktop\Thesis\Code\Evaluating Trained Algorithms\final_2000-2014no(hb,wbc)'
sql_dict = {'Toronto': 'SELECT * FROM _TorontoHoldOut30',
            'McGill': 'SELECT * FROM _McGillData',
            'McGill-TE': 'SELECT * FROM _McGillData_August2020 WHERE NEOPLASM=0 AND missingness <= 3 AND Fibrosis IS NOT NULL AND TE IS NOT NULL'}

toronto = pd.read_sql(sql_dict['Toronto'], db.connect(conn_str))
mcgill = pd.read_sql(sql_dict['McGill'], db.connect(conn_str))
mcgill_TE = pd.read_sql(sql_dict['McGill-TE'], db.connect(conn_str))

new_toronto = toronto[columns]
new_mcgill = mcgill[columns]
new_mcgill_TE = mcgill_TE[columns]

tor_mcg = pd.concat([new_toronto, new_mcgill])
tor_mcg_TE = pd.concat([new_toronto, new_mcgill_TE])

tor_mcg.to_excel(r'C:\Users\Darth\Desktop\Thesis\Data\August 2020 Data\tor_mcg.xlsx')
tor_mcg_TE.to_excel(r'C:\Users\Darth\Desktop\Thesis\Data\August 2020 Data\tor_mcg_TE.xlsx')


# print(toronto.columns)
# print(mcgill.columns)
# print(mcgill_TE.columns)

# toronto_set = set(toronto.columns)
# mcgill_set = set(mcgill.columns)
# mcgill_TE_set = set(mcgill_TE.columns)

# # print(toronto_set - mcgill_set)
# # print(mcgill_set - toronto_set)

# print(toronto_set - mcgill_TE_set)
# print(mcgill_set - mcgill_TE_set)
# print(mcgill_TE_set - toronto_set)
# print(mcgill_TE_set - mcgill_set)

