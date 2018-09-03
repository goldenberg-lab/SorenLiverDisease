import pyodbc as db
import pandas as pd
import numpy as np 
import threading

def calculateCorrectedBMI():
    #threading.Timer(30.0, calculateCorrectedBMI).start()
    #print('Threading started')
    conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Soren\Desktop\Thesis\Fibrosis.accdb;')
    cnxn = db.connect(conn_str)
    cursor = cnxn.cursor()
    cursor.execute("UPDATE BMI Set BMI_Value=(Weight/(Height*Height/(100*100))) WHERE Weight IS NOT NULL AND Weight <> 0 and Height Is Not Null and Height <> 0" )
    cursor.close()
    cnxn.commit()
    cnxn.close()

def updateBMIHeights(): 
    conn = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Soren\Desktop\Thesis\Fibrosis.accdb;')
    cnxn = db.connect(conn)
    cursor = cnxn.cursor()
    cursor.execute("SELECT BMI.MRN FROM BMI WHERE BMI_Value IS NULL ORDER BY MRN Asc")
    rows = cursor.fetchall()
    
    for i in range(0, len(rows)-1):# len(rows)-1):
        print(str(i) + "/" + str(len(rows)-1))
        #print(rows[i][0])
        cursor.execute("SELECT BMI.HEIGHT FROM BMI WHERE BMI.HEIGHT IS NOT NULL AND MRN='" + str(rows[i][0])+"'")
        heights = cursor.fetchall()
        #print(heights)
        if (len(heights) > 0):
            height = heights[0][0]
            cursor.execute("UPDATE BMI SET HEIGHT=" + str(height) + " WHERE MRN='" + str(rows[i][0])+"' AND HEIGHT IS NULL")
            cnxn.commit()

        #input('Press enter to continue')
    return rows

def updateCreatinine():
    conn = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'r'DBQ=C:\Users\Soren\Desktop\Thesis\Fibrosis.accdb;')
    cnxn = db.connect(conn)
    cursor = cnxn.cursor()
    cursor.execute("UPDATE CREATININE SET Creatinine_Number=Creatinine_Value")
    cnxn.commit()
    cnxn.close()
#rows = updateBMIHeights()

calculateCorrectedBMI()

updateCreatinine()

#3032


