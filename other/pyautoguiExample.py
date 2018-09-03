import pyautogui as py
import time as tm

# SAMPLE TASK: OPEN AN EXCEL FILE AND EXTRACT A SAMPLE PATIENT ID
sW, sH = py.size()
py.moveTo(sW/2, sH/2)
tm.sleep(0.5)
py.moveTo(sW/2-25, sH-15, duration=2)
py.doubleClick()
py.moveTo(sW/3 + 15, sH/3+ 15, duration=2)
py.click()
py.typewrite('This is a sample MRN!', interval=0.1)

