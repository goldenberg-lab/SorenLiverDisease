import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.style as style
from matplotlib.ticker import NullFormatter
style.use('seaborn-dark')

data = np.array([[37.8, 88.1, 77.3, 56.9, 62.1, 71.9, 64.2, 100-16.4],
                 [66.7, 82.1, 77.4, 72.7, 74.7, 82.7, 82.3, 100-27.9],
                 [82.7, 71.7, 76.8, 78.6, 77.6, 84.8, 86.8, 100-5.8]])
APRI_ENS3_p = ['< 0.001', '< 0.001', ' 0.022', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001']
FIB4_ENS3_p = ['< 0.001', '< 0.001', ' 0.54', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001']

APRI_CI = np.array([[14.12, 9.56, 18.66, 12.76, 9.92, 10.86, 13.69, 7.78],
                    [14.12, 9.15, 16.31, 12.02, 9.63, 9.81 , 15.42, 6.65]])
FIB4_CI = np.array([[16.59, 12.8, 15.4, 14.6, 10.9, 10.8, 13.7, 8.5],
                    [15.23, 11.2, 14.0, 12.5, 9.3, 8.6, 10.5, 7.9]])
ENS3_CI = np.array([[10.63, 13.49, 10.67, 13.64, 8.63, 8.21, 8.70, 4.72],
                    [9.21, 12.26 , 10.38, 11.66, 8.08, 6.78, 6.55, 3.93]])

#FIB4_CI_low = ([0, 6.06, 7.27, 1.03, 4.58, np.nan, np.nan, np.nan])
#FIB4_CI_high= ([0, 6.06, 7.27, 1.03, 4.58, np.nan, np.nan, np.nan])
#
#ENS3_CI_low = ([0, 6.06, 7.27, 1.03, 4.58, np.nan, np.nan, np.nan])
#ENS3_CI_high= ([0, 6.06, 7.27, 1.03, 4.58, np.nan, np.nan, np.nan])

#data = np.array([[37.8, 88.1, 77.3, 56.9, 62.1, 100-16.4],
#                 [66.7, 82.1, 77.4, 72.7, 74.7, 100-27.9],
#                 [82.7, 71.7, 76.8, 78.6, 77.6, 100-5.8]])
#                  [78.2, 75.5, 78.2, 75.5, 76.9, 83.9, 86.0, 100], # Formerly between 2 and 3 

#columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
#rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

plt.rcParams['figure.figsize'] = (12,5)

columns = ('Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'AUROC*100', 'AUPRC*100', '% Determinate')
rows = ['%s' % x for x in ('APRI', 'FIB-4', 'ENS2', 'APRI vs. ENS2 p-value', 'FIB-4 vs. ENS2 p-value')]

# Get some pastel shades for the colors
colors = ['cyan', 'lightblue', 'red', 'white', 'white']
n_cols = np.size(data,1)

index1 = np.linspace(0,1,np.size(data,1)+1)
index2 = []
for i in range(0, len(index1)-1):
    index2.append((index1[i] + index1[i+1])/2)
index2 = np.array(index2)

bar_width = 0.025

cell_text = []

plt.bar(index2 - bar_width, data[0,:], bar_width, color='cyan', linewidth=1, edgecolor='black', yerr=APRI_CI, capsize=3) # APRI Performance Metrics
plt.bar(index2, data[1,:], bar_width, color='lightblue', linewidth=1, edgecolor='black', yerr=FIB4_CI, capsize=3) # FIB4 Performance Metrics
plt.bar(index2 + bar_width, data[2,:], bar_width, color='red', linewidth=1, edgecolor='black', yerr=ENS3_CI, capsize=3) # FIB4 Performance Metrics
#plt.bar(index2 + (3/2)*bar_width, data[3,:], bar_width, color='red', linewidth=1, edgecolor='black') # FIB4 Performance Metrics

cell_text.append(['%0.1f' % x for x in data[0,:]])
cell_text.append(['%0.1f' % x for x in data[1,:]])
cell_text.append(['%0.1f' % x for x in data[2,:]])
cell_text.append(APRI_ENS3_p)
cell_text.append(FIB4_ENS3_p)
#cell_text.append(['%0.1f' % x for x in data[3,:]])

# Plot bars and create text labels for the table
#cell_text = []
#count = 0
#for col in range(n_cols):
##    plt.bar(bar_indices, data[row], bar_width, color=colors[row])
#    print(count)
#    print(index[count])
#    plt.bar(index[count], data[:,col], bar_width, color=colors[col])
#    cell_text.append(['%0.1f' % x for x in data[col]])
#    count += 1
# Reverse colors and text labels to display the last value at the top.
#colors = colors[::-1]
#cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      cellLoc='center',
                      bbox = [0,-0.3,1,0.3],
                      loc='bottom')

for (row, col), cell in the_table.get_celld().items():
    if (row == 0):
        cell.set_text_props(fontproperties=FontProperties(weight='bold'))

#    else:
#        cell.set_fontsize(50)
# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel('Performance (%)', fontsize=15)
plt.ylim([0,100])
#plt.xticks([0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1], label=False)
plt.xlim([0,1])
plt.grid(True, axis='y', color='gray', linestyle='--')
#plt.title('Figure 2c) Expert Test Set (34 F01, 11 F4)', fontsize=20)
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.set_facecolor('white')


plt.show()