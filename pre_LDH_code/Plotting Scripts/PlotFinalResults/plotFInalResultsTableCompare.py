import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.style as style
from matplotlib.ticker import NullFormatter
style.use('seaborn-dark')

# First row:: Lower threshold 25%, upper threshold 45% 
# Second row: Lower thresh: 46.5%, upper thresh: 66.5%

# TLC Data 
data = np.array([[82.7, 71.7, 76.8, 78.6, 77.6, 84.8, 86.8, 94.2],
                 [73.5, 83.0, 81.8, 75.0, 78.1, 84.7, 86.5, 92.3]])

ENS45_CI = np.array([[10.09, 14.09, 11.93, 12.89, 8.12, 8.35, 9.06, 4.83],
                    [10.37, 12.68, 10.65, 11.95, 7.94, 6.74, 6.65, 3.83 ]])
ENS66_CI = np.array([[11.83, 11.81, 12.50, 11.94, 8.3, 8.59, 9.18, 5.67],
                    [11.88, 9.45, 10.38, 11.29, 7.62, 6.65, 6.68, 4.90]])


# MUHC Data 
#data = np.array([[85.0, 65.6, 44.0, 93.2, 70.3, 82.0, 53.8, 95.0],
#                 [77.2, 76.8, 49.4, 92.0, 76.9, 82.2, 54.2, 95.8]])

#ENS45_CI = np.array([[9.43, 6.87, 9.10, 4.36, 5.57, 5.61, 12.74, 2.66],
#                    [8.54, 6.40, 8.78, 4.05, 5.57, 5.30, 12.77, 2.69]])
#ENS66_CI = np.array([[10.41, 6.17, 11.17, 4.24, 5.40, 5.92, 13.01, 2.32],
#                    [10.62, 5.75, 10.42, 3.98, 5.22, 5.60, 13.27, 2.27]])


# Expert Set 
#data = np.array([[80.0, 58.1, 38.1, 90.0, 63.4, 78.4, 64.7, 91.1],
#                 [70.0, 75.0, 46.7, 88.9, 73.8, 76.2, 62.9, 93.3]])
#
#ENS45_CI = np.array([[29.95, 16.80, 20.20, 14.85, 14.82, 18.05, 29.68, 9.12],
#                    [20.05, 16.04, 20.91, 10.15, 13.40, 14.94, 22.74, 6.43]])
#ENS66_CI = np.array([[31.43, 14.97, 25.85, 13.67, 13.73, 19.89, 29.9, 6.93],
#                    [30.13, 14.2, 25.82, 11.13, 12.32, 16.29, 22.91, 6.46]])
#


plt.rcParams['figure.figsize'] = (12,3)

columns = ('Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'AUROC*100', 'AUPRC*100', '% Determinate')
rows = ['%s' % x for x in ('ENS2 (25.0%,45.0%)','ENS2 (46.5%,66.5%)')]

# Get some pastel shades for the colors
colors = ['red', 'orange']
n_cols = np.size(data,1)

index1 = np.linspace(0,1,np.size(data,1)+1)
index2 = []
for i in range(0, len(index1)-1):
    index2.append((index1[i] + index1[i+1])/2)
index2 = np.array(index2)

bar_width = 0.025

cell_text = []

plt.bar(index2 - bar_width*0.5, data[0,:], bar_width, color=colors[0], linewidth=1, edgecolor='black',  yerr=ENS45_CI, capsize=3) # APRI Performance Metrics
plt.bar(index2 + bar_width*0.5, data[1,:], bar_width, color=colors[1], linewidth=1, edgecolor='black',  yerr=ENS66_CI, capsize=3) # FIB4 Performance Metrics

cell_text.append(['%0.1f' % x for x in data[0,:]])
cell_text.append(['%0.1f' % x for x in data[1,:]])
#cell_text.append(['%0.1f' % x for x in data[2,:]])
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