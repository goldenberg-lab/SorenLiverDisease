import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.style as style
from matplotlib.ticker import NullFormatter
style.use('seaborn-dark')

data = np.array([[ 34.5, 97.7, 76.9, 86.8, 86.0, 64.5, 46.1, 80.1],
                 [ 64.9, 91.2, 68.6, 89.8, 85.2, 86.8, 73.7, 82.7],
                 [ 82.4, 75.6, 56.0, 91.9, 77.4, 87.3, 76.7, 94.9]])

APRI_ENS3_p = ['< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001']
FIB4_ENS3_p = ['< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001']

APRI_CI = np.array([[16.4, 2.86, 27.4, 6.05, 5.74, 12.83, 19.45, 5.77],
                    [18.3, 2.31, 22.6, 5.18, 5.40, 12.3, 17.53, 4.94]])
FIB4_CI = np.array([[16.07, 5.14, 16.48, 5.57, 5.57, 8.82, 15.48, 5.59],
                    [15.30, 4.73, 56.05, 5.21, 4.90, 7.43, 11.21, 5.12]])
ENS3_CI = np.array([[11.4, 7.96, 12.46, 5.26, 5.75, 6.21, 12.13, 3.06],
                    [9.91, 6.93, 10.63, 4.77, 4.90, 5.20, 9.79, 3.06]])


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
    cell.set_fontsize(10)
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