import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.style as style
from matplotlib.ticker import NullFormatter
style.use('seaborn-dark')

data = np.array([[ 38.2, 89.1, 40.6, 88.1, 80.9, 64.6, 33.8,100-20.2],
                 [ 70.5, 79.7, 47.0, 91.3, 77.8, 81.7, 51.5,100-17.6],
                 [ 85.0, 65.6, 44.0, 93.2, 70.3, 82.0, 53.8,100-5]])

APRI_ENS3_p = ['< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001']
FIB4_ENS3_p = ['< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '0.36', '< 0.001', '< 0.001']

APRI_CI = np.array([[17.61, 4.3 , 17.23, 5.26, 5.21, 10.72, 15.57, 4.86],
                    [18.47, 4.48, 17.77, 4.47, 5.04, 11.15, 15.40, 4.69]])
FIB4_CI = np.array([[13.85, 6.17, 12.25, 4.75, 5.44, 7.13, 15.21, 5.06],
                    [13.61, 5.76, 12.00, 4.23, 5.64, 6.96, 15.32, 4.48]])
ENS3_CI = np.array([[9.11, 6.54, 9.14, 4.39, 5.46, 5.63, 12.59, 2.65],
                    [8.00, 7.13, 9.47, 3.8, 5.75, 5.63, 13.50, 2.69]])


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