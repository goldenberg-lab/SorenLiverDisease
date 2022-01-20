import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.style as style
from matplotlib.ticker import NullFormatter
#style.use('seaborn-dark')

data = np.array([[ 50.0, 72.4, 33.3, 84.0, 67.6, 74.1, 32.4,100-17.8],
                 [ 66.7, 69.2, 42.9, 85.7, 68.6, 71.2, 59.7,100-22.2],
                 [ 70.0, 75.0, 46.7, 88.9, 73.8, 76.2, 62.9, 100-6.7],
                 [ 54.5, 78.2, 45.9, 84.1, 72.4, np.nan, np.nan, 100],])

APRI_ENS3_p = ['< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001']
FIB4_ENS3_p = ['< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001']
EXP_ENS3_p = ['< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001']


APRI_CI = np.array([[36.2, 17.6, 24.65, 14.54, 14.98, 17.49, 21.31, 13.39],
                    [37.02, 15.46, 29.20, 12.04, 14.57, 15.14, 32.41, 11.06]])
FIB4_CI = np.array([[32.83, 19.14, 26.71, 17.55, 15.57, 24.77, 40.54, 13.29],
                    [33.84, 17.22, 27.91, 14.02, 14.25, 20.61, 26.94, 11.15]])
ENS3_CI = np.array([[31.51, 16.16, 23.83, 13.55, 14.41, 20.31, 30.12, 6.79],
                    [31.02, 15.56, 26.71, 11.45, 12.67, 16.48, 24.59, 6.60]])
EXP_CI = np.array([[20.28, 13.15, 16.09, 6.80, 11.36, np.nan, np.nan, 0],
                   [23.36, 12.14, 16.48, 6.70, 9.99, np.nan, np.nan, 0]])

#columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
#rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

plt.rcParams['figure.figsize'] = (12,6)

columns = ('Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy',  'AUROC*100', 'AUPRC*100', '% Determinate')
rows = ['%s' % x for x in ('APRI', 'FIB-4', 'ENS2', 'Expert Panel', 'APRI vs. ENS2 p-value', 'FIB-4 vs. ENS2 p-value', 'Expert vs. ENS2 p-value')]

# Get some pastel shades for the colors
colors = ['cyan', 'lightblue', 'red', 'mediumpurple', 'white', 'white', 'white']
n_cols = np.size(data,1)

index1 = np.linspace(0,1,np.size(data,1)+1)
index2 = []
for i in range(0, len(index1)-1):
    index2.append((index1[i] + index1[i+1])/2)
index2 = np.array(index2)

index = np.linspace(0,1,8)
bar_width = 0.02

cell_text = []

plt.bar(index2 - 1.5*bar_width, data[0,:], bar_width, color='cyan', linewidth=1, edgecolor='black', yerr=APRI_CI, capsize=3) # APRI Performance Metrics
plt.bar(index2 - 0.5*bar_width, data[1,:], bar_width, color='lightblue', linewidth=1, edgecolor='black', yerr=FIB4_CI, capsize=3) # FIB4 Performance Metrics
#plt.bar(index2, data[2,:], bar_width, color='orange', linewidth=1, edgecolor='black') # FIB4 Performance Metrics
plt.bar(index2 + 0.5*bar_width, data[2,:], bar_width, color='red', linewidth=1, edgecolor='black', yerr=ENS3_CI, capsize=3) # FIB4 Performance Metrics
plt.bar(index2 + 1.5*bar_width, data[3,:], bar_width, color='mediumpurple', linewidth=1, edgecolor='black', yerr=EXP_CI, capsize=3) # FIB4 Performance Metrics

cell_text.append(['%0.1f' % x for x in data[0,:]])
cell_text.append(['%0.1f' % x for x in data[1,:]])
cell_text.append(['%0.1f' % x for x in data[2,:]])
cell_text.append(['%0.1f' % x for x in data[3,:]])
cell_text.append(APRI_ENS3_p)
cell_text.append(FIB4_ENS3_p)
cell_text.append(EXP_ENS3_p)
#cell_text.append(['%0.1f' % x for x in data[4,:]])

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