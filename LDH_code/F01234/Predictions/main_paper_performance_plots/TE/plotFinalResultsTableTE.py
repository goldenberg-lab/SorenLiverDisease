import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.style as style
from matplotlib.ticker import NullFormatter
#style.use('seaborn-dark')

data = np.array([[92.3, 63.8, 67.9, 90.9, 76.7, 82.6, 72.3, 100],
                 [80.8, 63.8, 64.9, 80.0, 71.5, 77.3, 73.1, 100]])

# EXP_ENS3_p = ['< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001']


# ENS3_CI = np.array([[31.51, 16.16, 23.83, 13.55, 14.41, 20.31, 30.12, 6.79],
#                     [31.02, 15.56, 26.71, 11.45, 12.67, 16.48, 24.59, 6.60]])
# EXP_CI = np.array([[20.28, 13.15, 16.09, 6.80, 11.36, np.nan, np.nan, 0],
#                    [23.36, 12.14, 16.48, 6.70, 9.99, np.nan, np.nan, 0]])

#columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
#rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

plt.rcParams['figure.figsize'] = (12,5)

columns = ('Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy',  'AUROC*100', 'AUPRC*100', '% Dataset')
rows = ['%s' % x for x in ('TE(8.0, 8.0)', 'ENS(0.465)')]

# Get some pastel shades for the colors
colors = ['plum', 'red']
n_cols = np.size(data,1)

index1 = np.linspace(0,1,np.size(data,1)+1)
index2 = []
for i in range(0, len(index1)-1):
    index2.append((index1[i] + index1[i+1])/2)
index2 = np.array(index2)

index = np.linspace(0,1,8)
bar_width = 0.03

cell_text = []

plt.bar(index2 - 0.5*bar_width, data[0,:], bar_width, color=colors[0], linewidth=1, edgecolor='black') # FIB4 Performance Metrics
plt.bar(index2 + 0.5*bar_width, data[1,:], bar_width, color=colors[1], linewidth=1, edgecolor='black') # FIB4 Performance Metrics

cell_text.append(['%0.1f' % x for x in data[0,:]])
cell_text.append(['%0.1f' % x for x in data[1,:]])

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