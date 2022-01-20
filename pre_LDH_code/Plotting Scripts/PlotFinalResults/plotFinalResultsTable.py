import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.style as style
from matplotlib.ticker import NullFormatter
style.use('seaborn-dark')

data = np.array([[ 50.0, 72.4, 33.3, 84.0, 67.6, 74.1,    32.4, 17.8],
                 [ 66.7, 69.2, 42.9, 85.7, 68.6, 71.2,    59.7, 22.2],
                 [ 54.5, 78.2, 45.9, 84.1, 72.4, np.nan,  np.nan, 0],
                 [ 54.5, 79.4, 46.2, 84.4, 73.3, 75.9,    61.7,  0],
                 [ 70.0, 75.0, 46.7, 88.9, 73.8, 76.2,    62.9,  6.7]])

#columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
#rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

plt.rcParams['figure.figsize'] = (15,7.5)

columns = ('Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy',  'AUROC', 'AUPRC', '% Indeterminate')
rows = ['%s' % x for x in ('APRI', 'FIB4', 'Expert', 'ENS1', 'ENS3')]

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_cols = np.size(data,1)

index1 = np.linspace(0,1,9)
index2 = []
for i in range(0, len(index1)-1):
    index2.append((index1[i] + index1[i+1])/2)
index2 = np.array(index2)

index = np.linspace(0,1,8)
bar_width = 0.015

cell_text = []

plt.bar(index2 - bar_width*2, data[0,:], bar_width, color=colors[0], linewidth=1, edgecolor='black') # APRI Performance Metrics
plt.bar(index2 - bar_width*1, data[1,:], bar_width, color=colors[1], linewidth=1, edgecolor='black') # FIB4 Performance Metrics
plt.bar(index2, data[2,:], bar_width, color=colors[2], linewidth=1, edgecolor='black') # FIB4 Performance Metrics
plt.bar(index2 + bar_width*1, data[3,:], bar_width, color=colors[3], linewidth=1, edgecolor='black') # FIB4 Performance Metrics
plt.bar(index2 + bar_width*2, data[4,:], bar_width, color=colors[4], linewidth=1, edgecolor='black') # FIB4 Performance Metrics

cell_text.append(['%0.1f' % x for x in data[0,:]])
cell_text.append(['%0.1f' % x for x in data[1,:]])
cell_text.append(['%0.1f' % x for x in data[2,:]])
cell_text.append(['%0.1f' % x for x in data[3,:]])
cell_text.append(['%0.1f' % x for x in data[4,:]])

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
plt.grid(True)
plt.title('Expert Set Performance Metrics', fontsize=20)
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)

plt.show()