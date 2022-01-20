import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.style as style
from matplotlib.ticker import NullFormatter
style.use('seaborn-dark')

data = np.array([[60.0, 66.0, 15.8, 93.9, 65.4, 72.1, 17.0, 78.8],
                 [100.0, 48.9, 22.6, 100.0, 55.6, 73.3, 21.8, 81.8],
                 [100.0, 40.7, 22.0, 100.0, 49.2, 69.4, 31.5, 95.5]])

APRI_ENS3_p = ['< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001', '< 0.001']
FIB4_ENS3_p = ['nan', '< 0.001', '< 0.001', 'nan', '< 0.001', '< 0.001', '< 0.001', '< 0.001']

APRI_CI = np.array([[59.35, 13.52, 16.06, 9.27, 12.9, 26.25, 15.89, 10.39],
                    [40.65, 12.99, 19.06, 6.11, 12.81, 20.56, 36.43, 9.30]])
FIB4_CI = np.array([[0, 14.77, 13.88, 0, 13.85, 16.29, 18.94, 10.33],
                    [0, 14.52, 15.37, 0, 12.74, 15.38, 32.95, 9.37]])
ENS3_CI = np.array([[0, 13.32, 11.8, 0, 12.86, 16.93, 22.51, 6.07],
                    [0, 12.99, 12.85, 0, 12.17, 14.54, 26.7, 4.54]])


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