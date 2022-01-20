import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.style as style
from collections import namedtuple
style.use('seaborn-dark')

def set_ax(ax, title, case):
    ax.set_ylabel('Performance (%)')
    ax.set_xlabel('Metric')
    ax.set_title(title, fontsize=20)
    if (case==1):
        ax.set_xticks([0,1,2,3])
        ax.set_xticks([0,1,2,3,4])
    elif (case ==2):
        ax.set_xticks([0.25,1.25,2.25,3.25])
        ax.set_xticks([0.25,1.25,2.25,3.25,4.25])
    ax.set_xticklabels(('APRI', 'FIB4', 'Experts', 'ENS1\n66.5%', 'ENS3\n66.5%'))
    ax.minorticks_on()
    ax.grid(True)
    ax.set_ylim([0,100])
    
def plot_stuff(perfs, ax):
    index = np.arange(n_groups)
    bar_width = 0.5
    opacity = 1
    
    rects1 = ax.bar(index, perfs, bar_width,
                    alpha=opacity, linewidth=3, edgecolor='black', color='lightblue')
    labels = ["%0.1f" % perfs[i] for i in range(len(ax.patches))]
    #patterns = ['/','||','\\','-']    
    
    count = 0
    for rect, label in zip(ax.patches, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label,
                ha='center', va='bottom', fontsize=12.5)
        #rect.set_hatch(patterns[count])
        count += 1
        
def plot_stuff_2(perfs1, perfs2, ax):
    index = np.arange(n_groups)
    bar_width = 0.4
    opacity = 1

    rects1 = ax.bar(index, perfs1, bar_width,
                    alpha=opacity, linewidth=3, edgecolor='black', color='lightgreen', hatch='/',
                    label='')
    labels = ["%0.1f" % perfs1[i] for i in range(len(ax.patches))]
    for rect, label in zip(ax.patches, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label,
                ha='center', va='bottom')
    
    rects2 = ax.bar(index + bar_width, perfs2, bar_width,
                    alpha=opacity, linewidth=3, edgecolor='black', color='lightblue', hatch='\\',
                    label='')
    labels = ["%0.1f" % perfs2[i] for i in range(0,len(perfs2))]
    for rect, label in zip(rects2.patches, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, label,
                ha='center', va='bottom')

# Toronto Performance Metrics
n_groups = 5 # Sensitivity, PPV, Specificity, NPV
#SENS = (37.8, 66.7, 78.2, 82.7)
#SPEC = (88.1, 82.1, 75.5, 71.7)
#PPV = (77.3, 77.4, 78.2, 76.8)
#NPV = (56.9, 72.7, 75.5, 78.6)
#ACC = (62.1, 74.7, 76.9, 77.6)
#AUROC = (71.9, 82.7, 83.9, 84.8)
#AUPRC = (64.3, 78.1, 86.0, 86.8)
#INDET = (16.4, 27.9, 0, 5.8)


#APRI_perfs = (37.8, 88.1, 77.3, 56.9)
#FIB4_perfs = (66.7, 82.1, 77.4, 72.7)
#ENS3_30 = (88.5, 56.8, 70.8, 80.6)
#ENS3_50 = (81.1, 73.9, 78.2, 77.3)
#ENS3_70 = (70.2, 86.7, 84.6, 73.6)

# Montreal Performance Metrics 
    # APRI, FIB4, ENS1, ENS3
#SENS = (38.2, 70.5, 81.0, 85.0)
#SPEC = (89.1, 80.0, 68.8, 65.6)
#PPV = (40.6, 47.0, 45.1, 44.0)
#NPV = (88.1, 91.3, 92.0, 93.2)
#ACC = (80.9, 77.8, 71.8, 70.3)
#AUROC = (64.6, 81.7, 81.3, 82.0)
#AUPRC = (33.8, 51.5, 54.6, 53.9)
#INDET = (20.2, 17.6, 0, 5.0)
#
## Expert Performance Metrics 
## Apri, FIB4, 5 Experts, ENS1, ENS3
SENS = (50.0, 66.7, 54.6, 54.5, 70.0)
SPEC = (72.4, 69.2, 78.2, 79.4, 75.0)
PPV = (33.3, 42.9, 45.9, 46.2, 46.7)
NPV = (84.0, 85.7, 84.4, 88.9, 84.1)
ACC = (67.6, 68.6, 73.3, 73.8, 72.4)
AUROC = (74.1, 71.2, np.nan, 75.9, 76.2)
AUPRC = (32.4, 59.7, np.nan, 61.7, 62.9)
INDET = (17.8, 22.2, 0, 0, 6.7)

plt.rcParams['figure.figsize'] = (20,4)
fig, ((ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(1,6, sharex=False, sharey=False)

plot_stuff(SENS, ax1)
set_ax(ax1, 'Sensitivity',1)

plot_stuff(SPEC, ax2)
set_ax(ax2, 'Specificity',1)

plot_stuff(PPV, ax3)
set_ax(ax3, 'PPV',1)

plot_stuff(NPV, ax4)
set_ax(ax4, 'NPV',1)

plot_stuff(ACC, ax5)
set_ax(ax5, 'Accuracy',1)

plot_stuff(INDET, ax6)
set_ax(ax6, '% Indeterminate', 1)

#plot_stuff(AUROC, ax7)
#set_ax(ax7, 'AUROC',1)
#
#plot_stuff(AUPRC, ax8)
#set_ax(ax8, 'AUPRC', 1)


#plot_stuff(ENS3_3, ax3)
#set_ax(ax3, 'ENS3 @ 30%')
#
#plot_stuff(ENS3_50, ax4)
#set_ax(ax4, 'ENS3 @ 50%')
#
#plot_stuff(ENS3_70, ax5)
#set_ax(ax5, 'ENS3 @ 70%')

plt.minorticks_on()
plt.subplots_adjust(right=0.75, left=0)
fig.tight_layout()
plt.show()

