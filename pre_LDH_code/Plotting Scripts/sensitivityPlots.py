import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.style as style
from collections import namedtuple
style.use('seaborn-dark')

n_groups = 3
var = 'auroc'

if (var == 'sens'):
    name = 'Sensitivity (%)'
    cross_val = (83.4, 64.1, 89.3)
    cross_val_std = (7.8, 10.4, 5.7)
    tor30 = (88.9 ,66.7, 91.7)
    tor30exc = (100.0, 60.9, 86.21)
    mcgill30 = (90.6, 70.5, 100.0)
    mcgill30exc = (89.5, 60.0, 100.0)
    ylow = 45
    yhigh = 105
elif (var == 'spec'):
    name = 'Specificity (%)'
    cross_val = (73.5, 92.2, 70.7)
    cross_val_std = (8.8,4.9,9.7)
    tor30 = (65.0, 82.1, 64.1)
    tor30exc = (75.0, 93.3, 82.1)
    mcgill30 = (50.4, 68.5, 47.1)
    mcgill30exc = (76.6, 89.8, 72.0)
    ylow = 45
    yhigh = 100
elif (var == 'auroc'):
    name = 'AUROC*100'
    cross_val = (79.2, 85.8, 89.5)
    cross_val_std = (6.7, 5.5, 3.7)
    tor30 = (71.7, 82.6,86.4)
    tor30exc = (81.1, 87.6, 87.1)
    mcgill30 = (64.6, 75.0, 79.3)
    mcgill30exc = (74.8, 83.5, 92.7)
    ylow = 60
    yhigh = 95
elif (var == 'indet'):
    name = '% Indeterminate'
    cross_val = (53.7, 29.38, 15.08)
    cross_val_std = (5.5,4.2,2.5)
    tor30 = (54.8, 27.9, 16.4)
    tor30exc = (52.2, 23.2, 17.4)
    mcgill30 = (51.5, 18.5, 14.2)
    mcgill30exc = (55.9, 16.0, 17.6)
    ylow = 0
    yhigh = 60

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.1
bar_space = 0.025

opacity = 0.9
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, cross_val, bar_width,
                alpha=opacity, edgecolor='black',
                yerr=cross_val_std, error_kw=error_config,
                label='Cross Validation')
labels = ["%0.1f" % cross_val[i] for i in range(len(ax.patches))]
for rect, label in zip(ax.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


rects2 = ax.bar(index + bar_width + bar_space, tor30, bar_width,
                alpha=opacity,  edgecolor='black',
                error_kw=error_config,
                label='Toronto HO')
labels = ["%0.1f" % tor30[i] for i in range(0,len(tor30))]
for rect, label in zip(rects2.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')

rects3 = ax.bar(index + 2*(bar_width + bar_space), tor30exc, bar_width,
                alpha=opacity,  edgecolor='black',
                error_kw=error_config,
                label='Toronto HO w/ Exclusions')
labels = ["%0.1f" % tor30exc[i] for i in range(0,len(tor30exc))]
for rect, label in zip(rects3.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


rects4 = ax.bar(index + 3*(bar_width + bar_space), mcgill30, bar_width,
                alpha=opacity, edgecolor='black',
                error_kw=error_config,
                label='McGill HO')
labels = ["%0.1f" % mcgill30[i] for i in range(0,len(mcgill30))]
for rect, label in zip(rects4.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


rects5 = ax.bar(index + 4*(bar_width + bar_space), mcgill30exc, bar_width,
                alpha=opacity, edgecolor='black',
                error_kw=error_config,
                label='McGill HO w/ Exclusions')
labels = ["%0.1f" % mcgill30exc[i] for i in range(0,len(mcgill30exc))]
for rect, label in zip(rects5.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')

ax.set_ylabel(name, fontsize=20)
plt.rcParams['figure.figsize'] = (10,7)
plt.rcParams.update({'font.size': 14.5})
ax.tick_params(axis='both', labelsize=20)

#ax.set_title('Sensitivity')
ax.set_xticks(index + bar_width*5/2)
ax.set_xticklabels(('APRI', 'FIB4', 'ENS3'))
ax.minorticks_on()
ax.legend()
plt.grid(True, which='both')
plt.ylim(ylow,yhigh)
plt.minorticks_on()
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()

