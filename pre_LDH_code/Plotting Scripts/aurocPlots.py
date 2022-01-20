import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.style as style
from collections import namedtuple
style.use('seaborn-dark')

class alg:
    def __init__(self):
        self.name = ""
        self.sens = []
        self. 
        self.
        self.
        self.
        self.
        self.
        self.

# Specificities
n_groups = 3
cross_val = (79.2, 85.8, 89.5)
cross_val_std = (6.7, 5.5, 3.7)
tor30 = (71.7, 82.6,86.4)
tor30exc = (81.1, 87.6, 87.1)
mcgill30 = (64.6, 75.0, 79.3)
mcgill30exc = (74.8, 83.5, 92.7)

plt.rcParams['figure.figsize'] = (10,7)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.1

opacity = 1
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


rects2 = ax.bar(index + bar_width, tor30, bar_width,
                alpha=opacity, edgecolor='black',
                error_kw=error_config,
                label='Toronto HO')
labels = ["%0.1f" % tor30[i] for i in range(0,len(tor30))]
for rect, label in zip(rects2.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')

rects3 = ax.bar(index + 2*bar_width, tor30exc, bar_width,
                alpha=opacity, edgecolor='black',
                error_kw=error_config,
                label='Toronto HO w/ Exclusions')
labels = ["%0.1f" % tor30exc[i] for i in range(0,len(tor30exc))]
for rect, label in zip(rects3.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


rects4 = ax.bar(index + 3*bar_width, mcgill30, bar_width,
                alpha=opacity, edgecolor='black',
                error_kw=error_config,
                label='McGill HO')
labels = ["%0.1f" % mcgill30[i] for i in range(0,len(mcgill30))]
for rect, label in zip(rects4.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


rects5 = ax.bar(index + 4*bar_width, mcgill30exc, bar_width,
                alpha=opacity, edgecolor='black',
                error_kw=error_config,
                label='McGill HO w/ Exclusions')
labels = ["%0.1f" % mcgill30exc[i] for i in range(0,len(mcgill30exc))]
for rect, label in zip(rects5.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


ax.set_ylabel('AUROC*100 (%)')
ax.set_title('AUROC')
ax.set_xticks(index + bar_width*5/2)
ax.set_xticklabels(('APRI', 'FIB4', 'ENS3'))
ax.minorticks_on()
#ax.legend()
plt.grid(True, which='both')
plt.ylim(60,95)
plt.minorticks_on()
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()