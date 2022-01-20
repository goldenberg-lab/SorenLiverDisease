import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.style as style
from collections import namedtuple
style.use('seaborn-dark')


# Specificities
n_groups = 3
cross_val = (73.5, 92.2, 70.7)
cross_val_std = (8.8,4.9,9.7)
tor30 = (65.0, 82.1, 64.1)
tor30exc = (75.0, 93.3, 82.1)
mcgill30 = (50.4, 68.5, 47.1)
mcgill30exc = (76.6, 89.8, 72.0)

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
                alpha=opacity,  edgecolor='black',
                error_kw=error_config,
                label='Toronto HO w/ Exclusions')
labels = ["%0.1f" % tor30exc[i] for i in range(0,len(tor30exc))]
for rect, label in zip(rects3.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


rects4 = ax.bar(index + 3*bar_width, mcgill30, bar_width,
                alpha=opacity,  edgecolor='black',
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


ax.set_ylabel('Specificity (%)')
ax.set_title('Specificity')
ax.set_xticks(index + bar_width*5/2)
ax.set_xticklabels(('APRI', 'FIB4', 'ENS3'))
ax.minorticks_on()
#ax.legend()
plt.grid(True, which='both')
plt.ylim(45,100)
plt.minorticks_on()
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()
