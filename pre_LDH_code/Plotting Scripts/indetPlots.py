import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.style as style
from collections import namedtuple
style.use('seaborn-dark')

# Specificities
n_groups = 3
cross_val = (53.7, 29.38, 15.08)
cross_val_std = (5.5,4.2,2.5)
tor30 = (54.8, 27.9, 16.4)
tor30exc = (52.2, 23.2, 17.4)
mcgill30 = (51.5, 18.5, 14.2)
mcgill30exc = (55.9, 16.0, 17.6)

plt.rcParams['figure.figsize'] = (20,14)

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


ax.set_ylabel('% Indeterminate')
ax.set_title('Percentage of Indeterminate Records')
ax.set_xticks(index + bar_width*5/2)
ax.set_xticklabels(('APRI', 'FIB4', 'ENS3'))
ax.minorticks_on()
ax.legend()
plt.legend(loc='under', bbox_to_anchor=(0, -0.2, 1., .102),
           ncol=5, mode="expand", borderaxespad=1., prop={'size': 20})

plt.grid(True, which='both')
plt.minorticks_on()
plt.ylim(0,60)
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()