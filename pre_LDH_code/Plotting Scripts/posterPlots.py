import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

# Sensitivities
n_groups = 3
cross_val = (83.4, 64.1, 89.3)
cross_val_std = (7.8, 10.4, 5.7)
tor30 = (88.9 ,66.7, 91.7)
tor30exc = (100.0, 60.9, 86.21)
mcgill30 = (90.6, 70.5, 100.0)
mcgill30exc = (89.5, 60.0, 100.0)

plt.rcParams['figure.figsize'] = (11,10)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.1

opacity = 1
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, cross_val, bar_width,
                alpha=opacity, color='green', edgecolor='black',
                yerr=cross_val_std, error_kw=error_config,
                label='Cross Validation')
labels = ["%0.1f" % cross_val[i] for i in range(len(ax.patches))]
for rect, label in zip(ax.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


rects2 = ax.bar(index + bar_width, tor30, bar_width,
                alpha=opacity, color='lightblue', edgecolor='black',
                error_kw=error_config,
                label='Toronto HO')
labels = ["%0.1f" % tor30[i] for i in range(0,len(tor30))]
for rect, label in zip(rects2.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')

rects3 = ax.bar(index + 2*bar_width, tor30exc, bar_width,
                alpha=opacity, color='blue', edgecolor='black',
                error_kw=error_config,
                label='Toronto HO w/ Exclusions')
labels = ["%0.1f" % tor30exc[i] for i in range(0,len(tor30exc))]
for rect, label in zip(rects3.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


rects4 = ax.bar(index + 3*bar_width, mcgill30, bar_width,
                alpha=opacity, color='pink', edgecolor='black',
                error_kw=error_config,
                label='McGill HO')
labels = ["%0.1f" % mcgill30[i] for i in range(0,len(mcgill30))]
for rect, label in zip(rects4.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


rects5 = ax.bar(index + 4*bar_width, mcgill30exc, bar_width,
                alpha=opacity, color='red', edgecolor='black',
                error_kw=error_config,
                label='McGill HO w/ Exclusions')
labels = ["%0.1f" % mcgill30exc[i] for i in range(0,len(mcgill30exc))]
for rect, label in zip(rects5.patches, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')


ax.set_ylabel('Sensitivity (%)')
ax.set_title('Sensitivity')
ax.set_xticks(index + bar_width*5/2)
ax.set_xticklabels(('APRI', 'FIB4', 'ENS3'))
#ax.legend()
plt.grid(True)
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()


# Specificities
#n_groups = 3
#cross_val = (20, 35, 30 )
#cross_val_std = (2, 3, 4, )
#tor30 = (25, 32, 34)
#tor30exc = (25,15,25)
#mcgill30 = (25,32,34)
#mcgill30exc = (25,24,24)
#
#fig, ax = plt.subplots()
#
#index = np.arange(n_groups)
#bar_width = 0.35
#
#opacity = 0.4
#error_config = {'ecolor': '0.3'}
#
#rects1 = ax.bar(index, cross_val, bar_width,
#                alpha=opacity, color='b',
#                yerr=cross_val_std, error_kw=error_config,
#                label='Cross Validation')
#
#rects2 = ax.bar(index + bar_width, tor30, bar_width,
#                alpha=opacity, color='r',
#                error_kw=error_config,
#                label='Toronto Holdout')
#
#ax.set_ylabel('Scores')
#ax.set_title('Scores by group and gender')
#ax.set_xticks(index + bar_width / 2)
#ax.set_xticklabels(('Ensemble 3', 'APRI', 'FIB4'))
#ax.legend()
#
#fig.tight_layout()
#plt.show()
#
## AUROCS
#n_groups = 3
#cross_val = (20, 35, 30 )
#cross_val_std = (2, 3, 4, )
#tor30 = (25, 32, 34)
#tor30exc = (25,15,25)
#mcgill30 = (25,32,34)
#mcgill30exc = (25,24,24)
#
#fig, ax = plt.subplots()
#
#index = np.arange(n_groups)
#bar_width = 0.35
#
#opacity = 0.4
#error_config = {'ecolor': '0.3'}
#
#rects1 = ax.bar(index, cross_val, bar_width,
#                alpha=opacity, color='b',
#                yerr=cross_val_std, error_kw=error_config,
#                label='Cross Validation')
#
#rects2 = ax.bar(index + bar_width, tor30, bar_width,
#                alpha=opacity, color='r',
#                error_kw=error_config,
#                label='Toronto Holdout')
#
#ax.set_ylabel('Scores')
#ax.set_title('Scores by group and gender')
#ax.set_xticks(index + bar_width / 2)
#ax.set_xticklabels(('Ensemble 3', 'APRI', 'FIB4'))
#ax.legend()
#
#fig.tight_layout()
#plt.show()
#
#
## % Indetermiantes
#n_groups = 3
#cross_val = (20, 35, 30 )
#cross_val_std = (2, 3, 4, )
#tor30 = (25, 32, 34)
#tor30exc = (25,15,25)
#mcgill30 = (25,32,34)
#mcgill30exc = (25,24,24)
#
#fig, ax = plt.subplots()
#
#index = np.arange(n_groups)
#bar_width = 0.35
#
#opacity = 0.4
#error_config = {'ecolor': '0.3'}
#
#rects1 = ax.bar(index, cross_val, bar_width,
#                alpha=opacity, color='b',
#                yerr=cross_val_std, error_kw=error_config,
#                label='Cross Validation')
#
#rects2 = ax.bar(index + bar_width, tor30, bar_width,
#                alpha=opacity, color='r',
#                error_kw=error_config,
#                label='Toronto Holdout')
#
#ax.set_ylabel('Scores')
#ax.set_title('Scores by group and gender')
#ax.set_xticks(index + bar_width / 2)
#ax.set_xticklabels(('Ensemble 3', 'APRI', 'FIB4'))
#ax.legend()
#
#fig.tight_layout()
#plt.show()