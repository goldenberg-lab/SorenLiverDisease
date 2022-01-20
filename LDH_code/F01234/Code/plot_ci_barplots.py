import os
import shutil
import numpy as np
import pandas as pd 
import scipy.stats as st
import matplotlib.pyplot as plt 
import matplotlib.style as style
from matplotlib.font_manager import FontProperties

style.use('seaborn-dark')

dk = 'Toronto' # Toronto, Expert, McGill, NAFL, TE

ens_co = {'Toronto': {'APRI_det': 0.55, 'APRI_indet': 0.55, 'FIB4_det': 0.5, 'FIB4_indet': 0.5, 'all': 0.6},
        'McGill': {'APRI_det': 0.7525, 'APRI_indet': 0.7525, 'FIB4_det': 0.5875, 'FIB4_indet': 0.5875, 'all': 0.6},
        'TE': {'TE_det': 0.465, 'all': 0.6},
        'Expert': {'all': 0.6},
        'NAFL': {'NFS_det': 0.585, 'NFS_indet': 0.585, 'all': 0.6}}

# Order of metrics: sens, spec, ppv, npv, accuracy, auroc, auprc, % of dataset 
ENS = {'Toronto': {'APRI_det': {'means': [71.1, 88.1, 86.5, 74.0, 79.3, 87.6, 87.4, 83.7], 
                                'yerr_low': [14.8, 10.5, 11.6, 13.1, 9.2, 7.5, 9.6, 0.0], 
                                'yerr_high': [13.4, 9.1, 10.3, 12.4, 8.0, 6.7, 8.1, 0.0],
                                'p_val_labels': ['APRI vs. ENS p-value'],
                                'p_val_vals': [['<0.0001', '0.1799', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', 'N/A']],
                                'label': 'DET: ENS(0.55)',
                                'col': 'red'},
                   
                  'APRI_idt': {'means': [90.0, 57.1, 75.0, 80.0, 76.5, 84.3, 89.9, 16.3], 
                               'yerr_low': [24.0, 37.1, 25.2, 47.9, 18.0, 22.8, 20.8, 0.0], 
                               'yerr_high': [9.4, 42.9, 24.8, 18.8, 17.3, 15.4, 10.4, 0.0], 
                               'label': 'INDET: ENS(0.55)',
                               'col': 'salmon'},
                  
                  'FIB4_det': {'means': [75.0, 84.6, 81.8, 78.6, 80.0, 88.2, 88.1, 72.1], 
                               'yerr_low': [14.2, 12.3, 14.0, 13.6, 9.2, 8.2, 10.1, 0.0], 
                               'yerr_high': [14.4, 10.4, 12.1, 11.8, 8.2, 6.6, 7.3, 0.0],
                                'p_val_labels': ['FIB4 vs. ENS p-value'],
                                'p_val_vals': [['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', 'N/A']],
                               'label': 'DET: ENS(0.5)',
                               'col': 'red'},
                  
                  'FIB4_idt': {'means': [94.7, 60.0, 81.8, 85.7, 82.8, 81.1, 87.3, 27.9], 
                               'yerr_low': [11.7, 32.3, 17.0, 31.0, 14.1, 20.6, 16.7, 0.0], 
                               'yerr_high': [4.9, 29.3, 13.4, 13.6, 13.5, 16.3, 11.4, 0.0], 
                               'label': 'INDET: ENS(0.5)',
                               'col': 'salmon'},
                  
                  'ALL_EXP': {'means': [69.1, 83.7, 82.6, 70.7, 76.0, 87.0, 87.8, 100.0], 
                          'yerr_low': [12.3, 11.1, 12.1, 12.5, 8.0, 7.4, 9.5, 0.0], 
                          'yerr_high': [11.7, 8.6, 9.5, 10.6, 7.4, 6.0, 6.2, 0.0], 
                          'p_val_labels': ['APRI vs. ENS_EXP p-value', 
                                           'FIB4 vs. ENS_EXP p-value'],
                          'p_val_vals': [['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001'],
                                         ['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001']],
                          'label': 'ENS_EXP(0.6)', 
                          'col': 'red'}, 
                                    
                  'ALL_TE': {'means': [85.5, 73.5, 78.3, 81.8, 79.8, 87.0, 87.8, 100.0], 
                          'yerr_low': [10.1, 12.7, 11.2, 12.1, 7.9, 7.4, 9.5, 0.0], 
                          'yerr_high': [8.5, 11.9, 10.2, 10.2, 7.5, 6.0, 6.2, 0.0], 
                          'p_val_labels': ['APRI vs. ENS_TE p-value', 
                                           'FIB4 vs. ENS_TE p-value'],
                          'p_val_vals': [['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001'],
                                         ['<0.0001', '<0.0001', '0.0003', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001']],
                          'label': 'ENS_TE(0.465)', 
                          'col': 'orangered'}
                  },
       
       'McGill': {'APRI_det': {'means': [15.9, 91.7, 43.8, 72.9, 69.8, 66.4, 41.7, 75.5], 
                               'yerr_low': [7.6, 3.9, 17.3, 5.3, 5.3, 6.5, 9.2, 0.0], 
                               'yerr_high': [8.6, 3.3, 16.7, 5.2, 5.2, 6.7, 10.7, 0.0], 
                               'p_val_labels': ['APRI vs. ENS p-value'],
                               'p_val_vals': [['<0.0001', '0.0567', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', 'N/A']],
                               'label': 'DET: ENS(0.7525)',
                               'col': 'red'},
                  
                  'APRI_idt': {'means': [52.8, 71.7, 68.3, 56.9, 61.6, 74.4, 71.3, 24.5], 
                               'yerr_low': [13.9, 13.8, 15.3, 12.8, 9.1, 11.2, 13.8, 0.0], 
                               'yerr_high': [13.2, 13.6, 13.6, 11.9, 9.1, 9.6, 14.1, 0.0], 
                               'label': 'INDET: ENS(0.7525)', 
                               'col': 'salmon'},
                  
                  'FIB4_det': {'means': [58.5, 77.7, 53.9, 80.8, 71.8, 74.3, 55.5, 75.5], 
                               'yerr_low': [9.9, 5.7, 10.1, 5.2, 4.9, 6.1, 11.0, 0.0], 
                               'yerr_high': [9.7, 5.9, 10.3, 5.3, 4.6, 5.2, 10.6, 0.0], 
                               'p_val_labels': ['FIB4 vs. ENS p-value'],
                               'p_val_vals': [['<0.0001', '0.9338', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', 'N/A']],
                               'label': 'DET: ENS(0.5875)', 
                               'col': 'red'},
                  
                  'FIB4_idt': {'means': [68.1, 46.2, 53.3, 61.5, 56.6, 54.1, 48.9, 24.5], 
                               'yerr_low': [13.6, 14.0, 12.7, 14.2, 9.1, 10.5, 13.2, 0.0], 
                               'yerr_high': [13.8, 14.4, 13.3, 15.2, 9.1, 11.7, 14.7, 0.0], 
                               'label': 'INDET: ENS(0.5875)',
                               'col': 'salmon'},
                  
                  'ALL_EXP': {'means': [58.2, 72.6, 53.2, 76.4, 67.6, 71.6, 54.7, 100.0], 
                          'yerr_low': [8.5, 5.7, 7.6, 5.6, 4.6, 5.2, 8.9, 0.0], 
                          'yerr_high': [8.2, 5.2, 7.5, 5.2, 4.3, 4.9, 8.4, 0.0], 
                          'p_val_labels': ['APRI vs. ENS_EXP p-value', 
                                           'FIB4 vs. ENS_EXP p-value'],                         
                          'p_val_vals': [['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001'],
                                         ['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '0.9924', '<0.0001', '<0.0001']],
                          'label': 'ENS_EXP(0.6)', 
                          'col': 'red'}, 
                  
                  'ALL_TE': {'means': [79.4, 53.6, 47.9, 82.9, 62.6, 71.6, 54.7, 100.0], 
                          'yerr_low': [6.4, 6.3, 6.2, 6.3, 4.6, 5.2, 8.9, 0.0], 
                          'yerr_high': [6.5, 5.9, 6.1, 5.3, 4.6, 4.9, 8.4, 0.0], 
                          'p_val_labels': ['APRI vs. ENS_TE p-value', 
                                           'FIB4 vs. ENS_TE p-value'],  
                          'p_val_vals': [['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001'],
                                         ['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '0.9924', '<0.0001', '<0.0001']],
                          'label': 'ENS_TE(0.465)', 
                          'col': 'orangered'}                 
                  },
       
       'Expert': {'ALL_EXP': {'means': [63.6, 76.5, 46.7, 86.7, 73.3, 76.5, 61.5, 100.0], 
                          'yerr_low': [30.3, 14.6, 25.3, 13.4, 13.6, 16.9, 30.5, 0.0], 
                          'yerr_high': [28.1, 12.7, 26.6, 10.3, 10.9, 14.8, 23.1, 0.0],
                          'p_val_labels': ['EXP vs. ENS p-value'],
                          'p_val_vals': [['<0.0001', '0.4638', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', 'N/A']],
                          'label': 'ENS_EXP(0.6)', 
                          'col': 'red'}, 
                  
                  'ALL_TE': {'means': [81.8, 58.8, 39.1, 90.9, 64.4, 76.5, 61.5, 100.0], 
                          'yerr_low': [31.2, 16.4, 19.0, 13.0, 13.4, 16.9, 30.5, 0.0], 
                          'yerr_high': [18.2, 15.8, 20.3, 9.2, 13.2, 14.8, 23.1, 0.0],
                          'p_val_labels': ['EXP vs. ENS_TE p-value'],
                          'p_val_vals': [['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', 'N/A']],
                          'label': 'ENS_TE(0.465)', 
                          'col': 'orangered'}
                  },
       
       'NAFL': {'NFS_det': {'means': [50.0, 88.1, 67.3, 78.3, 75.6, 75.4, 68.1, 64.0], 
                             'yerr_low': [12.5, 5.3, 13.0, 6.2, 5.7, 7.9, 11.7, 0.0], 
                             'yerr_high': [12.3, 5.0, 12.7, 6.7, 5.6, 7.1, 10.0, 0.0],
                             'p_val_labels': ['NFS vs. ENS p-value'],
                             'p_val_vals': [['0.4836', '0.9258', '0.9342', '0.9822', '0.4719', '<0.0001', '<0.0001', 'N/A']],
                             'label': 'DET: ENS(0.585)', 
                             'col': 'red'},
                 
                 'NFS_idt': {'means': [72.7, 61.1, 69.6, 64.7, 67.5, 73.6, 77.8, 36.0], 
                             'yerr_low': [11.2, 12.9, 11.5, 13.6, 9.1, 9.2, 11.2, 0.0], 
                             'yerr_high': [11.1, 12.8, 10.1, 12.8, 7.6, 8.8, 9.1, 0.0], 
                             'label': 'INDET: ENS(0.585)',
                             'col': 'salmon'},
                 
                 'ALL_EXP': {'means': [57.4, 82.2, 69.0, 73.6, 72.1, 77.1, 72.8, 100.0],        # Result of ENS on all NAFL dataset
                         'yerr_low': [8.2, 5.7, 8.7, 6.0, 5.1, 5.0, 7.0, 0.0], 
                         'yerr_high': [7.9, 5.2, 8.3, 5.5, 4.5, 4.9, 7.0, 0.0],
                         'p_val_labels': ['NFS vs. ENS_EXP p-value'],
                         'p_val_vals': [['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001']],
                         'label': 'ENS_EXP(0.6)',
                         'col': 'red'},
                 
                 'ALL_TE': {'means': [77.9, 59.4, 57.0, 79.6, 67.0, 77.1, 72.8, 100.0],        # Result of ENS on all NAFL dataset
                         'yerr_low': [7.4, 6.9, 6.7, 6.8, 4.8, 5.0, 7.0, 0.0], 
                         'yerr_high': [6.6, 7.1, 6.7, 6.2, 4.8, 4.9, 7.0, 0.0],
                         'p_val_labels': ['NFS vs. ENS_TE p-value'],
                         'p_val_vals': [['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001']],
                         'label': 'ENS_TE(0.465)',
                         'col': 'orangered'},
                 },
       
       'TE': {'TE_det': {'means': [80.8, 63.8, 64.9, 80.0, 71.5, 77.3, 73.1, 100.0], 
                         'yerr_low': [9.6, 10.2, 8.8, 9.7, 6.8, 7.2, 10.1, 0.0], 
                         'yerr_high': [8.3, 9.6, 8.8, 9.2, 6.0, 6.3, 8.7, 0.0],
                         'p_val_labels': ['TE vs. ENS p-value'],
                         'p_val_vals': [['<0.0001', '0.4672', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '0.1885', 'N/A']],
                         'label': 'ENS(0.465)', 
                         'col': 'orangered'}, 
              
              'ALL_EXP': {'means': [53.8, 75.5, 64.6, 66.4, 65.7, 77.3, 73.1, 100.0], 
                      'yerr_low': [11.4, 9.2, 10.5, 9.3, 8.0, 7.2, 10.1, 0.0], 
                      'yerr_high': [11.5, 8.4, 11.7, 9.2, 7.1, 6.3, 8.7, 0.0], 
                      'p_val_labels': ['TE vs. ENS_EXP p-value'],
                      'p_val_vals': [['<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '0.1885', 'N/A']],
                      'label': 'ENS_EXP(0.6)', 
                      'col': 'red'},
              
              'ALL_TE': {'means': [80.8, 63.8, 64.9, 80.0, 71.5, 77.3, 73.1, 100.0], 
                      'yerr_low': [9.6, 10.2, 8.8, 9.7, 6.8, 7.2, 10.1, 0.0], 
                      'yerr_high': [8.3, 9.6, 8.8, 9.2, 6.0, 6.3, 8.7, 0.0], 
                      'p_val_labels': ['TE vs. ENS_TE p-value'],
                      'p_val_vals': [['<0.0001', '0.4672', '<0.0001', '<0.0001', '<0.0001', '<0.0001', '0.1885', 'N/A']],
                      'label': 'ENS_TE(0.465)', 
                      'col': 'orangered'}
              }}

APRI = {'Toronto': {'APRI_det': {'means': [37.8, 88.1, 77.3, 56.9, 62.1, 71.9, 64.3, 83.7], 
                    'yerr_low': [13.8, 9.9, 17.7, 12.6, 10.3, 10.9, 14.1, 0.0], 
                    'yerr_high': [14.7, 9.0, 17.3, 11.4, 9.2, 9.6, 15.3, 0.0], 
                    'label': 'DET: APRI(1,2)',
                    'col': 'cyan'},
                    
                    'ALL': {'means': [37.8, 88.1, 77.3, 56.9, 62.1, 71.9, 64.3, 83.7], 
                    'yerr_low': [14.7, 10.9, 18.9, 11.9, 10.2, 10.6, 14.6, 7.7], 
                    'yerr_high': [14.6, 9.4, 17.9, 11.3, 9.9, 10.3, 16.0, 6.8], 
                    'label': 'DET: APRI(1,2)',
                    'col': 'cyan'}},
        
        'McGill': {'APRI_det': {'means': [20.5, 91.7, 50.0, 74.0, 71.1, 61.8, 40.9, 75.5], 
                   'yerr_low': [7.7, 3.9, 16.6, 5.2, 4.9, 6.8, 9.5, 0.0], 
                   'yerr_high': [7.7, 3.6, 16.8, 5.0, 4.7, 6.9, 10.0, 0.0], 
                   'label': 'DET: APRI(1,2)', 
                   'col': 'cyan'},
                   
                   'ALL': {'means': [20.5, 91.7, 50.0, 74.0, 71.1, 61.8, 40.9, 75.5], 
                   'yerr_low': [7.8, 3.9, 15.9, 5.3, 5.3, 7.1, 9.3, 4.0], 
                   'yerr_high': [8.9, 3.7, 17.4, 5.1, 4.7, 7.1, 10.5, 3.7], 
                   'label': 'DET: APRI(1,2)', 
                   'col': 'cyan'}}}

FIB4 = {'Toronto': {'FIB4_det': {'means': [66.7, 82.1, 77.4, 72.7,74.7, 82.5, 82.2, 72.1], 
                    'yerr_low': [15.4, 12.5, 15.2, 13.4, 9.3, 9.8, 13.3, 0.0], 
                    'yerr_high': [16.0, 10.7, 13.7, 13.3, 9.4, 8.5, 10.2, 0.0], 
                    'label': 'DET: FIB4(1.45, 3.25)',
                    'col': 'lightblue'},
                    
                    'ALL': {'means': [66.7, 82.1, 77.4, 72.7,74.7, 82.5, 82.2, 72.1], 
                    'yerr_low': [15.3, 12.1, 15.7, 12.9, 10.0, 9.6, 14.0, 8.6], 
                    'yerr_high': [14.8, 10.7, 13.9, 12.8, 9.1, 8.6, 10.0, 8.8], 
                    'label': 'DET: FIB4(1.45, 3.25)',
                    'col': 'lightblue'},
                    }, 
        
        'McGill': {'FIB4_det': {'means': [53.2, 77.7, 51.5, 78.8, 70.2, 71.7, 52.2, 75.5], 
                   'yerr_low': [10.2, 5.7, 10.2, 5.4, 5.2, 5.5, 10.4, 0.0], 
                   'yerr_high': [9.8, 5.5, 10.0, 5.6, 5.0, 5.5, 10.0, 0.0], 
                   'label': 'DET: FIB4(1.45, 3.25)',
                   'col': 'lightblue'},
                   
                   
                   'ALL': {'means': [53.2, 77.7, 51.5, 78.8, 70.2, 71.7, 52.2, 75.5], 
                    'yerr_low': [10.4, 5.6, 10.5, 5.9, 5.4, 6.5, 10.4, 4.2], 
                    'yerr_high': [10.5, 5.2, 9.9, 5.6, 5.1, 5.9, 10.1, 4.2], 
                    'label': 'DET: FIB4(1.45, 3.25)',
                    'col': 'lightblue'},
                    }}

EXP = {'Expert': {'ALL': {'means': [54.5, 76.5, 42.9, 83.9, 71.1, 73.9, 49.2, 100.0], 
                  'yerr_low': [31.4, 15.0, 24.8, 13.0, 13.5, 18.2, 28.8, 0.0], 
                  'yerr_high': [30.2, 13.5, 29.7, 12.7, 13.1, 16.4, 29.3, 0.0], 
                  'label': 'EXP(0.5)', 
                  'col': 'lightgreen'}}}

NFS = {'NAFL': {'NFS_det': {'means': [50.0, 88.1, 67.3, 78.3, 75.6, 70.7, 55.1, 64.0], 
                 'yerr_low': [12.1 ,5.3, 12.5, 6.2, 5.7, 8.6, 12.0, 0.0], 
                 'yerr_high': [11.8, 5.1, 12.3, 6.5, 5.6, 8.3, 12.9, 0.0], 
                 'label': 'DET: NFS(-1.455,0.675)',
                 'col': 'yellow'},
                
                'ALL': {'means': [50.0, 88.1, 67.3, 78.3, 75.6, 70.7, 55.1, 64.0], 
                 'yerr_low': [12.2, 5.5, 13.0, 6.5, 5.9, 8.9, 12.1, 5.0], 
                 'yerr_high': [11.5, 5.3, 12.4, 6.3, 6.1, 7.6, 12.3, 4.7], 
                 'label': 'DET: NFS(-1.455,0.675)',
                 'col': 'yellow'},
                }}

TE = {'TE': {'TE_det': {'means': [92.3, 63.8, 67.9, 90.9, 76.7, 82.6, 72.3, 100.0], 
             'yerr_low': [6.6, 9.8, 8.7, 7.9, 6.5, 6.8, 11.6, 0.0], 
             'yerr_high': [5.0, 9.5, 8.5, 6.2, 6.3, 6.3, 11.7, 0.0], 
             'label': 'TE(8)',
             'col': 'plum'},
             
             'ALL': {'means': [92.3, 63.8, 67.9, 90.9, 76.7, 82.6, 72.3, 100.0], 
             'yerr_low': [6.6, 9.8, 8.7, 7.9, 6.5, 6.8, 11.6, 0.0], 
             'yerr_high': [5.0, 9.5, 8.5, 6.2, 6.3, 6.3, 11.7, 0.0], 
             'label': 'TE(8)',
             'col': 'plum'}}}

def plot_bargraph(alg_list):
    num_algs = len(alg_list)
    mets = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', '100*AUROC', '100*AUPRC', '% of Dataset']
   
    index1 = np.linspace(0,1,9)
    index2 = []
    for i in range(0, len(index1)-1):
        index2.append((index1[i] + index1[i+1])/2)
    index2 = np.array(index2)
    

    if (num_algs == 2):
        bar_width = 0.05
        bwa = [-0.025, 0.025]
        plt.figure(figsize=(12,2.5))
    elif (num_algs == 3):
        bar_width = 0.025
        bwa = [-0.025, 0, 0.025]
        plt.figure(figsize=(12,2.5))
    elif (num_algs == 4):
        bar_width = 0.02
        bwa = [-0.03, -0.01, 0.01, 0.03]
        plt.figure(figsize=(12,3))    
    
    # Step 1. Set appropriate style 
    cell_text = []
    rows = [l['label'] for l in alg_list]
    colors = [alg['col'] for alg in alg_list]
    
    # add releant p-value labels 
    for c, alg in enumerate(alg_list): 
        if 'p_val_labels' in alg.keys(): 
            for lab in alg['p_val_labels']:
                rows.append(lab)
                colors.append('white')

    print(rows)    

    for c, alg in enumerate(alg_list): 
        yerr = [alg['yerr_low'], alg['yerr_high']]
        plt.bar(index2 + bwa[c], alg['means'], bar_width, linewidth=1, color=alg['col'], yerr=yerr, edgecolor='black', capsize=3)
        cell_text.append(['%0.1f' % (alg['means'][m]) for m, met in enumerate(mets)])
        
    # Add p-value strings Do this later. 
    for c, alg in enumerate(alg_list):
        if ('p_val_labels' in alg.keys()):
            for d, val in enumerate(alg['p_val_labels']): 
                cell_text.append(['%s' % s for s in alg['p_val_vals'][d]])
    
    ts_x =0
    te_x = 1-ts_x
    
    ts_y = -0.75
    te_y = 0.75
        
    the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=tuple(mets),
                      cellLoc='center',
                      bbox = [ts_x,ts_y,te_x,te_y], # (left-x, bottom-y, length-x, length-y)
                      loc='bottom')
    the_table.auto_set_font_size(False)
    
    if (num_algs <= 3):
        the_table.set_fontsize(12)
    else:
        the_table.set_fontsize(10.5)
        
    for (row, col), cell in the_table.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    plt.ylabel('Performance (%)', fontsize=15)
    plt.ylim([0,100])
    plt.xlim([0,1])
    plt.grid(True, axis='y', color='gray', linestyle='--')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.set_facecolor('white')


if dk == 'Toronto' or dk == 'McGill':
    
    # Case 1&2: APRI Determinates & Indeterminates
    algs1 = [APRI[dk]['APRI_det'], ENS[dk]['APRI_det'], ENS[dk]['APRI_idt']]
    
    # Case 3&4: FIB4 Determinates & Indeterminates 
    algs2 = [FIB4[dk]['FIB4_det'], ENS[dk]['FIB4_det'], ENS[dk]['FIB4_idt']]
    
    # Case 5: APRI vs. FIB4 vs. ENS2 @ 60% 
    algs3 = [APRI[dk]['ALL'], FIB4[dk]['ALL'], ENS[dk]['ALL_TE'], ENS[dk]['ALL_EXP']]
    
    plot_bargraph(algs1)
    plot_bargraph(algs2)
    plot_bargraph(algs3)
    
elif dk == 'Expert':
    algs1 = [EXP[dk]['ALL'], ENS[dk]['ALL_EXP']]
    algs2 = [EXP[dk]['ALL'], ENS[dk]['ALL_TE'], ENS[dk]['ALL_EXP']]
    
    plot_bargraph(algs1)
    plot_bargraph(algs2)
    
elif dk == 'NAFL':
    algs1 = [NFS[dk]['NFS_det'], ENS[dk]['NFS_det'], ENS[dk]['NFS_idt']]
    algs2 = [NFS[dk]['ALL'], ENS[dk]['ALL_TE'], ENS[dk]['ALL_EXP']]
    
    plot_bargraph(algs1)
    plot_bargraph(algs2)
    
elif dk == 'TE':
    algs1 = [TE[dk]['TE_det'], ENS[dk]['TE_det']]
    algs2 = [TE[dk]['ALL'], ENS[dk]['ALL_TE'], ENS[dk]['ALL_EXP']]
    plot_bargraph(algs1)
    plot_bargraph(algs2)
    