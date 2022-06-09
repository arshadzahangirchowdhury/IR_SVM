#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: M Arshad Zahangir Chowdhury

Performs hyperparameter tuning and stores the result in spreadsheet and plots
"""



import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix    #confusion matrix
import seaborn as sns  #heat map
import glob # batch processing of images
from scipy import signal
from sklearn.metrics import *
import matplotlib as mpl
from math import *
if '../../' not in sys.path:
    sys.path.append('../../')
from src.spectral_datasets.IR_datasets import IR_data
from src.misc.utils import *
from src.misc.tuning import hyperparameter_tuning
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier #Shift + tab will show detains of the classifier
from sklearn.svm import SVC
import datetime


s = IR_data(data_start = 1600, data_end = 1610, resolution=0.01, verbosity = False)
s.load_IR_data()
print('Number of Compounds:', s.n_compounds)
print('Number of Spectrum:', s.n_spectrum)
print('Total Number of Spectra:', s.n_spectra)
print("Front trim :", s.front_trim_amount)
print("End trim :", s.end_trim_amount)
print('Data Start Input:',s.data_start)
print('Data End Input:',s.data_end)           
print('Sample Size of training data:', s.samplesize )
print('Rows discarded:', s.n_discard_rows)
print('Resolution (1/cm) = ', s.resolution)

print('\n labels of molecules present \n', s.labels)
print('\n target indices (integers) of molecules present', s.targets)
print('\n frequencies present in the data \n ', s.frequencies)

compounds = ['CO2', 'O3', 'CO', 'NO', 'SO2', 
             'HNO3', 'HF', 'HCl', 'HBr', 'HI', 
             'OCS', 'H2CO', 'HOCl', 'HCN', 'C2H2', 
             'PH3', 'H2S', 'HCOOH', 'C2H4', 'CH3OH', 
             'CH3CN', 'C4H2', 'HC3N', 'SO3', 'COCl2']
s.drop_compounds(compounds)

X=s.spectra
y=s.targets
print('shape of features:', X.shape)
print('shape of labels:', y.shape)

#Start 70-30 training for final classifiers


#seeds used 100,237, 786
from sklearn.model_selection import train_test_split

TRAIN_SIZE=0.70
TEST_SIZE=1-TRAIN_SIZE

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=TRAIN_SIZE,
                                                   test_size=TEST_SIZE,
                                                   random_state=123,
                                                   stratify=y
                                                   )

print("All:", np.bincount(y) / float(len(y))*100  )
print("Training:", np.bincount(train_y) / float(len(train_y))*100  )
print("Testing:", np.bincount(test_y) / float(len(test_y))*100  )

#Save the training data to calculate cross-validation score later
Val_train_X=train_X
Val_train_y=train_y


# Tuning

parameters = {
    "estimator__C": [0.0001,0.001,0.01,0.1,1,10,100,500,1000]
#     "estimator__kernel": ["poly","rbf"],
#     "estimator__degree":[1, 2, 3, 4],
}


lin_test_score=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="linear")), 
                                     parameters, train_X, train_y )
rbf_test_score=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma='scale')), 
                                     parameters, train_X, train_y )

rbf_test_score_2=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma=0.000001)), 
                                     parameters, train_X, train_y )

rbf_test_score_3=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma=0.001)), 
                                     parameters, train_X, train_y )

rbf_test_score_4=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma=1)), 
                                     parameters, train_X, train_y )

rbf_test_score_5=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma=1000)), 
                                     parameters, train_X, train_y )


rbf_test_score_6=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma=1000000)), 
                                     parameters, train_X, train_y )



poly3_test_score=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="poly",gamma='scale')), 
                                     parameters, train_X, train_y )


tuningDF= pd.DataFrame()
tuningDF['C'] = parameters["estimator__C"]
tuningDF['linear'] = lin_test_score
tuningDF['RBF(gmma=scaled)'] = rbf_test_score
tuningDF['RBF(gmma=10e-6)'] = rbf_test_score_2
tuningDF['RBF(gmma=10e-3)'] = rbf_test_score_3
tuningDF['RBF(gmma=10e-0)'] = rbf_test_score_4
tuningDF['RBF(gmma=10e+3)'] = rbf_test_score_5
tuningDF['RBF(gmma=10e+6)'] = rbf_test_score_6
tuningDF['poly3(scaled)'] = poly3_test_score


tuningDF.to_csv('Hyperparameter_tuning_1600_1610.csv')



xparams=parameters["estimator__C"]

fig, ax_def = plt.subplots()
ax_def.patch.set_edgecolor('black')  
ax_def.patch.set_linewidth('3') 

fig.set_figheight(12)
fig.set_figwidth(12)

plt.rc('font', weight='bold')


ax_def.set_title('Frequency range : 1600-1610 $(cm^{-1})$\nResolution : 0.01 $(cm^{-1})$'
    ,fontsize=18, fontweight='bold')


# Set the axis limits
# ax_def.set_xlim(0.0, 1000)
ax_def.set_ylim(0.0, 1.05)
ax_def.set_xscale('log')
ax_def.xaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=18)
ax_def.xaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=18)
# ax.tick_params(direction='out', length=6, width=2, colors='r',
#                grid_color='r', grid_alpha=0.2)

ax_def.yaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=18)
ax_def.yaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=18)


# ax_def.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
# ax_def.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

# ax_H2O.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.20))
# ax_H2O.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.10))
# Add the x and y-axis labels
ax_def.scatter(xparams,lin_test_score,linewidth=3, label='Linear Kernel')
ax_def.scatter(xparams,rbf_test_score,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ Scaled)')
ax_def.scatter(xparams,rbf_test_score_2,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ = 0.000001)')
ax_def.scatter(xparams,rbf_test_score_3,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ = 0.001)')
ax_def.scatter(xparams,rbf_test_score_4,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ = 1)')
ax_def.scatter(xparams,rbf_test_score_5,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ = 1000)')
ax_def.scatter(xparams,rbf_test_score_6,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ = 1000000)')
ax_def.scatter(xparams,poly3_test_score,linewidth=3, marker = 's', label='3rd degree polynomial kernel ')
ax_def.set_ylabel(r'Score', labelpad=4, fontsize=18, fontweight='bold')
ax_def.set_xlabel(r'Soft margin constant, C', labelpad=4, fontsize=18, fontweight='bold')
ax_def.legend(loc=2,prop={'size': 12.5})
# plt.show()

fig.savefig('RESULTS/Kernel_C_Effect_1600_1610_9_compounds.png', bbox_inches='tight',dpi=300)



#400-4000 section

s = IR_data(data_start = 400, data_end = 4000, resolution=1, verbosity = False)
s.load_IR_data()
print('Number of Compounds:', s.n_compounds)
print('Number of Spectrum:', s.n_spectrum)
print('Total Number of Spectra:', s.n_spectra)
print("Front trim :", s.front_trim_amount)
print("End trim :", s.end_trim_amount)
print('Data Start Input:',s.data_start)
print('Data End Input:',s.data_end)           
print('Sample Size of training data:', s.samplesize )
print('Rows discarded:', s.n_discard_rows)
print('Resolution (1/cm) = ', s.resolution)

print('\n labels of molecules present \n', s.labels)
print('\n target indices (integers) of molecules present', s.targets)
print('\n frequencies present in the data \n ', s.frequencies)

X=s.spectra
y=s.targets
print('shape of features:', X.shape)
print('shape of labels:', y.shape)


TRAIN_SIZE=0.70
TEST_SIZE=1-TRAIN_SIZE

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=TRAIN_SIZE,
                                                   test_size=TEST_SIZE,
                                                   random_state=123,
                                                   stratify=y
                                                   )

print("All:", np.bincount(y) / float(len(y))*100  )
print("Training:", np.bincount(train_y) / float(len(train_y))*100  )
print("Testing:", np.bincount(test_y) / float(len(test_y))*100  )

#Save the training data to calculate cross-validation score later
Val_train_X=train_X
Val_train_y=train_y

# Tuning

parameters = {
    "estimator__C": [0.0001,0.001,0.01,0.1,1,10,100,500,1000]
#     "estimator__kernel": ["poly","rbf"],
#     "estimator__degree":[1, 2, 3, 4],
}


lin_test_score=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="linear")), 
                                     parameters, train_X, train_y )
rbf_test_score=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma='scale')), 
                                     parameters, train_X, train_y )

rbf_test_score_2=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma=0.000001)), 
                                     parameters, train_X, train_y )

rbf_test_score_3=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma=0.001)), 
                                     parameters, train_X, train_y )

rbf_test_score_4=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma=1)), 
                                     parameters, train_X, train_y )

rbf_test_score_5=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma=1000)), 
                                     parameters, train_X, train_y )


rbf_test_score_6=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="rbf",gamma=1000000)), 
                                     parameters, train_X, train_y )



poly3_test_score=hyperparameter_tuning(OneVsRestClassifier(SVC(kernel="poly",gamma='scale')), 
                                     parameters, train_X, train_y )


tuningDF= pd.DataFrame()
tuningDF['C'] = parameters["estimator__C"]
tuningDF['linear'] = lin_test_score
tuningDF['RBF(gmma=scaled)'] = rbf_test_score
tuningDF['RBF(gmma=10e-6)'] = rbf_test_score_2
tuningDF['RBF(gmma=10e-3)'] = rbf_test_score_3
tuningDF['RBF(gmma=10e-0)'] = rbf_test_score_4
tuningDF['RBF(gmma=10e+3)'] = rbf_test_score_5
tuningDF['RBF(gmma=10e+6)'] = rbf_test_score_6
tuningDF['poly3(scaled)'] = poly3_test_score

tuningDF.to_csv('Hyperparameter_tuning_400_4000.csv')

### 
xparams=parameters["estimator__C"]

fig, ax_def = plt.subplots()
ax_def.patch.set_edgecolor('black')  
ax_def.patch.set_linewidth('3') 

fig.set_figheight(8)
fig.set_figwidth(8)

plt.rc('font', weight='bold')


ax_def.set_title('Frequency range : 400-4000 $(cm^{-1})$\nResolution : 1 $(cm^{-1})$'
    ,fontsize=18, fontweight='bold')


# Set the axis limits
# ax_def.set_xlim(0.0, 1000)
# ax_def.set_ylim(0.0, 1.05)
ax_def.set_xscale('log')
ax_def.xaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=18)
ax_def.xaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=18)
# ax.tick_params(direction='out', length=6, width=2, colors='r',
#                grid_color='r', grid_alpha=0.2)

ax_def.yaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=18)
ax_def.yaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=18)


# ax_def.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
# ax_def.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))

# ax_H2O.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.20))
# ax_H2O.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.10))
# Add the x and y-axis labels
ax_def.scatter(xparams,lin_test_score,linewidth=3, label='Linear Kernel')
ax_def.scatter(xparams,rbf_test_score,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ Scaled)')
ax_def.scatter(xparams,rbf_test_score_2,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ = 0.000001)')
ax_def.scatter(xparams,rbf_test_score_3,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ = 0.001)')
ax_def.scatter(xparams,rbf_test_score_4,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ = 1)')
ax_def.scatter(xparams,rbf_test_score_5,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ = 1000)')
ax_def.scatter(xparams,rbf_test_score_6,linewidth=3, marker = 'v', label='RBF kernel ($\gamma$ = 1000000)')
ax_def.scatter(xparams,poly3_test_score,linewidth=3, marker = 's', label='3rd degree polynomial kernel ')
ax_def.set_ylabel(r'Score', labelpad=4, fontsize=18, fontweight='bold')
ax_def.set_xlabel(r'Soft margin constant, C', labelpad=4, fontsize=18, fontweight='bold')
ax_def.legend(loc=4,prop={'size': 8})
# plt.show()
fig.savefig('RESULTS/Kernel_C_Effect_400_4000_34_compounds.png', bbox_inches='tight',dpi=300)

