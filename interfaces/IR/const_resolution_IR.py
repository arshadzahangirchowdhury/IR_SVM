#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: M Arshad Zahangir Chowdhury

keep resolution constant and assess performance at different features.

"""

#%matplotlib inline  
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  
import glob 
from scipy import signal
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
from sklearn.metrics import *
from math import *


import datetime
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC




import os, sys
if '../../' not in sys.path:
    sys.path.append('../../')
from src.misc.utils import *
from src.spectral_datasets.IR_datasets import IR_data
from random import seed, gauss

#1200-2200

s = IR_data(data_start = 1200, data_end = 2200, resolution=1, verbosity = False)
s.load_IR_data()
s.make_dataframe(s.spectra)
s.dataset_info()

# dropping compounds 
compounds = ['CO2', 'O3', 
             'CO',  'NO', 'SO2', 
               'HNO3', 'HF', 
             'HCl', 'HBr', 'HI', 'OCS', 
             'H2CO', 'HOCl', 'HCN',  
              'C2H2',  'PH3', 
             'H2S', 'HCOOH', 'C2H4', 'CH3OH', 
              'CH3CN', 'C4H2', 'HC3N', 
             'SO3', 'COCl2']
s.drop_compounds(compounds)
s.dataset_info()

X=s.spectra
y=s.targets
print('shape of features:', X.shape)
print('shape of labels:', y.shape)
print(s.label_id)

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

#OneVsRest (SVM-Linear Kernel)

classifier_OVR = OneVsRestClassifier(SVC(kernel='linear',C = 500,decision_function_shape = 'ovo',random_state=1)).fit(train_X, train_y)


pred_y = classifier_OVR.predict(test_X)



print("Test Accuracy:")

FCA_OVR_test_feat_1001_1200_2200, cm_OVR_test_feat_1001_1200_2200 = svm_clf_post_processor(pred_y, test_y, s.labels, 
                                         figure_save_file = 'RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/CM_Test_Res_1_1200-2200.png')



pred_y = classifier_OVR.predict(train_X)



print("Train Accuracy:")

FCA_OVR_train_feat_1001_1200_2200, cm_OVR_train_feat_1001_1200_2200 = svm_clf_post_processor(pred_y, train_y, s.labels, 
                                         figure_save_file = 'RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/CM_Train_Res_1_1200-2200.png')


# seed(1)
s.add_sinusoidal_noise()
Validation_X=s.val_sim_spectra

pred_y = classifier_OVR.predict(Validation_X)

print("Validation Accuracy]:")
FCA_OVR_valid_feat_1001_1200_2200, cm_OVR_valid_feat_1001_1200_2200 = svm_clf_post_processor(pred_y, y, s.labels, 
                                         figure_save_file = 'RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/CM_Valid_Res_1_1200-2200.png')






Y = label_binarize(y, classes=s.label_id)
n_classes = Y.shape[1]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=0)


classifier_OVR.fit(X_train, Y_train)


y_score = classifier_OVR.decision_function(X_test)

precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

#plot
fig = plt.figure(figsize=(1.6,0.9),dpi=300)
ax_def = fig.add_axes([0, 0, 1, 1])

plt.rc('font', weight='bold')
ax_def.step(recall['micro'], precision['micro'], where='post')

ax_def.set_title('Average precision score,\n micro-averaged over all classes:\n AP={0:0.2f}'
    .format(average_precision["micro"]),fontsize='medium', fontweight='bold')
# ax_def.legend(loc=2,prop={'size': 1})

# Set the axis limits
ax_def.set_xlim(0.0, 1.0)
ax_def.set_ylim(0.0, 1.05)

ax_def.xaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=8)
ax_def.xaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=8)
# ax.tick_params(direction='out', length=6, width=2, colors='r',
#                grid_color='r', grid_alpha=0.2)

ax_def.yaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=8)
ax_def.yaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=8)


ax_def.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax_def.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
# ax_H2O.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.20))
# ax_H2O.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.10))
# Add the x and y-axis labels
ax_def.set_ylabel(r'Precision', labelpad=4, fontsize='medium', fontweight='bold')
ax_def.set_xlabel(r'Recall', labelpad=4, fontsize='medium', fontweight='bold')

plt.show()

fig.savefig('RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/PR_Curve_Res_1_1200-2200.png', bbox_inches='tight',dpi=300)

RECALL_CASE4=recall['micro']
PRECISION_CASE4=precision['micro']

#1600-1700

s = IR_data(data_start = 1600, data_end = 1700, resolution=1, verbosity = False)
s.load_IR_data()
s.make_dataframe(s.spectra)
s.dataset_info()

# dropping compounds 
compounds = ['CO2', 'O3', 
             'CO',  'NO', 'SO2', 
               'HNO3', 'HF', 
             'HCl', 'HBr', 'HI', 'OCS', 
             'H2CO', 'HOCl', 'HCN',  
              'C2H2',  'PH3', 
             'H2S', 'HCOOH', 'C2H4', 'CH3OH', 
              'CH3CN', 'C4H2', 'HC3N', 
             'SO3', 'COCl2']
s.drop_compounds(compounds)
s.dataset_info()

X=s.spectra
y=s.targets
print('shape of features:', X.shape)
print('shape of labels:', y.shape)
print(s.label_id)

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

#OneVsRest (SVM-Linear Kernel)


classifier_OVR = OneVsRestClassifier(SVC(kernel='linear',C = 500,decision_function_shape = 'ovo',random_state=1)).fit(train_X, train_y)


pred_y = classifier_OVR.predict(test_X)



print("Test Accuracy:")
FCA_OVR_test_feat_101_1600_1700, cm_OVR_test_feat_101_1600_1700 = svm_clf_post_processor(pred_y, test_y, s.labels, 
                                         figure_save_file = 'RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/CM_Test_Res_1_1600-17000.png')

pred_y = classifier_OVR.predict(train_X)



print("Train Accuracy:")
FCA_OVR_train_feat_101_1600_1700, cm_OVR_train_feat_101_1600_1700 = svm_clf_post_processor(pred_y, train_y, s.labels, 
                                         figure_save_file = 'RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/CM_Train_Res_1_1600-17000.png')



# seed(1)
s.add_sinusoidal_noise()
Validation_X=s.val_sim_spectra

pred_y = classifier_OVR.predict(Validation_X)

print("Validation Accuracy]:")
FCA_OVR_valid_feat_101_1600_1700, cm_OVR_valid_feat_101_1600_1700 = svm_clf_post_processor(pred_y, y, s.labels, 
                                         figure_save_file = 'RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/CM_Valid_Res_1_1600-17000.png')


Y = label_binarize(y, classes=s.label_id)
n_classes = Y.shape[1]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=0)


classifier_OVR.fit(X_train, Y_train)


y_score = classifier_OVR.decision_function(X_test)

precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

#plot
fig = plt.figure(figsize=(1.6,0.9),dpi=300)
ax_def = fig.add_axes([0, 0, 1, 1])

plt.rc('font', weight='bold')
ax_def.step(recall['micro'], precision['micro'], where='post')

ax_def.set_title('Average precision score,\n micro-averaged over all classes:\n AP={0:0.2f}'
    .format(average_precision["micro"]),fontsize='medium', fontweight='bold')
# ax_def.legend(loc=2,prop={'size': 1})

# Set the axis limits
ax_def.set_xlim(0.0, 1.0)
ax_def.set_ylim(0.0, 1.05)

ax_def.xaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=8)
ax_def.xaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=8)
# ax.tick_params(direction='out', length=6, width=2, colors='r',
#                grid_color='r', grid_alpha=0.2)

ax_def.yaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=8)
ax_def.yaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=8)


ax_def.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax_def.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
# ax_H2O.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.20))
# ax_H2O.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.10))
# Add the x and y-axis labels
ax_def.set_ylabel(r'Precision', labelpad=4, fontsize='medium', fontweight='bold')
ax_def.set_xlabel(r'Recall', labelpad=4, fontsize='medium', fontweight='bold')

plt.show()

fig.savefig('RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/PR_Curve_Res_1_1600-1700.png', bbox_inches='tight',dpi=300)
RECALL_CASE5=recall['micro']
PRECISION_CASE5=precision['micro']

#1600-1610

s = IR_data(data_start = 1600, data_end = 1610, resolution=1, verbosity = False)
s.load_IR_data()
s.make_dataframe(s.spectra)
s.dataset_info()

# dropping compounds 
compounds = ['CO2', 'O3', 
             'CO',  'NO', 'SO2', 
               'HNO3', 'HF', 
             'HCl', 'HBr', 'HI', 'OCS', 
             'H2CO', 'HOCl', 'HCN',  
              'C2H2',  'PH3', 
             'H2S', 'HCOOH', 'C2H4', 'CH3OH', 
              'CH3CN', 'C4H2', 'HC3N', 
             'SO3', 'COCl2']
s.drop_compounds(compounds)
s.dataset_info()

X=s.spectra
y=s.targets
print('shape of features:', X.shape)
print('shape of labels:', y.shape)
print(s.label_id)

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

#OneVsRest (SVM-Linear Kernel)


classifier_OVR = OneVsRestClassifier(SVC(kernel='linear',C = 500,decision_function_shape = 'ovo',random_state=1)).fit(train_X, train_y)

pred_y = classifier_OVR.predict(test_X)

print("Test Accuracy:")
FCA_OVR_test_feat_11_1600_1610, cm_OVR_test_feat_11_1600_1610 = svm_clf_post_processor(pred_y, test_y, s.labels, 
                                         figure_save_file = 'RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/CM_Test_Res_1_1600-1610.png')


pred_y = classifier_OVR.predict(train_X)



print("Train Accuracy:")

FCA_OVR_train_feat_11_1600_1610, cm_OVR_train_feat_11_1600_1610 = svm_clf_post_processor(pred_y, train_y, s.labels, 
                                         figure_save_file = 'RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/CM_Train_Res_1_1600-1610.png')


# seed(1)
s.add_sinusoidal_noise()
Validation_X=s.val_sim_spectra

pred_y = classifier_OVR.predict(Validation_X)

print("Validation Accuracy]:")
FCA_OVR_valid_feat_11_1600_1610, cm_OVR_valid_feat_11_1600_1610 = svm_clf_post_processor(pred_y, y, s.labels, 
                                         figure_save_file = 'RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/CM_Valid_Res_1_1600-1610.png')



Y = label_binarize(y, classes=s.label_id)
n_classes = Y.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=0)

classifier_OVR.fit(X_train, Y_train)


y_score = classifier_OVR.decision_function(X_test)

precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

#plot
fig = plt.figure(figsize=(1.6,0.9),dpi=300)
ax_def = fig.add_axes([0, 0, 1, 1])

plt.rc('font', weight='bold')
ax_def.step(recall['micro'], precision['micro'], where='post')

ax_def.set_title('Average precision score,\n micro-averaged over all classes:\n AP={0:0.2f}'
    .format(average_precision["micro"]),fontsize='medium', fontweight='bold')
# ax_def.legend(loc=2,prop={'size': 1})

# Set the axis limits
ax_def.set_xlim(0.0, 1.0)
ax_def.set_ylim(0.0, 1.05)

ax_def.xaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=8)
ax_def.xaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=8)
# ax.tick_params(direction='out', length=6, width=2, colors='r',
#                grid_color='r', grid_alpha=0.2)

ax_def.yaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=8)
ax_def.yaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=8)


ax_def.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax_def.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
# ax_H2O.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.20))
# ax_H2O.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.10))
# Add the x and y-axis labels
ax_def.set_ylabel(r'Precision', labelpad=4, fontsize='medium', fontweight='bold')
ax_def.set_xlabel(r'Recall', labelpad=4, fontsize='medium', fontweight='bold')

plt.show()

fig.savefig('RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/PR_Curve_Res_1_1600-1610.png', bbox_inches='tight',dpi=300)


RECALL_CASE6=recall['micro']
PRECISION_CASE6=precision['micro']

Cases = ['$10^1$', '$10^2$', '$10^3$']

Train = [100*FCA_OVR_train_feat_11_1600_1610,100*FCA_OVR_train_feat_101_1600_1700,100*FCA_OVR_train_feat_1001_1200_2200]
Test = [100*FCA_OVR_test_feat_11_1600_1610,100*FCA_OVR_test_feat_101_1600_1700,100*FCA_OVR_test_feat_1001_1200_2200]
Validation = [100*FCA_OVR_valid_feat_11_1600_1610,100*FCA_OVR_valid_feat_101_1600_1700,100*FCA_OVR_valid_feat_1001_1200_2200]


x = np.arange(len(Cases))  # the label locations
width = 0.20 # the width of the bars


fig, ax = plt.subplots()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('3') 

fig.set_figheight(8)
fig.set_figwidth(8)

# fig = plt.figure(figsize=(1.6,1.6),dpi=300)
# ax_def = fig.add_axes([0, 0, 1, 1])
rects1 = ax.bar(x - width, Train, width, label='Train', color='red', edgecolor='black',linewidth=3)
rects2 = ax.bar(x , Test, width, label='Test', edgecolor='black',linewidth=3)
rects3 = ax.bar(x + width, Validation, width, label='Validation', color='olive', edgecolor='black',linewidth=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.legend(loc=2,prop={'size': 2})
ax.set_ylabel('Accuracy(%)',fontsize=18, fontweight='bold')
ax.set_xlabel('No. of features or variables',fontsize=18, fontweight='bold')
ax.set_title('Accuracy vs no. of features or variable'
    .format(average_precision["micro"]),fontsize=18, fontweight='bold')

ax.set_title('Same nine compounds\n classified in each group'
    .format(average_precision["micro"]),fontsize=18, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(Cases,fontsize=18, fontweight='bold')
ax.set_yticklabels(ax.get_yticks(),fontsize=18, fontweight='bold')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.xaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=18)
ax.xaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=18)
ax.yaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=18)
ax.yaxis.set_tick_params(which='minor', size=6, width=1, direction='out',labelsize=18)

# ax.legend(loc=2,prop={'size': 17})


# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)


# fig.tight_layout()

plt.show()


fig.savefig('RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/Accuracy_Compared.png', bbox_inches='tight',dpi=300)






#plot
fig = plt.figure(figsize=(1.6,0.9),dpi=300)
ax_def = fig.add_axes([0, 0, 1, 1])

plt.rc('font', weight='bold')
ax_def.step(RECALL_CASE6, PRECISION_CASE6, where='post', linewidth=1.0,label='1600-1610 $cm^{-1}$, 11 features')
ax_def.step(RECALL_CASE5, PRECISION_CASE5, where='post', linewidth=1.0,label='1600-1700 $cm^{-1}$, 101 features')
ax_def.step(RECALL_CASE4, PRECISION_CASE4, where='post', linewidth=1.0,label='1200-2200 $cm^{-1}$, 1001 features')



ax_def.legend(loc=3,prop={'size': 4})
ax_def.set_title('Same nine compounds classified in each group\n'
    .format(average_precision["micro"]),fontsize=4, fontweight='bold')
# Set the axis limits
ax_def.set_xlim(0.0, 1.0)
ax_def.set_ylim(0.0, 1.05)

ax_def.xaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=5)
ax_def.xaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=5)
# ax.tick_params(direction='out', length=6, width=2, colors='r',
#                grid_color='r', grid_alpha=0.2)

ax_def.yaxis.set_tick_params(which='major', size=6, width=1, direction='out',labelsize=5)
ax_def.yaxis.set_tick_params(which='minor', size=2, width=1, direction='out',labelsize=5)


ax_def.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax_def.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
ax_def.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
ax_def.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
# Add the x and y-axis labels
ax_def.set_ylabel(r'Precision', labelpad=4, fontsize=5, fontweight='bold')
ax_def.set_xlabel(r'Recall', labelpad=4, fontsize=5, fontweight='bold')

plt.show()

fig.savefig('RESULTS/Substudy_Figures/Study_Same_Compounds_Variable_Feature_Const_Res/PR_Curves_Compared.png', bbox_inches='tight',dpi=300)
