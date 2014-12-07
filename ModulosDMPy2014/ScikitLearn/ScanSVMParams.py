#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
=====================================
POLYNOMIAL AND RBF SVM AD_HOC SCAN PARAMS
=====================================

@author: spolex

@version: 0.1

Created on 28/11/2014

This code optimizes the parameters `gamma`,`C` and
degree of the polynomial kernel SVM.You can optimize 
rbf too using degree parameter.

The ad-hoc scan is used for this purpose to compare with 
the search in grid implemented in api, and and to illustrate
the results of a scan ad hoc, and to illustrate the results of 
a scan enable ad hoc addition to studying the ranges in which 
varying gamma c.

The parameters are readjusted in evaluation by F1-score to positive class. 

@precondition: The 'class' must be in last position.
                The 'the positive label' must be in first position.

@param train: Dataset for training the svm function. 
@param dev: Dataset for hold out evaluation.
@param kernel: kernel will be used. Defaul 'rbf'. Posible 'poly' 
                in this first version.

@return: At the end of the scan according to the evolution of
kernel parameters are displayed and the optimal classifier is
serialized.


'''
import arff
import sys
import numpy as np
import os
import time
#from scipy.io.arff.arffread import MetaData
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer#, FeatureHasher
#from sklearn.tree.tree import ExtraTreeClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn import cross_validation as cs
from sklearn.externals import joblib as jl 
from sklearn.metrics.scorer import make_scorer
import matplotlib.pyplot as plt




class Main:
    num = 0
    
def main(self,argv=sys.argv):
#######   
    try:
        kernel =sys.argv[3] 
    except :# catch *all* exceptions
        kernel='rbf'
        
    print('Training data loading....')
    data = arff.load(open(argv[1],'rb'))
    labeled_set = data['data']
    train_set = [fila[0:len(fila)-1] for fila in labeled_set]
    
    atts = data['attributes']
    atts_names = [fila[0] for fila in atts]
    att_values = [fila [1] for fila in atts]
    labels = np.array(att_values[len(att_values)-1])
    
    print 'TRAIN DATA SAMPLES'
    print len(train_set)
    print 'Attributes NUM'
    print len(atts_names)
    print 'LABELS FOR POSITIVE CLASS'
    print labels[0]
    
    print('Develop data loading....')
    datadev_set = arff.load(open(argv[2],'rb'))
    dev_labeled_set = datadev_set['data']
    dev_set = [fila[0:len(fila)-1] for fila in dev_labeled_set]
    
    dev_atts = data['attributes']
    dev_atts_names = [fila[0] for fila in dev_atts]
    dev_att_values = [fila [1] for fila in dev_atts]
    dev_labels = np.array(dev_att_values[len(dev_att_values)-1])
    
    print 'DEV DATA SAMPLES'
    print len(dev_set)
    print 'DEV Attributes NUM'
    print len(dev_atts_names)
    print 'LABELS FOR POSITIVE DEV CLASS'
    print dev_labels[0]
    
####    
    print ('Preprocesing data...')
    # #    parse a un dict para poder vectorizar los att categoricos
    print ('Parsing categorical data...')

    dict_list = []
    N,F = len(train_set),len(train_set[0])
    for n in range(N):
        d = {}
        for f in range(F):
            feature = atts_names[f]
            d[feature] = train_set[n][f]
        dict_list.append(d)
        
    dev_dict_list = []
    N,F = len(dev_set),len(dev_set[0])
    for n in range(N):
        d = {}
        for f in range(F):
            feature = dev_atts_names[f]
            d[feature] = dev_set[n][f]
        dev_dict_list.append(d)
    
    
    #Fit vectorizer for each dict
    v = DictVectorizer(sparse=False,dtype=np.float64)
    
    v_train_set = v.fit_transform(dict_list[0])
    for i in range(1,len(dict_list)):
        train_set_instance = v.fit_transform(dict_list[i])
        v_train_set = np.vstack((v_train_set,train_set_instance))
    
    v_dev_set = v.fit_transform(dev_dict_list[0])
    for j in range(1,len(dev_dict_list)):   
        v_dev_set_instance = v.fit_transform(dev_dict_list[j])        
        v_dev_set = np.vstack((v_dev_set,v_dev_set_instance))
       
    v_train_set = np.asarray(v_train_set)
    v_dev_set = np.asarray(v_dev_set)
            
    # # transform non-numerical labels to numerical
    # # transform non-numerical labels to numerical
    train_set_labels=[]
    for fila in labeled_set:
        train_set_labels.append(fila[-1])
    le = preprocessing.LabelEncoder()    
    le.fit(train_set_labels)
    
    dev_set_labels=[]
    for fila in dev_labeled_set:
        dev_set_labels.append(fila[-1])
    le = preprocessing.LabelEncoder()    
    le.fit(dev_set_labels)
    
    
    #dataset for decision function visualization
    train_numeric_labels=le.transform(train_set_labels);
    X_2d = v_train_set[:, -2:]
    Y_2d = train_numeric_labels
    
    dev_numeric_labels=le.transform(dev_set_labels)
    X_dev_2d = v_dev_set[:, -2:]
    Y_dev_2d = dev_numeric_labels
    
    # It is usually a good idea to scale the data for SVM training.
    # We are cheating a bit in this example in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.
    scaler = preprocessing.StandardScaler()
    v_train_set = scaler.fit_transform(v_train_set)
    v_dev_set = scaler.fit_transform(v_dev_set)
    X_2d = scaler.fit_transform(X_2d)
    X_dev_2d = scaler.fit_transform(X_dev_2d)

#########    
#Training the models
##########
    cBest = 0.
    gBest = 0.  
    dBest = 0.  
    print('Start scaning data for %s kernel....' %(kernel))
    f1Aux=0.0
    f1Best=0.0
    if kernel=='rbf':
        maxD=4
    else:
        maxD=5
    #
    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-3, 3)
    degree_range = np.arange(3,maxD)

    for d in degree_range:#2,5
        for c in C_range:#-15,12
            for g in gamma_range:#-3,5
                print("Hyperparameters: C = %r gamma = %r degree = %d...." %(c,g,d))
                #   fit the model 
             
                model = svm.SVC(kernel=kernel, gamma=g, C=c, degree=d)
                model.fit(v_train_set, train_numeric_labels)  
                #     make predictions
                print "Making predictions..."
                expected=dev_numeric_labels
                predicted = model.predict(v_dev_set)
                print "Making Hold out evaluation with dev set..."
                f1Aux = metrics.f1_score(expected, predicted, pos_label=0)
                print ("New F1Score = %r" %f1Aux)
                if f1Aux>f1Best:
                    print ("Maximun F1Score = %r" %f1Aux)
                    f1Best=f1Aux    
                    print('Hyperparameters has been changed New degree = %d New C= %r New gamma = %r ' %(d,c,g))
                    cBest = c
                    gBest = g  
                    dBest = d 
   # Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it  takes a while to train)
    C_2d_range = [1, 1e2, 1e4]
    gamma_2d_range = [1e-1, 1, 1e1]
    degree_range = [2,3]
    classifiers = []
    for D in degree_range:
        for C in C_2d_range:
            for gamma in gamma_2d_range:
                clf = svm.SVC(C=C, gamma=gamma, degree=D)
                clf.fit(X_2d, Y_2d)
                classifiers.append((C, gamma, clf))

##############################################################################
# visualization
#
# draw visualization of parameter effects
    plt.figure(figsize=(10, 8))
    xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
        plt.subplot(len(C_2d_range)*len(degree_range), len(gamma_2d_range), k + 1)
        plt.title("g 10^%d, C 10^%d, d %r" % (np.log10(gamma), np.log10(C), clf.degree),
                  size='medium')

    # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.jet)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=Y_2d, cmap=plt.cm.jet)
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')
    plt.show()
                                        
    #make the f1score function for the positive class
    f1positivescore=make_scorer(metrics.f1_score,pos_label=0) 
    
    # summarize the fit of the model
    print('Optimized hyperparameters from %s kernel are : C = %r gamma = %r degree = %d'%(kernel,cBest, gBest, dBest))
    #Concat train+dev
    X_all = np.vstack((v_train_set, v_dev_set))
    expected_all = np.concatenate((train_numeric_labels,dev_numeric_labels), axis=0)
    
    print('Start Dis-honest evaluation with train for test')
    model = svm.SVC(kernel=kernel, gamma=gBest, C=cBest, degree=dBest).fit(v_train_set, train_numeric_labels, sample_weight=None)   
    predicted = model.predict(v_train_set) 
    print(metrics.classification_report(train_numeric_labels, predicted, labels=None))
    
    print('Start Hold-Hout evaluation with train ,dev for test')
    model = svm.SVC(kernel=kernel, gamma=gBest, coef0=cBest, degree=dBest).fit(v_train_set, train_numeric_labels, sample_weight=None)   
    predicted = model.predict(v_dev_set) 
                #     make predictions
    print(metrics.classification_report(dev_numeric_labels, predicted, labels=None))
    print(metrics.confusion_matrix(dev_numeric_labels, predicted))
    print
    print(metrics.f1_score(dev_numeric_labels, predicted, pos_label=0))        
     
    print "Making 10-FCV with train+dev..."
    scores = cs.cross_val_score(model, X_all, expected_all, f1positivescore, cv=10, verbose=True)
    print("F1score positive class: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#     scores = cs.cross_val_score(model, X_all, expected_all, metrics.classification_report, cv=10, n_jobs=-1, verbose=True)
#     for score in scores:
#         print(score)
#     
    if not os.path.isdir('Modelos'):
        os.mkdir('Modelos')
    date = time.strftime("%H%M%d%m%Y")
    jl.dump(model, 'Modelos/CSVM'+kernel+date+'.pkl') 
    
    return model
################################################################################
#     
#
#      rf = RandomForestClassifier(n_estimators = 20, n_jobs = 8)
#      rf.fit(X,y)    
#     
if __name__=='__main__':
    main(sys.argv[1:])
    

  
    