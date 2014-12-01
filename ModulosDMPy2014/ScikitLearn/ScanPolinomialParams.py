#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 28/11/2014

@author: spolex

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
    train_set = np.asarray([fila[0:len(fila)-1] for fila in labeled_set])
    train_set_labels = np.asarray([fila[-1] for fila in labeled_set])
    
    atts = data['attributes']
    atts_names = [fila[0] for fila in atts]
    att_values = [fila [1] for fila in atts]
    labels = np.array(att_values[len(att_values)-1])
    
    print 'TRAIN DATA SHAPE'
    print train_set.shape
    print 'Attributes NUM'
    print len(atts_names)
    print 'LABELS FOR CLASS'
    print labels
    
    print('Develop data loading....')
    datadev_set = arff.load(open(argv[2],'rb'))
    dev_labeled_set = datadev_set['data']
    dev_set = np.asarray([fila[0:len(fila)-1] for fila in dev_labeled_set])
    dev_set_labels = np.asarray([fila[-1] for fila in dev_labeled_set])
    
    dev_atts = data['attributes']
    dev_atts_names = [fila[0] for fila in dev_atts]
    dev_att_values = [fila [1] for fila in dev_atts]
    dev_labels = np.array(dev_att_values[len(dev_att_values)-1])
    
    print 'DEV DATA SHAPE'
    print dev_set.shape
    print 'DEV Attributes NUM'
    print len(dev_atts_names)
    print 'LABELS FOR DEV CLASS'
    print dev_labels
    
####    
    print ('Preprocesing data...')
    # #    parse a un dict para poder vectorizar los att categoricos
    print ('Parsing categorical data...')

    dict_list = []
    N,F = train_set.shape
    for n in range(N):
        d = {}
        for f in range(F):
            feature = atts_names[f]
            d[feature] = train_set[n,f]
        dict_list.append(d)
        
    dev_dict_list = []
    N,F = dev_set.shape
    for n in range(N):
        d = {}
        for f in range(F):
            feature = dev_atts_names[f]
            d[feature] = dev_set[n,f]
        dev_dict_list.append(d)
    
    
    #Fit vectorizer for each dict
    v = DictVectorizer(sparse=False,dtype=np.float16)
    
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
    le = preprocessing.LabelEncoder()    
    le.fit(train_set_labels)
    train_numeric_labels = le.transform(train_set_labels)
    le.fit(dev_set_labels)
    dev_numeric_labels = le.transform(dev_set_labels)
#########    
#     print ('Fitting the model')
#     #Fit the model
#     model = svm.SVC(kernel='rbf', gamma=2, C=1, degree=0).fit(v_train_set, train_numeric_labels, sample_weight=None)
#     print "Making predictions..."
#     expected=dev_numeric_labels
#     predicted = model.predict(v_dev_set)
#     print "Making Hold out evaluation with dev set..."
#     f1Aux = metrics.f1_score(expected, predicted, pos_label=0)
#     print ("New F1Score = %r" %f1Aux)
#     print(metrics.classification_report(expected, predicted, labels=None))
##########
    cBest = 0.
    gBest = 0.  
    dBest = 0.          
    print('Start scaning data for Polinomial kernel....')
    f1Aux=0.0
    f1Best=0.0
    if kernel=='rbf':
        maxD=3
    else:
        maxD=5
    for d in range(2,maxD):#2,5
        for i in range(-15,12):#-15,12
            c=2**i
            for j in range(-3,5):#-3,5
                g=2**j
                print("Hyperparameters: coef0 = %r gamma = %r degree = %d...." %(c,g,d))
                #   fit the model 
             
                model = svm.SVC(kernel=kernel, gamma=g, coef0=c, degree=d, class_weight='auto').fit(v_train_set, train_numeric_labels, sample_weight=None)  
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
                    print('Hyperparameters has been changed New degree = %d New coef0= %r New gamma = %r ' %(d,c,g))
                    cBest = c
                    gBest = g  
                    dBest = d          
    # summarize the fit of the model
    print('Optimized hyperparameters from %s kernel are : coef0 = %r gamma = %r degree = %d'%(kernel,cBest, gBest, dBest))
    #Concat train+dev
    X_all = np.vstack((v_train_set, v_dev_set))
    expected_all = np.concatenate((train_numeric_labels,dev_numeric_labels), axis=0)
    
    print('Start Dis-honest evaluation with train for test')
    model = svm.SVC(kernel='rbf', gamma=gBest, coef0=cBest, degree=dBest).fit(v_train_set, train_numeric_labels, sample_weight=None)   
    predicted = model.predict(v_train_set) 
    print(metrics.classification_report(train_numeric_labels, predicted, labels=None))
    
    print('Start Hold-Hout evaluation with train ,dev for test')
    model = svm.SVC(kernel='rbf', gamma=gBest, coef0=cBest, degree=dBest).fit(v_train_set, train_numeric_labels, sample_weight=None)   
    predicted = model.predict(v_dev_set) 
                #     make predictions
    print(metrics.classification_report(dev_numeric_labels, predicted, labels=None))
    print(metrics.confusion_matrix(dev_numeric_labels, predicted))
    print
    print(metrics.f1_score(dev_numeric_labels, predicted, pos_label=0))        
    print(metrics.homogeneity_completeness_v_measure(dev_numeric_labels, predicted))
     
    print "Making 10-FCV with train+dev..."
    scores = cs.cross_val_score(model, X_all, expected_all, metrics.f1_score, cv=10, n_jobs=-1, verbose=True)
    print("F1score weighted: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cs.cross_val_score(model, X_all, expected_all, metrics.classification_report, cv=10, n_jobs=-1, verbose=True)
    for score in scores:
        print(score)
    
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
    

  
    