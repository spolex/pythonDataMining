'''
Created on 30/11/2014

@author: spolex
'''
import arff
import numpy as np
import os
import time
#from scipy.io.arff.arffread import MetaData
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer#, FeatureHasher
#from sklearn.tree.tree import ExtraTreeClassifier
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn import metrics
from sklearn import cross_validation as cs
from sklearn.externals import joblib as jl 
import sys
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble.forest import RandomForestClassifier


def main(self,argv=sys.argv):
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
    
    #select 
    
    # # transform non-numerical labels to numerical
    le = preprocessing.LabelEncoder()    
    le.fit(train_set_labels)
    train_numeric_labels = le.transform(train_set_labels)
    le.fit(dev_set_labels)
    dev_numeric_labels = le.transform(dev_set_labels)
    
    clf=RandomForestClassifier(n_estimators=10,min_samples_split=1,random_state=0)
    scores = cross_val_score(clf, v_dev_set, dev_numeric_labels, metrics.classification_report, cv=10)
    for score in scores:
        print score
    return clf




if __name__ == '__main__':
    main(sys.argv[1:])