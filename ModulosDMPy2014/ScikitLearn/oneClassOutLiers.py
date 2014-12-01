#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 29/11/2014

@author: spolex

pre: La clase debe estar en última posición
'''
import sys
import os
import numpy as np
from sklearn import svm
#from sklearn import preprocessing
#from sklearn import metrics
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib as jl
import arff as a
from sklearn.cross_validation import train_test_split


def main(self,argv=sys.argv):
        
    print('Loading data...')
    data = a.load(open(argv[1],'rb'))
    labeled_set = data['data']
    train_set = np.asarray([fila[0:len(fila)-1] for fila in labeled_set])
#    train_set_labels = np.asarray([fila[-1] for fila in labeled_set])
    
    atts = data['attributes']
    atts_names = [fila[0] for fila in atts]
    att_values = [fila [1] for fila in atts]
    labels = np.array(att_values[len(att_values)-1])
    
    print 'DATA SHAPE'
    print train_set.shape
    print 'Attributes NUM'
    print len(atts_names)
    print 'LABELS FOR CLASS'
    print labels
    
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
# 
# #    Para convertir los datos categoricos que NO pueden ser utilizados
# #    por el clasificador en numericos convertir las instancias en un
# #    diccionario que puede ser procesado: DictVectorizer
# 
    #Fit vectorizer for each dict
    v = DictVectorizer(sparse=False,dtype=np.float16)
    
    v_train_set = v.fit_transform(dict_list[0])
    for i in range(1,len(dict_list)):
        train_set_instance = v.fit_transform(dict_list[i])
        v_train_set = np.vstack((v_train_set,train_set_instance))
    
    v_train_set,v_train_set_test = train_test_split(v_train_set,test_size=0.70, random_state=42)
    print type(v_train_set_test)
#     
# # transform non-numerical labels to numerical
#     le = preprocessing.LabelEncoder()    
#     le.fit(train_set_labels)
#     y_numeric = le.transform(train_set_labels)
#     
# 
# #   fit the model 
    print('Training OneClassSVM...')
    one = svm.OneClassSVM(kernel='rbf', gamma=0.1, nu=0.01 )
    one.fit(v_train_set)
    print('End of train...')
    
    print('Start Outlier Detection...')
    predicted = one.predict(v_train_set_test)
    outliers = predicted[predicted == -1]
    
    if not os.path.isdir('Modelos'):
        os.mkdir('Modelos')
    date = time.strftime("%H%M%d%m%Y")
    jl.dump(one, 'Modelos/'+date+'oneClass.pkl') 
    
    
# # summarize the fit of the model
#     print(metrics.classification_report(expected, predicted))
#     print('CONFUSION MATRIX')
#     print(metrics.confusion_matrix(expected, predicted, labels=None))
#     print
#     print('F1-SCORING FOR POSITIVE CLASS')
#     print(metrics.f1_score(expected, predicted, pos_label=0))
#     print('ACCURACY')
#     print(metrics.accuracy_score(expected, predicted))
#     print "El número de outliers encontrados es %d" %len(outliers)
    
    
#     print ('Número de Outliers')
#     print n_error_train
#     
#     return self
    return one;
if __name__ == '__main__':
    main(sys.argv[1:])