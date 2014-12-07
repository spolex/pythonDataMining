#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 29/11/2014

@author: spolex

'''
import sys
import numpy as np
from sklearn import svm
#from sklearn import preprocessing
#from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


import arff as a
#from sklearn.cross_validation import train_test_split


def main(self,argv=sys.argv):
        
    print('Loading data...')
    data = a.load(open(argv[1],'rb'))
    labeled_set = data['data']
    train_set = [fila[0:len(fila)-1] for fila in labeled_set]
#    train_set_labels = np.asarray([fila[-1] for fila in labeled_set])
    
    atts = data['attributes']
    atts_names = [fila[0] for fila in atts]
    att_values = [fila [1] for fila in atts]
    labels = np.array(att_values[len(att_values)-1])
    
    print 'Attributes NUM'
    print len(atts_names)
    print 'LABELS FOR CLASS'
    print labels
    
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
# 
# #    Para convertir los datos categoricos que NO pueden ser utilizados
# #    por el clasificador en numericos convertir las instancias en un
# #    diccionario que puede ser procesado: DictVectorizer
# 
    #Fit vectorizer for each dict
    v = DictVectorizer(sparse=False,dtype=np.float64)
    
    v_train_set = v.fit_transform(dict_list[0])
    for i in range(1,len(dict_list)):
        train_set_instance = v.fit_transform(dict_list[i],)
        v_train_set = np.vstack((v_train_set,train_set_instance))
        
#Split for obtain set for train an another test for novelty detection
#    v_train_set,v_train_set_test = train_test_split(v_train_set,test_size=0.33, random_state=42)
    
    
    
    
    # # transform non-numerical labels to numerical
    train_set_labels=[]
    for fila in labeled_set:
        train_set_labels.append(fila[-1])
    le = preprocessing.LabelEncoder()    
    le.fit(train_set_labels)
    
    
#dataset for decision function visualization
    train_numeric_labels = le.transform(train_set_labels)
    X_2d = v_train_set[:, -2:]
    Y_2d = train_numeric_labels
    
# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.
    scaler = preprocessing.StandardScaler()
    v_train_set = scaler.fit_transform(v_train_set)
    X_2d = scaler.fit_transform(X_2d)

#train classifier
#     print('Training OneClassSVM...')
#     one = svm.OneClassSVM(kernel='rbf', gamma=10., nu=0.1 )

    print('Training OneClassSVM...')
    one = svm.OneClassSVM(gamma=2.,nu=0.1)
    one.fit(X_2d,Y_2d)  
##############################################################################
# visualization
#
#    draw visualization 
    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(-2, 5, 100), np.linspace(-2, 5, 100)) 
    # visualize parameter's effect on decision function
    Z = one.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
# visualize decision function for these parameters.
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.jet)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=Y_2d,cmap=plt.cm.jet)
    plt.xticks()
    plt.yticks()
    plt.axis('tight')
    plt.show()
    print ('End')
# 
#     

#     if not os.path.isdir('Modelos'):  
#     jl.dump(one, 'Modelos/'+date+'oneClass.pkl') 
    
#     print "El número de outliers encontrados es %d" %len(outliers)
    
    
#     print ('Número de Outliers')
#     print n_error_train
#     
#     return self
    return one;
if __name__ == '__main__':
    main(sys.argv[1:])