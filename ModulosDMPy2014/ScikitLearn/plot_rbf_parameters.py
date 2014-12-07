'''
========================================
RBF SVM parameters with Grid Search Scan
========================================
**************************************************************************
This example illustrates the effect of the parameters `gamma`            
and `C` of the rbf kernel SVM.

Intuitively, the `gamma` parameter defines how far the influence
of a single training example reaches, with low values meaning 'far'
and high values meaning 'close'.
The `C` parameter trades off misclassification of training examples
against simplicity of the decision surface. A low C makes
the decision surface smooth, while a high C aims at classifying
all training examples correctly.

Two plots are generated.  The first is a visualization of the
decision function for a variety of parameter values, and the second
is a heatmap of the classifier's cross-validation accuracy as
a function of `C` and `gamma`. For this example we explore a relatively
large grid for illustration purposes. In practice, a logarithmic
grid from `10**-3` to `10**3` is usually sufficient.
**************************************************************************
@author: spolex

Created on 06/12/2014

This code is an adaptation for the scikit learn rbf svm parameters, but
had been adapted for categorical and numerical data:

http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

A high C helps to balance unbalanced training set

Intuitively, the gamma parameter defines how far the influence of a single training 
example reaches, with low values meaning ‘far’ and high values meaning ‘close’.

Scoring F1-score for positive class in first position.

'''
from sklearn.metrics import metrics
from sklearn.metrics.scorer import make_scorer

import numpy as np
from sklearn.feature_extraction import DictVectorizer
import sys
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import arff as a
from sklearn import preprocessing


##############################################################################
# Load and prepare data set
#
# dataset for grid search
def main(self,argv=sys.argv):
    print(__doc__)
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
# I am cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.
    scaler = preprocessing.StandardScaler()
    v_train_set = scaler.fit_transform(v_train_set)
    X_2d = scaler.fit_transform(X_2d)
#     iris = load_iris()
#     X = iris.data
#     Y = iris.target
# 
#      dataset for decision function visualization
#     X_2d = X[:, :2]
#     X_2d = X_2d[Y > 0]
#     Y_2d = Y[Y > 0]
#     Y_2d -= 1

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.

    scaler = StandardScaler()

    v_train_set = scaler.fit_transform(v_train_set)
    X_2d = scaler.fit_transform(X_2d)

##############################################################################
# Train classifier
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

    C_range = 10.0 ** np.arange(-2, 9)
    gamma_range = 10.0 ** np.arange(-5, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedKFold(y=train_numeric_labels, n_folds=3)
    f1positivescore=make_scorer(metrics.f1_score,pos_label=0)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, scoring=f1positivescore)
    grid.fit(v_train_set, train_numeric_labels)

    print("The best classifier is: ", grid.best_estimator_)

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it  takes a while to train)
    C_2d_range = [1, 1e2, 1e4]
    gamma_2d_range = [1e-1, 1, 1e1]
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            clf.fit(X_2d, Y_2d)
            classifiers.append((C, gamma, clf))

##############################################################################
# visualization
#
# draw visualization of parameter effects
    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
        plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        plt.title("gamma 10^%d, C 10^%d" % (np.log10(gamma), np.log10(C)),
                  size='medium')

    # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.jet)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=Y_2d, cmap=plt.cm.jet)
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')

# plot the scores of the grid
# grid_scores_ contains parameter settings and scores
    score_dict = grid.grid_scores_

# We extract just the scores
    scores = [x[1] for x in score_dict]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))
    print max([max(score) for score in scores])

# draw heatmap of accuracy as a function of gamma and C
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)

    plt.show()
    return clf;
if __name__ == '__main__':
    main(sys.argv[1:])
