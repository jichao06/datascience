import pandas as pd
import numpy as np
import scipy as sp
from sklearn.cross_validation import KFold

def distance(p0, p1):
    return ((p0-p1)**2).sum()
    
def seeds_classify(k_neighbors, training_features, training_labels, test_feature):
    distances = training_features.apply(lambda x: distance(x, test_feature), axis=1).order()
    return training_labels[distances[:k_neighbors].index].value_counts().argmax()
    
def one_fold_test(k_neighbors, training_features, training_labels, test_features, test_labels):
    pred_labels = [seeds_classify(k_neighbors, training_features, training_labels, test_features.ix[i]) for i in test_features.index]
    return (pred_labels == test_labels).mean()
    
def test_seeds_classify(k_neighbors, features, labels):
    kf = KFold(len(features), n_folds=10)
    accuracy = 0
    for train, test in kf:
        accuracy += one_fold_test(k_neighbors, features.ix[train], labels[train], features.ix[test], labels[test])
    accuracy /= 10
    return accuracy
    
def evaluate_seeds(k_neighbors=1):
    seeds = pd.read_table('seeds_dataset.txt', names=['area','perimeter','compactness','l_kernel','w_kernel','coefficient','l_groove','target'])
    features = seeds.ix[:,:7]
    features = (features - features.mean(axis=0))/features.std(axis=0)
    labels = seeds.ix[:,7]
    print 'Accuracy:%f' % test_seeds_classify(k_neighbors, features, labels)