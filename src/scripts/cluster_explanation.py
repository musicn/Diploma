import six
import sys
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
import numpy as np
from scipy.spatial.distance import cdist

class RULES:
    def __init__(self, library_num):
        self.library_num = library_num
        self.rules = []
        self.accuracy = []

    def calc_rules(self, X, labels):
        if self.library_num == 0:
            for cluster in np.unique(labels):
                # create target variable for individual cluster / all same cluster labels are set to one for this for loop
                yc = (labels == cluster) * 1
                # use SkopeRules to identify rules with a maximum of two comparison terms
                # dodaj iteracijo cez max_depth -> tradeoff med globino in precision in recall
                sr = SkopeRules(max_depth=4).fit(X, yc)
                # print best decision rule
                print(cluster, sr.rules_[0][0])
                self.rules.append(sr.rules_[0][0])
                # print precision and recall of best decision rule
                print(f"Precision: {sr.rules_[0][1][0]:.2f}",
                    f"Recall   : {sr.rules_[0][1][1]:.2f}\n")
                self.accuracy.append((sr.rules_[0][1][0],sr.rules_[0][1][1]))

    def get_rules(self):
        return self.rules
    
    def get_accuracy(self):
        return self.accuracy
    
    def reset(self):
        self.rules = []
        self.accuracy = []

class MEDOID:
    def __init__(self, distance_metric):
        self.distance_metric = distance_metric
        self.medoid = None

    def calc_medoid(self, X):
        if self.distance_metric == 'euclidean':    
            distances = cdist(X, X, metric='euclidean')
        total_distances = np.sum(distances, axis=1)
        medoid_index = np.argmin(total_distances)
        medoid = X[medoid_index]
        print(medoid)
        self.medoid = medoid
    
    def get_medoid(self):
        return self.medoid