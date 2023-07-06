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
        self.medoid = []

    def calc_medoid(self, X, y, labels):
        for cluster in np.unique(labels):
            yc = (labels == cluster) * 1
            indices = np.where(yc == 1)[0]
            X1 = np.copy(X)
            X1 = X1[indices,:]
            y1 = np.copy(y)
            y1 = y1[indices]
            if self.distance_metric == 'euclidean':    
                distances = cdist(X1, X1, metric='euclidean')
            total_distances = np.sum(distances, axis=1)
            medoid_index = np.argmin(total_distances)
            medoid = [X1[medoid_index], y1[medoid_index]]
            print(medoid)
            self.medoid.append(medoid)
    
    def get_medoid(self):
        return self.medoid
    
    def reset(self):
        self.medoid = []

class CLASS_PROB:
    def __init__(self):
        self.class_probs = []

    def calc_probs(self, y, labels):
        for cluster in np.unique(labels):
            yc = (labels == cluster) * 1
            class_labels = y[yc == 1]
            prob_dict = dict()
            for class_l in np.unique(class_labels):
                count = np.count_nonzero(class_labels == class_l)
                prob_dict[class_l] = count/len(class_labels)
            print(prob_dict)
            self.class_probs.append(prob_dict)
    
    def get_probs(self):
        return self.class_probs
    
    def reset(self):
        self.class_probs = []