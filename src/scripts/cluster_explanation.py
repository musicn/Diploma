import six
import sys
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
import numpy as np
from scipy.spatial.distance import cdist
import pysubgroup as ps
import pandas as pd

class RULES:
    def __init__(self, library_num):
        self.library_num = library_num
        self.rulesOut = []
        self.accuracyOut = []
        self.rulesIn = []
        self.accuracyIn = []
        self.model = None
    
    def calc_rules_outCluster(self, X, labels):
        if self.library_num == 0:
            for cluster in np.unique(labels):
                # create target variable for individual cluster / all same cluster labels are set to one for this for loop
                yc = (labels == cluster) * 1
                count_method1 = np.count_nonzero(yc == 1)
                # use SkopeRules to identify rules with a maximum of two comparison terms
                # dodaj iteracijo cez max_depth -> tradeoff med globino in precision in recall
                sr = SkopeRules(max_depth=None, max_depth_duplication=10,n_estimators=30).fit(X, yc)
                # print best decision rule
                print(cluster, sr.rules_[0][0])
                self.rulesOut.append(sr.rules_[0][0])
                # print precision and recall of best decision rule
                print(f"Precision: {sr.rules_[0][1][0]:.2f}",
                    f"Recall   : {sr.rules_[0][1][1]:.2f}\n")
                self.accuracyOut.append((sr.rules_[0][1][0],sr.rules_[0][1][1]))

    def calc_rules_inCLuster(self, X, y, labels):
        for cluster in np.unique(labels):
            yc = (labels == cluster) * 1
            indices = np.where(yc == 1)[0]
            X1 = np.copy(X)
            X1 = X1[indices,:]
            y1 = np.copy(y)
            y1 = y1[indices]
            if len(np.unique(y1)) == 1:
                print(cluster, 'all samples of class ' + str(y1[0]))
                self.rulesIn.append([[np.unique(y1)[0],'all samples of class ' + str(y1[0])]])
                self.accuracyIn.append([[np.unique(y1)[0],(1.0,1.0)]])
                continue
            print(cluster)
            rules_temp = []
            acc_temp = []
            for class_label in np.unique(y1):
                yl = (y1 == class_label) * 1
                sr = SkopeRules(max_depth=None,max_depth_duplication=20,recall_min=0.05,n_estimators=30).fit(X1, yl)
                # print best decision rule
                #print(class_label, sr.rules_[0][0])
                #self.rulesIn.append(sr.rules_[0][0])
                # print precision and recall of best decision rule
                #print(f"Precision: {sr.rules_[0][1][0]:.2f}",
                #    f"Recall   : {sr.rules_[0][1][1]:.2f}\n")
                #self.accuracyIn.append((sr.rules_[0][1][0],sr.rules_[0][1][1]))
                t1 = []
                t2 = []
                for ix in range(len(sr.rules_)):
                    t1.append([class_label,sr.rules_[ix][0]])
                    t2.append([class_label,(sr.rules_[ix][1][0],sr.rules_[ix][1][1])])
                rules_temp.append(t1)
                acc_temp.append(t2)
            self.rulesIn.append(rules_temp)
            self.accuracyIn.append(acc_temp)

    def get_rules_outCluster(self):
        return self.rulesOut
    
    def get_rules_inCluster(self):
        return self.rulesIn

    def get_accuracy_outCluster(self):
        return self.accuracyOut
    
    def get_accuracy_inCluster(self):
        return self.accuracyIn

    def reset(self):
        self.rulesOut = []
        self.accuracyOut = []
        self.rulesIn = []
        self.accuracyIn = []

class MEDOID:
    def __init__(self, distance_metric):
        self.distance_metric = distance_metric
        self.medoid = []

    def calc_medoid(self, X, y, labels):
        for cluster in np.unique(labels):
            yc = (labels == cluster) * 1
            indices = np.where(yc == 1)[0]
            X1 = np.copy(X)
            X1 = X1[indices,:].astype(float)
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

# class SUBGROUP_DISCOVERY:
#     def __init__(self):
#         self.rules = []

#     def calc_rules(self, X, y, labels):
#         for cluster in np.unique(labels):
#             yc = (labels == cluster) * 1
#             indices = np.where(yc == 1)[0]
#             X1 = np.copy(X)
#             X1 = X1[indices,:]
#             y1 = np.copy(y)
#             y1 = y1[indices]
#             data = np.hstack((X1, y1.reshape(-1, 1)))
#             selectors = ps.create_selectors(data, ignore=[-1])
#             result = ps.BeamSearch().execute(data, min_quality=0.5, beam_width=5)
#             a=5
