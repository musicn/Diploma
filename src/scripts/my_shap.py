import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP

class SHAP:
    def __init__(self, model, mode, X_train):
        shap.initjs()
        masker = shap.maskers.Independent(data = X_train)
        if mode == 0: self.explainer = shap.TreeExplainer(model)
        elif mode == 1: self.explainer = shap.LinearExplainer(model, masker=masker)

    def calc_shap_val(self, X_train):
        return self.explainer.shap_values(X_train)
    
    def calc_shap_val_clusters(self, X_train, labels):
        shap_clusters = []
        Xs = []
        for cluster in np.unique(labels):
            # create target variable for individual cluster / all same cluster labels are set to one for this for loop
            yc = (labels == cluster) * 1
            Xf = X_train[yc == 1]
            shap_clusters.append(self.explainer.shap_values(Xf))
            Xs.append(Xf)
        return shap_clusters, Xs
    
    def plot_force(self, shap_val, X):
        shap.save_html('force_plot.html', shap.force_plot(self.explainer.expected_value, shap_val, X))

    def plot_summary(self, shap_val, X):
        shap.summary_plot(shap_val, X)

    def plot_bar(self, shap_val):
        shap.bar_plot(shap_val)

    def plot_dependence(self, shap_val, X, feature_index):
        shap.dependence_plot(feature_index, shap_val, X)
        plt.show()

    def plot_clusters_2d(self, shap_val, y):
        #colors = np.where(np.array(y) == 0, 'blue', 'red')
        plt.scatter(shap_val[:,0], shap_val[:,1], c=y)
        plt.show()

    def plot_clusters_PCA(self, shap_val, y):
        colors = np.where(np.array(y) == 0, 'blue', 'red')
        pca = PCA(n_components=2)
        shap_val_2d = pca.fit_transform(np.copy(shap_val))
        plt.scatter(shap_val_2d[:,0], shap_val_2d[:,1], c=colors)
        plt.show()

    def plot_clusters_UMAP(self, shap_val, y):
        colors = np.where(np.array(y) == 0, 'blue', 'red')
        shap_val_2d = UMAP(
               n_components=2, n_neighbors=200, min_dist=0
           ).fit_transform(np.copy(shap_val))
        plt.scatter(shap_val_2d[:,0], shap_val_2d[:,1], c=colors)
        plt.show()