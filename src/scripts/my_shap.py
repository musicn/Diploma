import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP

class SHAP:
    def __init__(self, model):
        shap.initjs()
        self.explainer = shap.TreeExplainer(model)

    def calc_shap_val(self, X_train):
        return self.explainer.shap_values(X_train)
    
    def plot_force(self, shap_val, X):
        shap.save_html('force_plot.html', shap.force_plot(self.explainer.expected_value, shap_val, X))

    def plot_summary(self, shap_val, X):
        shap.summary_plot(shap_val, X)

    def plot_dependence(self, shap_val, X, feature_index):
        shap.dependence_plot(feature_index, shap_val, X)
        plt.show()

    def plot_clusters_2d(self, shap_val, y):
        colors = np.where(np.array(y) == 0, 'blue', 'red')
        plt.scatter(shap_val[:,0], shap_val[:,1], c=colors)
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