import matlab.engine
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.metrics import silhouette_score
import io
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import kneed
from scipy.ndimage import gaussian_filter1d
from hdbscan import HDBSCAN
from hdbscan import validity
import hdbscan
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

class MDEC:
    def __init__(self):
        self.eng = matlab.engine.start_matlab()        
        self.eng.cd(r'D:\FRI\Diploma\src\MDEC-master') # change directory to the location of the demo_1.m file
        self.X = None
        self.result_MDEC_HC = None
        self.result_MDEC_SC = None
        self.result_MDEC_BG = None
        self.percentage_explained = [1.0,1.0,1.0]
        self.model = None
        self.name = 'MDEC'

    def cluster(self, data_matrix, target_var_vec, num_clusters=0):
        #result = self.eng.demo_1(nargout=0) # call the function and return nothing
        self.X = np.copy(data_matrix)
        mat_data = matlab.double(data_matrix.tolist())
        K = len(np.unique(target_var_vec))
        result_MDEC_HC = None
        result_MDEC_SC = None
        result_MDEC_BG = None
        # silence output
        if num_clusters == 0: result_MDEC_HC, result_MDEC_SC, result_MDEC_BG = self.eng.runMDEC(mat_data, K, nargout=3, stdout=io.StringIO())
        else: result_MDEC_HC, result_MDEC_SC, result_MDEC_BG = self.eng.runMDEC(mat_data, num_clusters, nargout=3, stdout=io.StringIO())
        # unsilence output
        self.result_MDEC_HC = np.asarray(result_MDEC_HC)
        self.result_MDEC_SC = np.asarray(result_MDEC_SC)
        self.result_MDEC_BG = np.asarray(result_MDEC_BG)

    def get_labels(self):
        return self.result_MDEC_HC, self.result_MDEC_SC, self.result_MDEC_BG
    
    def evaluate_silhuette_avg(self):
        try:
            return [silhouette_score(self.X, self.result_MDEC_HC.ravel()), silhouette_score(self.X, self.result_MDEC_SC.ravel()), silhouette_score(self.X, self.result_MDEC_BG.ravel())]
        except ValueError:  #raised if `y` is empty.
            #print('poglej X in results')
            return [-9999,-9999,-9999]

    def evaluate_dbcv(self):
        try:
            return [validity.validity_index(self.X, self.result_MDEC_HC.ravel().astype(int)), validity.validity_index(self.X, self.result_MDEC_SC.ravel().astype(int)), validity.validity_index(self.X, self.result_MDEC_BG.ravel().astype(int))]
        except ValueError:  #raised if `y` is empty.
            #print('poglej X in results')
            return [-9999,-9999,-9999]


    def reset(self):
        self.X = None
        self.result_MDEC_HC = None
        self.result_MDEC_SC = None
        self.result_MDEC_BG = None
    
    def plot(self):
        plt.subplot(1,3,1)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.result_MDEC_HC, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_HC')
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.result_MDEC_SC, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_SC')
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.result_MDEC_BG, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_BG')
        plt.colorbar()
        plt.show()

    def plotPCA(self):
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(self.X)
        plt.subplot(1,3,1)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.result_MDEC_HC, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_HC')
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.result_MDEC_SC, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_SC')
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.result_MDEC_BG, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_BG')
        plt.colorbar()
        plt.show()

    def plotUMAP(self):
        X_2d = UMAP(
                  n_components=2, n_neighbors=200, min_dist=0
               ).fit_transform(self.X)
        plt.subplot(1,3,1)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.result_MDEC_HC, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_HC')
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.result_MDEC_SC, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_SC')
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.result_MDEC_BG, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_BG')
        plt.colorbar()
        plt.show()

    def plotTSNE(self, X_embedded):
        plt.subplot(1,3,1)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=self.result_MDEC_HC, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_HC')
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=self.result_MDEC_SC, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_SC')
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=self.result_MDEC_BG, cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('MDEC_BG')
        plt.colorbar()
        plt.show()

    def quit(self):
        self.eng.quit()

class KMEANS:
    def __init__(self):
        self.model = None
        self.X = None
        self.labels = []
        self.centers = []
        self.percentage_explained = [1.0]
        self.name = 'KMEANS'

    def cluster(self, data_matrix, target_var_vec, num_clusters=1):
        self.X = np.copy(data_matrix)
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        kmeans.fit(self.X)
        self.model = kmeans
        self.labels.append(kmeans.labels_)
        self.centers.append(kmeans.cluster_centers_)

    def get_labels(self, ix):
        return self.labels[ix]
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate_silhuette_avg(self):
        try:
            return [silhouette_score(self.X, self.labels[0])]
        except ValueError:  #raised if `y` is empty.
            #print('poglej X in results')
            return [-9999]

    def evaluate_dbcv(self):
        try:
            return [validity.validity_index(self.X, self.labels[0])]
        except ValueError:  #raised if `y` is empty.
            #print('poglej X in results')
            return [-9999]

    def reset(self):
        self.model = None
        self.X = None
        self.labels = []
        self.centers = []

    def plot(self):
        scatter = plt.scatter(self.X[:,0], self.X[:,1], c=self.labels[0])
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('KMEANS')
        legend_handles, legend_labels = scatter.legend_elements()
        plt.legend(legend_handles, legend_labels, loc='best')
        plt.show()

    def plotPCA(self):
        print('not implemented')

    def plotUMAP(self):
        print('not implemented')

    def plotTSNE(self, X_embedded):
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=self.labels[0], cmap='coolwarm')
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.title('KMEANS')
        plt.colorbar()
        plt.show()

    def quit(self):
        print('not implemented')

class DBSCAN_C:
    def __init__(self):
        self.X = None
        self.labels = []
        self.core_samples = []
        self.percentage_explained = []
        self.model = None
        self.name = 'DBSCAN'

    def cluster(self, data_matrix, target_var_vec, num_clusters=1):
        self.X = np.copy(data_matrix)
        data_dims = self.X.shape[1]
        min_samples = 2 * data_dims # Minimum number of samples in a neighborhood for a point to be considered a core point
        for ix in range(9):
            min_samples = 2 * data_dims
            min_samples = min_samples + (int(round((ix-3) * (0.2 * min_samples))))
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors_fit = neighbors.fit(self.X)
            distances, indices = neighbors_fit.kneighbors(self.X)
            distances = np.sort(distances, axis=0)
            distances = distances[:,1]
            x = np.linspace(0,len(distances)-1,len(distances))
            sigma = 1.5  # Adjust the sigma parameter for the desired smoothing effect
            smoothed_y = gaussian_filter1d(distances, sigma)
            # plt.figure(figsize=(8, 6))
            # plt.plot(distances)
            # plt.show()
            # plt.figure(figsize=(8, 6))
            # plt.plot(smoothed_y)
            # plt.show()
            kneedle = kneed.KneeLocator(x,smoothed_y,curve='convex',polynomial_degree=7)
            knee_point = kneedle.knee
            elbow_point = kneedle.elbow
            # print('Elbow: ', knee_point)
            # print('Knee: ', knee_point)
            # kneedle.plot_knee()
            eps = distances[int(round(elbow_point))]  # Maximum distance between samples for them to be considered neighbors
            x_range_unit = (0.05 * (len(distances) - 1)) / 9
            eps = distances[int(round(elbow_point)) + int(round(x_range_unit * (num_clusters - 6)))]
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(self.X)
            self.model = dbscan
            self.labels.append(dbscan.labels_)
            self.core_samples.append(dbscan.core_sample_indices_)
            self.percentage_explained.append(1 - (np.count_nonzero(dbscan.labels_ == -1)/len(dbscan.labels_)))

    def get_labels(self, ix):
        return self.labels[ix]
    
    def evaluate_silhuette_avg(self):
        # points that have -1 are outliers and do not belong to any cluster -> throw them out
        ret = []
        for ix in range(len(self.labels)):
            valid_indices = np.where(self.labels[ix] >= 0)[0]
            X_filtered = self.X[valid_indices]
            labels_filtered = self.labels[ix][self.labels[ix] != -1]
            try:    
                ret.append(silhouette_score(X_filtered, labels_filtered))
            except ValueError:  #raised if `y` is empty.
                #print('poglej X in results')
                ret.append(-9999)
        return ret
    
    def evaluate_dbcv(self):
        # points that have -1 are outliers and do not belong to any cluster -> throw them out
        ret = []
        for ix in range(len(self.labels)):
            #valid_indices = np.where(self.labels[ix] >= 0)[0]
            #X_filtered = self.X[valid_indices]
            #labels_filtered = self.labels[ix][self.labels[ix] != -1]
            try:
                dbcv_score = validity.validity_index(self.X.astype(np.double), self.labels[ix])
            except ValueError as ve:  #raised if `y` is empty.
                print(ve)
                dbcv_score = -9999
            ret.append(dbcv_score)
        return ret
    
    def reset(self):
        self.X = None
        self.labels = []
        self.core_samples = []
        self.percentage_explained = []

    def plot(self):
        for ix in range(9):
            plt.subplot(3,3,ix+1)
            scatter = plt.scatter(self.X[:,0], self.X[:,1], c=self.labels[ix], cmap='rainbow')
            #plt.colorbar()
            legend_handles, legend_labels = scatter.legend_elements()
            plt.legend(legend_handles, legend_labels, loc='best')
            plt.xlabel('feat1')
            plt.ylabel('feat2')
            plt.title('DBSCAN')
        plt.show()

    def plotPCA(self):
        print('not implemented')

    def plotUMAP(self):
        print('not implemented')

    def plotTSNE(self, X_embedded):
        for ix in range(9):
            plt.subplot(3,3,ix+1)
            scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=self.labels[ix], cmap='rainbow')
            #plt.colorbar()
            legend_handles, legend_labels = scatter.legend_elements()
            plt.legend(legend_handles, legend_labels, loc='best')
            plt.xlabel('feat1')
            plt.ylabel('feat2')
            plt.title('DBSCAN')
        plt.show()

    def quit(self):
        print('not implemented')

class HDBSCAN_C:
    def __init__(self):
        self.X = None
        self.labels = []
        self.n_clusters = []
        self.percentage_explained = []
        self.model = None
        self.name = 'HDBSCAN'

    def cluster(self, data_matrix, target_var_vec, num_clusters=1):
        self.X = np.copy(data_matrix)
        for ix in range(9):
            hdbscan_obj = HDBSCAN(min_cluster_size=(num_clusters*5), min_samples=(ix+2)*5, prediction_data=True)
            labels = hdbscan_obj.fit_predict(self.X)
            self.model = hdbscan_obj
            self.labels.append(labels)
            self.n_clusters.append(len(set(labels)) - 1)
            self.percentage_explained.append(1 - (np.count_nonzero(labels == -1)/len(labels)))

    def get_labels(self, ix):
        return self.labels[ix]
    
    def evaluate_silhuette_avg(self):
        ret = []
        for ix in range(len(self.labels)):
            valid_indices = np.where(self.labels[ix] >= 0)[0]
            X_filtered = self.X[valid_indices]
            labels_filtered = self.labels[ix][self.labels[ix] != -1]
            try:    
                ret.append(silhouette_score(X_filtered, labels_filtered))
            except ValueError:  #raised if `y` is empty.
                #print('poglej X in results')
                ret.append(-9999)
        return ret

    def evaluate_dbcv(self):
        # points that have -1 are outliers and do not belong to any cluster -> throw them out
        # valid_indices = np.where(self.labels >= 0)[0]
        # X_filtered = self.X[valid_indices]
        # labels_filtered = self.labels[self.labels != -1]
        # try:
        #     return [validity.validity_index(X_filtered, labels_filtered)]
        # except ValueError:  #raised if `y` is empty.
        #     #print('poglej X in results')
        #     return [-9999]
        # points that have -1 are outliers and do not belong to any cluster -> throw them out
        ret = []
        for ix in range(len(self.labels)):
            # valid_indices = np.where(self.labels[ix] >= 0)[0]
            # X_filtered = self.X[valid_indices]
            # labels_filtered = self.labels[ix][self.labels[ix] != -1]
            try:
                dbcv_score = validity.validity_index(self.X.astype(np.double), self.labels[ix])
            except ValueError:  #raised if `y` is empty.
                #print('poglej X in results')
                dbcv_score = -9999
            ret.append(dbcv_score)
        return ret
    
    def reset(self):
        self.X = None
        self.labels = []
        self.n_clusters = []
        self.percentage_explained = []

    def plot(self):
        # scatter = plt.scatter(self.X[:,0], self.X[:,1], c=self.labels, cmap='rainbow')
        # #plt.colorbar()
        # legend_handles, legend_labels = scatter.legend_elements()
        # plt.legend(legend_handles, legend_labels, loc='best')
        # plt.xlabel('feat1')
        # plt.ylabel('feat2')
        # plt.title('HDBSCAN')
        # plt.show()
        for ix in range(9):
            plt.subplot(3,3,ix+1)
            scatter = plt.scatter(self.X[:,0], self.X[:,1], c=self.labels[ix], cmap='rainbow')
            #plt.colorbar()
            legend_handles, legend_labels = scatter.legend_elements()
            plt.legend(legend_handles, legend_labels, loc='best')
            plt.xlabel('feat1')
            plt.ylabel('feat2')
            plt.title('HDBSCAN')
        plt.show()

    def plotPCA(self):
        print('not implemented')

    def plotUMAP(self):
        print('not implemented')

    def plotTSNE(self, X_embedded):
        for ix in range(9):
            plt.subplot(3,3,ix+1)
            scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=self.labels[ix], cmap='rainbow')
            #plt.colorbar()
            legend_handles, legend_labels = scatter.legend_elements()
            plt.legend(legend_handles, legend_labels, loc='best')
            plt.xlabel('feat1')
            plt.ylabel('feat2')
            plt.title('HDBSCAN')
        plt.show()

    def quit(self):
        print('not implemented')

'''
def main():
    mdec = MDEC()
    mdec.cluster()
    mdec.plot()


if __name__ == "__main__":
    main()
'''