import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from umap import UMAP
from keras.datasets import mnist
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import anndata
import scanpy as sc

### common functions

def read_txt(path):
    f = open(path)
    data = [] 
    run = True
    while(run):
        row = f.readline()
        if row == '':
            break
        data.append(row)
    return data

### 1

def construct_artificial_binary_3d_1():
    np.random.seed(42)

    num_samples = 1000
    num_clusters = 4
    points_per_cluster = num_samples // num_clusters

    random_numbers = np.random.rand(num_samples)

    angles = np.linspace(0, 2 * np.pi, points_per_cluster)
    radius = np.random.uniform(0.0, 1.0, points_per_cluster)
    x_coords = np.cos(angles) * radius
    y_coords = np.sin(angles) * radius
    
    X = np.zeros((1000,3))
    y = np.zeros(1000)

    offset_from_center = 0.55
    # cluster00
    c00 = np.column_stack((x_coords - offset_from_center, y_coords + offset_from_center))
    X[:250,:2] = c00
    y[:250] = 1
    # cluster01
    c01 = np.column_stack((x_coords + offset_from_center, y_coords - offset_from_center))
    X[250:500,:2] = c01
    y[250:500] = 1
    # cluster10
    c10 = np.column_stack((x_coords - offset_from_center, y_coords - offset_from_center))
    X[500:750,:2] = c10
    # cluster11
    c11 = np.column_stack((x_coords + offset_from_center, y_coords + offset_from_center))
    X[750:,:2] = c11

    rand_column = np.random.uniform(0.0, 1.0, num_samples)
    X[:,2] = rand_column

    return X, y

def plot2dPCA_artificial_binary_3d_1(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA - 2D: artificial binary 3d data')
    plt.show()

def plot2dORIG_artificial_binary_3d_1(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('First two dimensions: artificial binary 3d data')
    plt.show()

def plot3d_artificial_binary_3d_1(X, y):
    print()

### 2 ... 1v4 circural groups

def construct_artificial_binary_2d_1():
    X0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=250)
    X1 = np.concatenate([np.random.multivariate_normal([3, 4], [[0.5, 0], [0, 1]], size=75),
                      np.random.multivariate_normal([2, -4], [[0.95, 0], [0, 1]], size=75),
                      np.random.multivariate_normal([-2, 3], [[1.2, 0], [0, 1]], size=75),
                      np.random.multivariate_normal([-2, -4], [[0.8, 0], [0, 1]], size=75)])
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(250), np.ones(300)])
    return X, y

def plot_artificial_binary_2d_1(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Synthetic Binary Classification Dataset')
    plt.colorbar()
    plt.show()

### 3 ... spiral groups

def construct_artificial_binary_2d_2():
    np.random.seed(0)
    theta = np.linspace(0, 4 * np.pi, 400)
    r = np.linspace(0.5, 1.5, 400)
    X_1 = np.zeros((400, 2))
    X_1[:, 0] = r * np.cos(theta) + np.random.normal(0, 0.1, 400)
    X_1[:, 1] = r * np.sin(theta) + np.random.normal(0, 0.1, 400)

    X_2 = np.zeros((400, 2))
    X_2[:, 0] = (r + 0.5) * np.cos(theta + np.pi) + np.random.normal(0, 0.1, 400)
    X_2[:, 1] = (r + 0.5) * np.sin(theta + np.pi) + np.random.normal(0, 0.1, 400)

    X_3 = np.concatenate([np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 1]], size=50),
                          np.random.multivariate_normal([2, 2], [[0.95, 0], [0, 1]], size=50)])
    
    X_4 = np.concatenate([np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 1]], size=50),
                          np.random.multivariate_normal([2, -2], [[0.95, 0], [0, 1]], size=50)])

    X = np.vstack((X_1, X_2, X_3, X_4))
    y = np.zeros(1000)
    y[400:900] = 1
    return X, y

def plot_artificial_binary_2d_2(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Synthetic Binary Classification Dataset with Spiraling Subgroups')
    plt.colorbar()
    plt.show()

### 4 ... spiral divided in half with binary classes

def construct_artificial_binary_2d_3():
    data = read_txt('D:\\FRI\\Diploma\\data\\spiral_data.txt')
    for ix, row in enumerate(data):
        data[ix] = data[ix].replace('\n', '').split(', ')
    X = np.array(data)[:,:2].astype(np.float64)
    y = np.array(data)[:,2].astype(np.int64)
    return X, y

def plot_artificial_binary_2d_3(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Synthetic Binary Classification Dataset with Spiraling Subgroups')
    plt.colorbar()
    plt.show()

### 5 ... age, income, hearth disease

def construct_hearth_disease_2d():
    data = read_txt('D:\\FRI\\Diploma\\data\\hearth_disease_new.txt')
    for ix, row in enumerate(data):
        data[ix] = data[ix].replace('\n', '').split(' ')
    X = np.array(data)[:,:2].astype(float)
    y = np.array(data)[:,2].astype(float)
    return X, y

def plot_hearth_disease_2d(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Hearth Disease Classification Dataset')
    plt.colorbar()
    plt.show()

### 6 ... on the left clean 0s, in the middle mixed 1s and 0s, on the right clean 1s

def construct_artificial_binary_2d_4():
    data = read_txt('D:\\FRI\\Diploma\\data\\artificial_data_1_new.txt')
    for ix, row in enumerate(data):
        data[ix] = data[ix].replace('\n', '').split(' ')
    X = np.array(data)[:,:2].astype(np.float64)
    y = np.array(data)[:,2].astype(np.float64)
    return X, y

def plot_artificial_binary_2d_4(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Synthetic Binary Classification Dataset Mixed and Clean Clusters')
    plt.colorbar()
    plt.show()

### 7 ... 50d from library only 5 informative 3 clusters per class

def construct_artificial_binary_5d_1():
    X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=5,
    n_classes=2,
    n_clusters_per_class=3,
    shuffle=False
    )
    return X, y

def plot2dPCA_artificial_binary_5d_1(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA - 2D: artificial binary 5d data')
    plt.show()

def plot2dUMAP_artificial_binary_5d_1(X, y):
    X_2d = UMAP(
               n_components=2, n_neighbors=200, min_dist=0
           ).fit_transform(X)
    
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP - 2D: artificial binary 5d data')
    plt.show()

### all concepts in one dataset 2d 4 classes

def construct_artificial_4c_2d_all():
    data = read_txt('D:\\FRI\\Diploma\\data\\artificial_data_all.txt')
    for ix, row in enumerate(data):
        data[ix] = data[ix].replace('\n', '').split(', ')
    X = np.array(data)[:,:2].astype(np.float64)
    y = np.array(data)[:,2].astype(np.int64)
    return X, y

def plot_artificial_4c_2d_all(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Synthetic Dataset 3 classes - multiconcepts')
    plt.colorbar()
    plt.show()

### density test dataset

def construct_density_binary_2d():
    data = read_txt('D:\\FRI\\Diploma\\data\\density_data.txt')
    for ix, row in enumerate(data):
        data[ix] = data[ix].replace('\n', '').split(', ')
    X = np.array(data)[:,:2].astype(np.float64)
    y = np.array(data)[:,2].astype(np.int64)
    return X, y

def plot_density_binary_2d(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Density Binary Dataset - 2D')
    plt.colorbar()
    plt.show()

### simple dataset

def construct_simple():
    data = read_txt('D:\\FRI\\Diploma\\data\\simple_data.txt')
    for ix, row in enumerate(data):
        data[ix] = data[ix].replace('\n', '').split(', ')
    X = np.array(data)[:,:2].astype(np.float64)
    y = np.array(data)[:,2].astype(np.int64)
    return X, y

def plot_simple(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Simple Binary Dataset - 2D')
    plt.colorbar()
    plt.show()

### MNIST

def construct_MNIST():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    #return np.concatenate((train_X, test_X), axis=0), np.concatenate((train_y, test_y), axis=0)
    return train_X.reshape(train_X.shape[0], -1), train_y

def plot_MNIST(X, y):
    # X_2d = UMAP().fit_transform(X)
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    # plt.xlabel('UMAP1')
    # plt.ylabel('UMAP2')
    # plt.title('UMAP - 2D: MNIST')
    # plt.show()

    tsne = TSNE(n_components=2, random_state=42)  # You can adjust n_components and other parameters
    X_embedded = tsne.fit_transform(X[:20000])
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y[:20000])
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.title('TSNE - 2D: MNIST')
    plt.show()

### KDD99

def construct_KDD99():
    # data = fetch_kddcup99()
    # data_df = pd.DataFrame(data.data, columns=data.feature_names)
    # data_df['target'] = data.target
    # data_df.drop_duplicates(keep='first', inplace = True)
    # retain_classes = [b'back.', b'satan.', b'warezclient.']
    # filtered_data_df = data_df[data_df['target'].isin(retain_classes)]
    # categorical_columns = ['protocol_type', 'service', 'flag']
    # data_df_encoded = pd.get_dummies(filtered_data_df, columns=categorical_columns)
    # X = data_df_encoded.drop(columns=['target']).values
    # y = data_df_encoded['target'].values
    # label_encoder = LabelEncoder()
    # y = label_encoder.fit_transform(y)
    # return X, y
    data = fetch_kddcup99()
    data_df = pd.DataFrame(data.data, columns=data.feature_names)
    data_df['target'] = data.target
    data_df.drop_duplicates(keep='first', inplace = True)
    categorical_columns = ['protocol_type', 'service', 'flag']
    data_df_encoded = pd.get_dummies(data_df, columns=categorical_columns)
    X = data_df_encoded.drop(columns=['target']).values
    y = data_df_encoded['target'].values
    random_indices = np.loadtxt("D:\\FRI\\Diploma\\random_indices4.txt", dtype=int)
    X_subset = X[random_indices]
    y_subset = y[random_indices]
    for ix in range(len(y_subset)):
        if y_subset[ix] == b'normal.':
            y_subset[ix] = 0
        else:
            y_subset[ix] = 1
    return X_subset.astype(float), y_subset.astype(int)


def plot_KDD99(X, y):
    # X_2d = UMAP().fit_transform(X)
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    # plt.xlabel('UMAP1')
    # plt.ylabel('UMAP2')
    # plt.title('UMAP - 2D: KDD99')
    # plt.show()

    tsne = TSNE(n_components=2, random_state=42)  # You can adjust n_components and other parameters
    X_embedded = tsne.fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
    plt.xlabel('TSNE1')
    plt.ylabel('TSNE2')
    plt.title('TSNE - 2D: KDD99')
    plt.show()

### DIABETES

def construct_DIABETES():
    url = "https://ndownloader.figshare.com/files/26759658"

    # Load the Tabula Muris bone marrow tissue dataset
    bone_marrow_dataset = sc.read_10x_h5(url)
    return 0


def plot_DIABETES(X, y):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # X_2d = UMAP().fit_transform(X)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP - 2D: KDD99')
    plt.show()

    # tsne = TSNE(n_components=2, random_state=42)  # You can adjust n_components and other parameters
    # X_embedded = tsne.fit_transform(X)
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
    # plt.xlabel('TSNE1')
    # plt.ylabel('TSNE2')
    # plt.title('TSNE - 2D: DIABETES')
    # plt.show()