import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
from sklearn.decomposition import PCA

class Data:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = None
        self.y = None
        self.DATASET1 = 'artificial_binary_3d_1'
        self.DATASET2 = 'artificial_binary_2d_1'
        self.DATASET3 = 'artificial_binary_2d_2'
        self.DATASET4 = 'artificial_binary_2d_3'
        self.DATASET5 = 'hearth_disease_2d'
        self.DATASET6 = 'artificial_binary_2d_4'
        self.DATASET7 = 'artificial_binary_5d_1'
        self.DATASET8 = 'artificial_4c_2d_all'
        self.DATASET9 = 'density_binary_2d'
        self.DATASET10 = 'simple_data'
        self.DATASET11 = 'MNIST'
        self.DATASET12 = 'KDD99'
        self.DATASET13 = 'DIABETES'

    def construct(self, n_samples = 100):
        if self.dataset == self.DATASET1:
            self.X, self.y = hf.construct_artificial_binary_3d_1()
        if self.dataset == self.DATASET2:
            self.X, self.y = hf.construct_artificial_binary_2d_1()
        if self.dataset == self.DATASET3:
            self.X, self.y = hf.construct_artificial_binary_2d_2()
        if self.dataset == self.DATASET4:
            self.X, self.y = hf.construct_artificial_binary_2d_3()
        if self.dataset == self.DATASET5:
            self.X, self.y = hf.construct_hearth_disease_2d()
        if self.dataset == self.DATASET6:
            self.X, self.y = hf.construct_artificial_binary_2d_4()
        if self.dataset == self.DATASET7:
            self.X, self.y = hf.construct_artificial_binary_5d_1()
        if self.dataset == self.DATASET8:
            self.X, self.y = hf.construct_artificial_4c_2d_all()
        if self.dataset == self.DATASET9:
            self.X, self.y = hf.construct_density_binary_2d()
        if self.dataset == self.DATASET10:
            self.X, self.y = hf.construct_simple()
        if self.dataset == self.DATASET11:
            self.X, self.y = hf.construct_MNIST()
        if self.dataset == self.DATASET12:
            self.X, self.y = hf.construct_KDD99()
        if self.dataset == self.DATASET13:
            self.X, self.y = hf.construct_DIABETES()

    def plot(self):
        if self.dataset == self.DATASET1:
            hf.plot2dORIG_artificial_binary_3d_1(self.X, self.y)
            hf.plot2dPCA_artificial_binary_3d_1(self.X, self.y)
        if self.dataset == self.DATASET2:
            hf.plot_artificial_binary_2d_1(self.X, self.y)
        if self.dataset == self.DATASET3:
            hf.plot_artificial_binary_2d_2(self.X, self.y)
        if self.dataset == self.DATASET4:
            hf.plot_artificial_binary_2d_3(self.X, self.y)
        if self.dataset == self.DATASET5:
            hf.plot_hearth_disease_2d(self.X, self.y)
        if self.dataset == self.DATASET6:
            hf.plot_artificial_binary_2d_4(self.X, self.y)
        if self.dataset == self.DATASET7:
            hf.plot2dPCA_artificial_binary_5d_1(self.X, self.y)
            hf.plot2dUMAP_artificial_binary_5d_1(self.X, self.y)
        if self.dataset == self.DATASET8:
            hf.plot_artificial_4c_2d_all(self.X, self.y)
        if self.dataset == self.DATASET9:
            hf.plot_density_binary_2d(self.X, self.y)
        if self.dataset == self.DATASET10:
            hf.plot_simple(self.X, self.y)
        if self.dataset == self.DATASET11:
            hf.plot_MNIST(self.X, self.y)
        if self.dataset == self.DATASET12:
            hf.plot_KDD99(self.X, self.y)
        if self.dataset == self.DATASET13:
            hf.plot_DIABETES(self.X, self.y)

def main():
    data_class = Data('simple_data')
    data_class.construct()
    data_class.plot()


if __name__ == "__main__":
    main()