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


def main():
    data_class = Data('artificial_binary_2d_1')
    data_class.construct()
    data_class.plot()


if __name__ == "__main__":
    main()