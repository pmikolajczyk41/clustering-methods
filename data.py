import numpy as np

DATASETS_2D = ['data/dane_2D_1.txt', 'data/dane_2D_2.txt', 'data/dane_2D_3.txt', 'data/dane_2D_4.txt',
               'data/dane_2D_5.txt', 'data/dane_2D_6.txt', 'data/dane_2D_7.txt', 'data/dane_2D_8.txt']

DATASETS_2D_Ks = [10, 10, 5, 5, 37, 17, 5, 5]

from sklearn.preprocessing import MinMaxScaler


def get_data_w_labels(filename):
    data = np.loadtxt(filename)
    x, y = np.hsplit(data, [-1])
    return MinMaxScaler().fit(x).transform(x), y


def get_data_wo_labels(filename):
    data = np.loadtxt(filename)
    return MinMaxScaler().fit(data).transform(data)
