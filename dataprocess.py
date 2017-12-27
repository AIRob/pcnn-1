#encoding:UTF-8

import os
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

DEBUG = False

def cal_sum(args):
    print('here')

def load_csv(path, window_size=100):
    print('load csv...')
    points = [10, 20, 30, 50, 100]
    points_list = [[10, -90, 20, -80],
                   [20, -80, 30, -70],
                   [30, -70, 50, -50],
                   [50, -50, 100, None]]
    arr = pd.read_csv(path)
    arr = np.array(arr)
    p = np.sum(arr[:, [6, 12]], axis=1)/2.0
    culsum = np.convolve(p, np.ones(window_size, dtype=np.int), 'valid')
    labels = np.zeros(p.shape[0])
    #前100(window size)个不参与计算
    labels[:window_size-1] = -2
    #后100个不参与计算
    labels[-100:] = -2
    label_up   = np.zeros((len(culsum)-100, 4))
    label_down = np.zeros((len(culsum)-100, 4))
    for i, j in enumerate(points_list):
        label_up[:, i] = culsum[j[0]:j[1]] <= culsum[j[2]:j[3]]
        label_down[:, i] = culsum[j[0]:j[1]] >= culsum[j[2]:j[3]]
    
    labels[window_size-1:-100][np.all(label_up, axis=1)] = 1
    labels[window_size-1:-100][np.all(label_down, axis=-1)] = -1
    #print('end')
    
    #----------------------test----------------------------------
    if DEBUG:
        for i in range(len(p)):
            if i < window_size - 1:
                assert(labels[i]==-2)
            elif i >= len(p) - 100:
                assert(labels[i]==-2)
            else:
                sum_10 = np.sum(p[i-90+1:i+10+1])
                sum_20 = np.sum(p[i-80+1:i+20+1])
                sum_30 = np.sum(p[i-70+1:i+30+1])
                sum_50 = np.sum(p[i-50+1:i+50+1])
                sum_100 = np.sum(p[i+1:i+100+1])
                if sum_10 <= sum_20 <= sum_30 <= sum_50 <= sum_100:
                    assert(labels[i]==1)
                elif sum_10 >= sum_20 >= sum_30 >= sum_50 >= sum_100:
                    assert(labels[i]==-1)
                else:
                    assert(labels[i]==0)
    
    return p, labels

def data_generator(path, window_size=100, batch_size=32):
    X, y = load_csv(path)
    up = np.where(y==1)[0]
    down = np.where(y==-1)[0]
    nodir = np.where(y==0)[0]
    while True:
        ret_X = np.zeros((batch_size, window_size), dtype=np.float32)
        ret_y = np.zeros((batch_size, 1), dtype=np.int)
        for i in range(batch_size):
            r = np.random.rand()
            if r <= 0.33:
                index = np.random.choice(up)
            elif r <= 0.66:
                index = np.random.choice(down)
            else:
                index = np.random.choice(nodir)
            ret_X[i] = X[index-100:index]
            ret_y[i] = y[index]
        yield ret_X, ret_y
                    
    
if __name__ == '__main__':
    #load_csv('data/data/20140102.csv')
    datagen = data_generator('data/data/20140102.csv')
    for i in itertools.count():
        x, y = next(datagen)