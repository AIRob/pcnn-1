#encoding:UTF-8

import os
import math
import time
import timeit
import itertools
import numpy as np
from numpy.random import RandomState
import pandas as pd
import scipy as sp
import scipy.stats
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DEBUG = False

def get_primes():
    D = {}
    q = 2
    while True:
        if q not in D:
            yield q
            D[q*q] = [q]
        else:
            for p in D[q]:
                D.setdefault(p+q, []).append(p)
            del D[q]
        q += 1
        
        
def is_prime(n):
    if n % 2 == 0 and n > 2: 
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


def plot_surface(Z, title='title'):
    minz = np.min(Z)
    maxz = np.max(Z)
    Z = (Z - minz)/(maxz - minz)
    X = np.arange(Z.shape[0])
    Y = np.arange(Z.shape[1])
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf)
    plt.show()
    
    
def get_prime_array(prime, w):
    tmp = np.array(range(w), dtype=np.int)
    tmp *= prime
    ret = np.zeros(w, dtype=np.int)
    ret[:-int(w/2)] = tmp[::2][::-1]
    ret[-int(w/2):] = tmp[1::2]
    return ret
    

def get_prime_matrix(arr, w=227):
    arr = np.array(arr)
    if w==1:
        return arr[0]
    else:
        if w%2 == 0:
            w += 1
        # get prime number
        prime_gen = get_primes()
        primes = [1]
        for i in range(w-1):
            primes.append(next(prime_gen))
        # 尾数补齐    
        if w * primes[-1] > len(arr):
            arr = np.concatenate([arr, np.tile(arr[-1], w * primes[-1]-len(arr))])
        ret = np.zeros((w, w))
        last_ret = get_prime_matrix(arr, w-2)
        ret[1:-1, 1:-1] = last_ret
        ret[0] = arr[get_prime_array(primes[-2], w)]
        ret[-1] = arr[get_prime_array(primes[-1], w)]
        if w > 3:
            if w == 5:
                ret[1:-1, -1] = 3 * last_ret[:, -1] - 2 * last_ret[:, -2]
            else:
                ret[1:-1, -1] = 2 * last_ret[:, -1] - last_ret[:, -2]
            ret[1:-1, 0] = 2 * last_ret[:, 0] - last_ret[:, 1]
        else:
            ret[1:-1, 0] = [2]
            ret[1:-1, -1] = [1]
        return ret

            
def get_wrapped_rope(arr, w=227):
    if w <= 0:
        raise ValueError('the width of wrapped rope must be greater than 0!')
    if w == 1:
        return arr[0]
    else:
        arr = np.array(arr[::-1])
        if w%2 == 0:
            w += 1
        if w**2 > len(arr):
            arr = np.concatenate([arr, np.tile(arr[-1], w**2-len(arr))])
        wrapped_rope = np.zeros((w, w))
        wrapped_rope[1:-1, 1:-1] = get_wrapped_rope(arr, w-2)
        start = (w-2)**2
        end = start + w - 1
        wrapped_rope[1:, -1] = arr[start:end]
        start = end
        end = start + w - 1
        wrapped_rope[-1, :-1] = arr[start:end][::-1]
        start = end
        end = start + w - 1
        wrapped_rope[:-1, 0] = arr[start:end][::-1]
        start = end
        end = start + w - 1
        wrapped_rope[0, 1:] = arr[start:end]
    return wrapped_rope
    

def get_rectancle_img(x):
    batch_size, length = x.shape
    ret_x = None
    for i in range(1, length+1):
        sub_length = 2*length + 1 - 2*i
        sub_x = np.tile(x[:, [-i]], (1, sub_length*sub_length))
        sub_x = sub_x.reshape((batch_size, sub_length, sub_length, 1))        
        if ret_x is None:
            ret_x = sub_x
        else:
            ret_x[:, (i-1):-(i-1), (i-1):-(i-1), :] = sub_x
    return ret_x
        

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
                sum_10 = np.sum(p[i-window_size+10+1:i+10+1])
                sum_20 = np.sum(p[i-window_size+20+1:i+20+1])
                sum_30 = np.sum(p[i-window_size+30+1:i+30+1])
                sum_50 = np.sum(p[i-window_size+50+1:i+50+1])
                sum_100 = np.sum(p[i-window_size+100+1:i+100+1])
                if sum_10 <= sum_20 <= sum_30 <= sum_50 <= sum_100:
                    assert(labels[i]==1)
                elif sum_10 >= sum_20 >= sum_30 >= sum_50 >= sum_100:
                    assert(labels[i]==-1)
                else:
                    assert(labels[i]==0)
    
    return p, labels
        

def load_random(path, window_size=100, random_size=1e6):
    print('load csv...')
    points = [10, 20, 30, 50, 100]
    points_list = [[10, -90, 20, -80],
                   [20, -80, 30, -70],
                   [30, -70, 50, -50],
                   [50, -50, 100, None]]
    random_ = RandomState(1234567890)
    arr = pd.Series(random_.randn(int(random_size)))
    #arr = pd.Series(np.random.randn(int(random_size)))
    arr = arr.cumsum()
    arr = np.array(arr)
    p = arr
    culsum = np.convolve(p, np.ones(100, dtype=np.int), 'valid')
    labels = np.zeros(p.shape[0])
    #前100(window size)个不参与计算
    labels[:window_size-1] = -2
    #后100个不参与计算
    labels[-100:] = -2
    label_up   = np.zeros((len(culsum)-100, 4))
    label_down = np.zeros((len(culsum)-100, 4))
    sum_diff = np.zeros((len(culsum)-100, 4))
    #求取置信区间
    for i, j in enumerate(points_list):
        sum_diff[:, i] = culsum[j[0]:j[1]] - culsum[j[2]:j[3]]
    a = sum_diff.reshape(-1)
    mean, sigma = np.mean(a), np.std(a)
    conf_int = sp.stats.norm.interval(0.68, loc=mean, scale=sigma)
    for i, j in enumerate(points_list):
        label_up[:, i] = culsum[j[0]:j[1]] <= culsum[j[2]:j[3]] - conf_int[1]
        label_down[:, i] = culsum[j[0]:j[1]] >= culsum[j[2]:j[3]] - conf_int[0]
        
    
    labels[window_size-1:-100][np.all(label_up[window_size-100:], axis=1)] = 1
    labels[window_size-1:-100][np.all(label_down[window_size-100:], axis=-1)] = -1
    #print('end')
    
    #----------------------test----------------------------------
    if DEBUG:
        for i in range(len(p)):
            if i < window_size - 1:
                assert(labels[i]==-2)
            elif i >= len(p) - 100:
                assert(labels[i]==-2)
            else:
                sum_10 = np.sum(p[i-100+10+1:i+10+1])
                sum_20 = np.sum(p[i-100+20+1:i+20+1])
                sum_30 = np.sum(p[i-100+30+1:i+30+1])
                sum_50 = np.sum(p[i-100+50+1:i+50+1])
                sum_100 = np.sum(p[i-100+100+1:i+100+1])
                if (sum_10 <= sum_20 - conf_int[1]) and\
                   (sum_20 <= sum_30 - conf_int[1]) and\
                   (sum_30 <= sum_50 - conf_int[1]) and\
                   (sum_50 <= sum_100 - conf_int[1]):
                    assert(labels[i]==1)
                elif (sum_10 >= sum_20 - conf_int[0]) and\
                     (sum_20 >= sum_30 - conf_int[0]) and\
                     (sum_30 >= sum_50 - conf_int[0]) and\
                     (sum_50 >= sum_100 - conf_int[0]):
                    assert(labels[i]==-1)
                else:
                    assert(labels[i]==0)
    
    return p, labels

def data_generator(path, window_size=100, random_size=1e6, batch_size=32, use_np=True, rand=True, mode='rope'):
    if rand:
        X, y = load_random(path, window_size, random_size)
    else:
        X, y = load_csv(path, window_size)
    up = np.where(y==1)[0]
    down = np.where(y==-1)[0]
    nodir = np.where(y==0)[0]
    if mode != 'rope' and mode != 'prime_v0':
        w = int(np.sqrt(window_size))
        prime_mat = get_prime_matrix(range(330000), w).astype(np.int)
        prime_mat[prime_mat>=window_size] = window_size - 1
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
            ret_X[i] = X[index-window_size+1:index+1]
            ret_y[i] = y[index]
        ret_y = ret_y + 1
        ret_y = ret_y.reshape(-1)
        ret_y = np.eye(3)[ret_y]
        if use_np:
            if mode == 'rope':
                ret_X = np.apply_along_axis(get_wrapped_rope, 1, ret_X)
            elif mode == 'prime_v0':
                ret_X = np.apply_along_axis(get_prime_matrix, 1, ret_X)
            else:
                ret_X = ret_X[:, prime_mat.reshape(-1)].reshape(batch_size, w, w)
        yield np.tile(np.expand_dims(ret_X, -1), (1, 1, 1, 3)), ret_y
        
def test_get_rectancle_img():
    for i in range(1, 10):
        x = np.array([range(i)])
        b = get_rectancle_img(x)
        b = np.squeeze(b)
        print(b)
        

def test_get_wrapped_rope():
    arr = range(26)
    for i in range(1, 10):
        print(get_wrapped_rope(arr, i))

def test_get_wrapped_rope1():
    arr = np.random.rand(227*227)
    get_wrapped_rope(arr, 227)
    

def test_get_prime_array():
    print(get_prime_array(5, 11))
    print(get_prime_array(6, 11))
    
    
def test_get_prime_matrix():
    arr = range(227)
    for i in range(1, 228, 2):
        print(get_prime_matrix(arr, i))
    
    
def test_get_prime_matrix1():
    arr = range(227)
    get_prime_matrix(arr, 227)
                    
    
if __name__ == '__main__':
    #load_csv('data/data/20140102.csv')
    #test_get_rectancle_img()
    test_get_prime_array()
    test_get_prime_matrix()
    #--------prime matrix time----------------------------
    start = time.time()
    for i in range(100):
        test_get_prime_matrix1()
    end = time.time()
    print('total time is ', end-start)
    print('average time is ', (end-start)/100)
    #--------prime matrix time 1----------------------------
    start = time.time()
    datagen = data_generator('', batch_size=1, window_size=227*227, mode='prime_v0')
    for i in range(100):
        x, y = next(datagen)
    end = time.time()
    print('total time is ', end-start)
    print('average time is ', (end-start)/100)
    #--------prime matrix time 2----------------------------
    start = time.time()
    datagen = data_generator('', window_size=227*227, mode='prime_v1')
    for i in range(100):
        x, y = next(datagen)
    end = time.time()
    print('total time is ', end-start)
    print('average time is ', (end-start)/100)
    test_get_wrapped_rope()    
    #--------rope matrix time-----------------------------
    start = time.time()
    for i in range(100):
        test_get_wrapped_rope1()
    end = time.time()
    print('total time is ', end-start)
    print('average time is ', (end-start)/100)
    #timeit.Timer('for i in range(3): test_get_wrapped_rope1()', "from __main__ import test_get_wrapped_rope1").timeit()
    #datagen = data_generator('data/data/20140102.csv', window_size=227*227)
    datagen = data_generator('', window_size=227*227)
    for i in itertools.count():
        x, y = next(datagen)