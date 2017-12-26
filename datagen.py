#encoding:UTF-8

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.app.flags.DEFINE_string('training_data_path', 'data/data/', 'training dataset to use!')

FLAGS = tf.app.flags.FLAGS

def load_txt(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    lines = [i.split() for i in lines]
    lines = np.array(lines)
    lines = lines.astype(np.float32)
    return lines

def get_files():
    files = os.listdir(FLAGS.training_data_path)
    return [FLAGS.training_data_path+i for i in files]

def generator(input_size=100, batch_size=32):
    files = get_files()
    arr = load_txt(files[0])
    length = 10000
    arr = arr[:, 30000:40000]
    while True:
        try:
            X = np.zeros((batch_size, input_size, 40), dtype=np.float32)
            y = np.zeros((batch_size, 5), dtype=np.float32)
            seletecd_x = np.random.choice(range(length-input_size), size=[batch_size])
            seletecd_y = [i+input_size for i in seletecd_x]
            for i in range(batch_size):
                a = arr[:40, range(seletecd_x[i], seletecd_x[i]+input_size)]
                X[i] = np.transpose(a, [1, 0])
                y[i] = arr[-5:, seletecd_y[i]]
            y = y.astype(np.int)
            y = y.reshape(-1)
            y = np.eye(3)[y-1]
            y = y.reshape((batch_size, 5, -1))
            yield X, y
        except:
            print('error occur!')
    
    
def main():
    generator()
    
if __name__ == '__main__':
    main()