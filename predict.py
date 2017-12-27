#encoding:UTF-8

import time
import itertools
import numpy as np
import tensorflow as tf
import model
import datagen


tf.app.flags.DEFINE_string('name', 'tabl', '')
tf.app.flags.DEFINE_float('learning_rate', 0.1, '')
tf.app.flags.DEFINE_string('checkpoint_path', 'model/', '')
tf.app.flags.DEFINE_integer('max_steps', 1000000, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_boolean('restore', True, '')
tf.app.flags.DEFINE_integer('summary_step', 1000, '')

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    inputs = tf.placeholder(tf.float32, shape=[None, 100, 40], name='inputs')
    labels = tf.placeholder(tf.float32, shape=[None, 3], name='labels')
    m, _ = model.pcnn1(inputs)
    saver = tf.train.Saver(tf.global_variables())
    print('Loading data...')
    data_generator = datagen.generator(batch_size=256)
    start = time.time()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(tf.global_variables_initializer())
        #for step in range(FLAGS.max_steps):
        for step in itertools.count():
            X, y = next(data_generator)
            y = y[:, 0, :]
            y = y.reshape((y.shape[0], 3))
            result = sess.run(m, feed_dict={inputs:X, labels:y})
            
            
if __name__ == '__main__':
    tf.app.run()