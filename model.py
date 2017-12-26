#encoding:UTF-8

import os
import numpy as np
import tensorflow as tf


def pcnn(input_, scope='PCNN'):
    with tf.variable_scope(scope):
        input_ = tf.expand_dims(input_, 3)
        l2_loss = tf.constant(0.0)
        filter1 = tf.Variable(tf.random_normal([1, 40, 1, 16]))
        model = tf.nn.conv2d(input_, filter1, [1, 1, 1, 1], padding='VALID', name='conv2')
        model = tf.squeeze(model, 2)
        filter2 = tf.Variable(tf.random_normal([4, 16, 16]))
        model = tf.nn.conv1d(model, filter2, 1, padding='VALID', name='conv1_0')
        model = tf.expand_dims(model, 2)
        model = tf.nn.max_pool(model, [1, 48, 1, 1], [1, 1, 1, 1], padding='VALID')
        model = tf.squeeze(model, 2)
        filter3 = tf.Variable(tf.random_normal([3, 16, 16]))
        model = tf.nn.conv1d(model, filter3, 1, padding='VALID')
        filter4 = tf.Variable(tf.random_normal([3, 16, 16]))
        model = tf.nn.conv1d(model, filter4, 1, padding='VALID')
        model = tf.expand_dims(model, 2)
        model = tf.nn.max_pool(model, [1, 25, 1, 1], [1, 1, 1, 1], padding='VALID')
        model = tf.reshape(model, (-1, 22 * 16))
        w1 = tf.get_variable('w1', (22*16, 32))
        b1 = tf.get_variable('b1', (32))
        l2_loss += tf.nn.l2_loss(w1)
        l2_loss += tf.nn.l2_loss(b1)
        model = tf.nn.xw_plus_b(model, w1, b1)
        w2 = tf.get_variable('w2', (32, 3))
        b2 = tf.get_variable('b2', (3))
        l2_loss += tf.nn.l2_loss(w2)
        l2_loss += tf.nn.l2_loss(b2)
        model = tf.nn.xw_plus_b(model, w2, b2)
    
    return model, l2_loss

def loss(X, y):
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        model, l2_loss = pcnn(X)
        
    with tf.variable_scope('loss'):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=model, 
                                                         labels=y)
        losses = tf.reduce_mean(losses) + 0.1 * l2_loss
    
    return losses

def test_pcnn():
    input_ = tf.placeholder(tf.float32, shape=[None, 100, 40], name='input')
    model = pcnn(input_)


if __name__ == '__main__':
    #----------------test_ctabl-----------------------------------------
    test_pcnn()
    with tf.Session() as sess:
        tf.summary.scalar('fake', 0)
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./tmp', graph=sess.graph)
        writer.add_summary(sess.run(summary))
        writer.flush()
    print('end')