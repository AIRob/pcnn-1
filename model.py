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
        model = tf.nn.dropout(model, keep_prob=0.9)
        w2 = tf.get_variable('w2', (32, 3))
        b2 = tf.get_variable('b2', (3))
        l2_loss += tf.nn.l2_loss(w2)
        l2_loss += tf.nn.l2_loss(b2)
        model = tf.nn.xw_plus_b(model, w2, b2)
        model = tf.nn.dropout(model, keep_prob=0.9)
        model = tf.nn.softmax(model)
    
    return model, l2_loss

def pcnn1(input_, scope='PCNN'):
    with tf.variable_scope(scope):
        input_ = tf.expand_dims(input_, 3)
        l2_loss = tf.constant(0.0)
        model = tf.layers.conv2d(input_, 16, (4, 40))
        model = tf.layers.batch_normalization(model)
        model = tf.nn.relu(model)
        model = tf.squeeze(model, 2)
        model = tf.layers.conv1d(model, 16, (4, ))
        model = tf.layers.batch_normalization(model)
        model = tf.nn.relu(model)
        model = tf.layers.max_pooling1d(model, 50, 1)
        model = tf.layers.conv1d(model, 32, (3, ))
        model = tf.layers.batch_normalization(model)
        model = tf.nn.relu(model)
        model = tf.layers.conv1d(model, 32, (3, ))
        model = tf.layers.batch_normalization(model)
        model = tf.nn.relu(model)
        model = tf.layers.max_pooling1d(model, 25, 1)
        shape = model.get_shape().as_list()
        dim = np.prod(shape[1:])
        model = tf.reshape(model, (-1, dim))
        model = tf.layers.dense(model, 32)
        model = tf.layers.dropout(model)
        model = tf.layers.dense(model, 3)
        model = tf.layers.dropout(model)
        model = tf.nn.softmax(model)
        
    return model, tf.constant(0.0)

def loss(X, y):
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        model, l2_loss = pcnn(X)
        
    with tf.variable_scope('loss'):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=model, 
                                                         labels=y)
        losses = tf.reduce_mean(losses) + 0.1 * l2_loss
    
    with tf.variable_scope('accuracy'):
        correct_predictions = tf.equal(tf.argmax(model, 1),
                                       tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'),
                                  name='accuracy')
    
    return losses, accuracy

def loss1(X, y):
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        model, l2_loss = pcnn1(X)
        
    with tf.variable_scope('loss'):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=model, 
                                                         labels=y)
        losses = tf.reduce_mean(losses) + 0.1 * l2_loss
    
    with tf.variable_scope('accuracy'):
        correct_predictions = tf.equal(tf.argmax(model, 1),
                                       tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'),
                                  name='accuracy')
    
    return losses, accuracy


def test_pcnn():
    input_ = tf.placeholder(tf.float32, shape=[None, 100, 40], name='input')
    model = pcnn(input_)
    
def test_pcnn1():
    input_ = tf.placeholder(tf.float32, shape=[None, 100, 40], name='input')
    model = pcnn1(input_)


if __name__ == '__main__':
    #----------------test_ctabl-----------------------------------------
    test_pcnn1()
    with tf.Session() as sess:
        tf.summary.scalar('fake', 0)
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./tmp', graph=sess.graph)
        writer.add_summary(sess.run(summary))
        writer.flush()
    print('end')