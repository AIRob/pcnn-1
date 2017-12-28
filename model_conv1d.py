#encoding:UTF-8

import os
import numpy as np
import tensorflow as tf


def pcnn(input_, scope='PCNN'):
    with tf.variable_scope(scope):
        l2_loss = tf.constant(0.0)
        #model = tf.layers.conv1d(input_, 32, (4, ))
        #model = tf.layers.batch_normalization(model)
        #model = tf.nn.relu(model)
        #model = tf.layers.conv1d(model, 64, (4, ))
        #model = tf.layers.batch_normalization(model)
        #model = tf.nn.relu(model)
        #model = tf.layers.max_pooling1d(model, 2, 1)
        #model = tf.layers.conv1d(model, 128, (3, ))
        #model = tf.layers.batch_normalization(model)
        #model = tf.nn.relu(model)
        #model = tf.layers.conv1d(model, 256, (3, ))
        #model = tf.layers.batch_normalization(model)
        #model = tf.nn.relu(model)
        #model = tf.layers.max_pooling1d(model, 2, 1)
        model = input_
        shape = model.get_shape().as_list()
        dim = np.prod(shape[1:])
        model = tf.reshape(model, (-1, dim))
        #model = tf.layers.dense(model, 4096)
        #model = tf.layers.dropout(model)
        #model = tf.layers.dense(model, 1000)
        #model = tf.layers.dropout(model)
        #model = tf.layers.dense(model, 3)
        #model = tf.layers.dropout(model)
        w1 = tf.Variable(tf.random_normal([dim, 4096]), name='w1')
        b1 = tf.Variable(tf.random_normal([4096]), name='b1')
        model = tf.matmul(model, w1) + b1
        w2 = tf.Variable(tf.random_normal([4096, 3]), name='w2')
        b2 = tf.Variable(tf.random_normal([3]), name='b2')
        model = tf.matmul(model, w2) + b2
        model = tf.nn.softmax(model)
        
    return model, l2_loss


def calculate_loss(y_true, y_pred):
    with tf.variable_scope('loss'):
        c = tf.constant(1e6)
        y_pred = tf.maximum(y_pred, 1e-5)
        loss = tf.log(y_pred)
        loss = tf.multiply(y_true, loss)
        a = c/tf.cast(tf.shape(y_true)[0], tf.float32)
        loss = -tf.multiply(a, loss)
        loss = tf.reduce_sum(loss)
        return loss

def loss(X, y):
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        model, l2_loss = pcnn(X)
        
    with tf.variable_scope('loss'):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=model, 
                                                         labels=y)
        losses = tf.reduce_mean(losses)
    
    #losses = calculate_loss(y, model)
    
    with tf.variable_scope('accuracy'):
        correct_predictions = tf.equal(tf.argmax(model, 1),
                                       tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'),
                                  name='accuracy')
    
    return losses, accuracy

    
def test_pcnn1():
    input_ = tf.placeholder(tf.float32, shape=[None, 100, 40], name='input')
    model = pcnn(input_)


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