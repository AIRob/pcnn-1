#encoding:UTF-8

import os
import numpy as np
import tensorflow as tf


def pcnn1(input_, scope='PCNN'):
    with tf.variable_scope(scope):
        l2_loss = tf.constant(0.0)
        model = tf.layers.conv1d(input_, 16, (4, ))
        model = tf.layers.batch_normalization(model)
        model = tf.nn.relu(model)
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