#encoding:UTF-8

import time
import itertools
import numpy as np
import tensorflow as tf
import model_conv1d
import dataprocess


tf.app.flags.DEFINE_string('name', 'tabl', '')
tf.app.flags.DEFINE_float('learning_rate', 0.01, '')
tf.app.flags.DEFINE_string('checkpoint_path', 'model/', '')
tf.app.flags.DEFINE_integer('max_steps', 1000000, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_boolean('restore', False, '')
tf.app.flags.DEFINE_integer('summary_step', 1000, '')

FLAGS = tf.app.flags.FLAGS

DEBUG = True

def initial_w_lamda_b(tvars, sess):
    for i in tvars:
        if 'w:0' in i.name:
            value = np.ones(i.shape) / float(i.shape[0].value)
            sess.run(i.assign(value))
        elif 'lambda_:0' in i.name:
            sess.run(i.assign([0.5]))
        elif 'b:0' in i.name:
            sess.run(i.assign(np.zeros(i.shape)))

def main(argv=None):
    inputs = tf.placeholder(tf.float32, shape=[None, 100, 1], name='inputs')
    labels = tf.placeholder(tf.float32, shape=[None, 3], name='labels')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                  trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                               global_step,
                                               decay_steps=20000,
                                               decay_rate=0.94,
                                               staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    #opt = tf.train.AdamOptimizer(learning_rate)
    loss, accuracy = model_conv1d.loss(inputs, labels)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    tvars = tf.trainable_variables()
    #grads = opt.compute_gradients(loss, tvars)
    #grads = tf.gradients(loss, tvars)
    #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.0)
    #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    #with tf.control_dependencies([apply_gradient_op]):
        #train_op = tf.no_op(name='train_op')
    #train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
    saver = tf.train.Saver(tf.global_variables())
    print('Loading data...')
    data_generator = dataprocess.data_generator(path='data/data/20140102.csv', batch_size=256)
    start = time.time()
    with tf.Session() as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(tf.global_variables_initializer())
            #initial_w_lamda_b(tvars, sess)
        #for step in range(FLAGS.max_steps):
        if DEBUG:
            cur_weights = {}
            old_weights = {}
        for step in itertools.count():
            X, y = next(data_generator)
            y = y.astype(np.float32)
            X = np.expand_dims(X, 2)
            tl, ac, _ = sess.run([loss, accuracy, train_op], feed_dict={inputs:X, labels:y})
            if step % 100 == 0:
                avg_time_per_step = (time.time() - start) / 100
                start = time.time()
                #print('Step {:06d}, loss {:.4f}, acc {:g} {:.2f} seconds/step'.format(
                    #step, tl, ac, avg_time_per_step))
            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)
            if step % FLAGS.summary_step == 0:
                lr = sess.run(learning_rate)
                print('learning rate is %.6f'%lr)
            
            if DEBUG:
                for var in tvars:
                    cur_weights[var] = sess.run(var)
                if old_weights:
                    for var in cur_weights.keys():
                        diff = cur_weights[var]-old_weights[var]
                        sum_ = diff.sum()
                        variance = diff.var()
                        print('var name : {0: ^70} sum ：{1:.3} var : {2:.3}'.format(var.name, sum_, variance))
                    print('#'*70)
                    print('\n')
                for key, value in cur_weights.items():
                    old_weights[key] = value
            
            
if __name__ == '__main__':
    tf.app.run()