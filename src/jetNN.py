#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from src.util import *

def train_jet(gp, ip, x_train, y_train, x_test, y_test):
    """


    """

    #print x_train.shape
    #print y_train.shape

    #print x_test.shape
    #print y_test.shape

    print "dropout:", ip.dropout
    # Input/output layer dimensions
    n_x = x_train.shape[0]
    n_y = y_train.shape[0]

    #--> set random seed: only for testing purposes - TO BE REMOVED
    tf.set_random_seed(1)

    sess = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=8))

    # Input placeholders
    with tf.name_scope('input'):
        x  = tf.placeholder(tf.float32, [n_x, None], name='x-input')
        y_ = tf.placeholder(tf.float32, [n_y, None], name='y-input')

    # First FC hidden layer with 200 units
    hidden1 = nn_layer(x, 200, n_x, 'layer1')

    # Second FC hidden layer with 50 units
    hidden2 = nn_layer(hidden1, 50, 200, 'layer2')

    # Dropout layer
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden2, keep_prob)

    # Do not apply softmax activation yet, see below.
    y = nn_layer(dropped, n_y, 50, 'layer3', act=tf.identity)
    #y = nn_final_layer(dropped, n_y, 50, 'layer3')

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.transpose(y_), logits=tf.transpose(y))
        #diff = tf.nn.weighted_cross_entropy_with_logits(targets=tf.transpose(y_), logits=tf.transpose(y), pos_weight=10.0)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        if ip.optimizer == 'Adam':
            train_step = tf.train.AdamOptimizer(ip.learning_rate).minimize(cross_entropy)
        elif ip.optimizer == 'Adagrad':
            train_step = tf.train.AdagradOptimizer(ip.learning_rate).minimize(cross_entropy)
        elif ip.optimizer == 'GraDe':
            train_step = tf.train.GradientDescentOptimizer(ip.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('y_prob'):
            y_prob = tf.sigmoid(y)
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.round(y_prob), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(ip.log_dir + '/train/'+ip.run_dir, sess.graph)
    devel_writer = tf.summary.FileWriter(ip.log_dir + '/devel/'+ip.run_dir)
    tf.global_variables_initializer().run()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    def feed_dict(gp, train):
        """ Make a TensorFlow feed_dict: maps data onto Tensor placeholders. """
        if train:
            # use mini-batches of 128 events:
            xs, ys = next_batch(gp, 128, x_train, y_train)
            # use the full batch of training events:
            #xs, ys = x_train , y_train
            # dropout value
            k = ip.dropout
        else:
            xs, ys = x_test, y_test
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    for i in range(ip.max_steps+1):
        if i % 100 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(gp, False))
            devel_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))

        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(gp, True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Train
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(gp,True))
                train_writer.add_summary(summary, i)

        if i==ip.max_steps:
            summary, acc, diff, y_prob, y_ = sess.run([merged, accuracy, diff, y_prob, y_], feed_dict=feed_dict(gp, False))

            # save model
            save_path = saver.save(sess, ip.log_dir + '/models/'+ip.run_dir)
            print("Model saved in path: %s" % save_path)

    train_writer.close()
    devel_writer.close()

    return y_prob
