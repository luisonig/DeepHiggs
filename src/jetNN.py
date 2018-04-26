#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from src.util import *

def train_jet(gp, ip, x_train, y_train, x_test, y_test):
    """
    

    """

    #print x_train.shape
    #print y_train.shape

    # print x_test.shape
    # print y_test.shape

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
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)), reduction_indices=[1]))
        #
        # can be numerically unstable.So here we use
        # tf.nn.softmax_cross_entropy_with_logits on the raw outputs
        # of the nn_layer above, and then average across the batch.
        # --> will have to be substituted with tf.nn.softmax_cross_entropy_with_logits_v2 (see tf online manual)
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(y_), logits=tf.transpose(y))
        #diff = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.transpose(y_), logits=tf.transpose(y))
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
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, axis=0), tf.argmax(y_, axis=0))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(ip.log_dir + '/train/'+ip.run_dir, sess.graph)
    devel_writer = tf.summary.FileWriter(ip.log_dir + '/devel/'+ip.run_dir)
    tf.global_variables_initializer().run()

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
            summary, acc, diff, y, y_ = sess.run([merged, accuracy, diff, y, y_], feed_dict=feed_dict(gp, False))
            ysoft= tf.nn.softmax(y, dim=0)
            print diff.shape
            print diff
            print y.shape
            print y
            print ysoft.shape
            print sess.run(ysoft)
            print "Total number of events " , len(x_test)
            nr_ggf=0
            nr_vbf=0
            nr_ggf_rec=0
            nr_vbf_rec=0
            for i in range(x_test.shape[1]):
                if y_[0, i]==0.0:
                    nr_vbf +=1
                elif y_[0, i]==1.0:
                    nr_ggf+=1
                if y[0, i] > y[1, i]:
                    nr_ggf_rec+=1
                elif y[0, i]<y[1, i]:
                    nr_vbf_rec+=1
                
            print "Number of GGF events ", nr_ggf
            print "Number of VBF events ", nr_vbf
            print "Number of reconstructed GGF events ", nr_ggf_rec
            print "Number of reconstructed VBF events ", nr_vbf_rec
            print ""        

            prob_range = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
        
            # for prob in prob_range:
            #     x_new=[]
            #     y_new=[]
            #     yuscore_new=[]
            #     for i in range(len(x_test)):
            #         if sess.run(ysoft)[i][0] >prob or sess.run(ysoft)[i][1] >prob:
            #             x_new.append(x_test[i])
            #             y_new.append(y[i])
            #             yuscore_new.append(y_[i])

            #     nr_ggf_new=0
            #     nr_vbf_new=0
            #     nr_ggf_rec_new=0
            #     nr_vbf_rec_new=0
            #     for i in range(len(x_new)):
            #         if yuscore_new[i][0]==0.0:
            #             nr_vbf_new +=1
            #         elif yuscore_new[i][0]==1.0:
            #             nr_ggf_new+=1
            #         if y_new[i][0]>y_new[i][1]:
            #             nr_ggf_rec_new+=1
            #         elif y_new[i][0]<y_new[i][1]:
            #             nr_vbf_rec_new+=1

            #     print ""
            #     print " Results after cut on probability:", prob        
            #     print "Total number of events ", len(x_new)
            #     print "Number of GGF events ", nr_ggf_new
            #     print "Number of VBF events ", nr_vbf_new
            #     print "Number of reconstructed GGF events ", nr_ggf_rec_new
            #     print "Number of reconstructed VBF events ", nr_vbf_rec_new        

            #y_new=np.array(y_new)
            #yuscore_new=np.array(yuscore_new)
            #correct_prediction = tf.equal(tf.argmax(y_new, 1), tf.argmax(yuscore_new, 1))
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #print "accuracy_new", sess.run(accuracy)
            
    train_writer.close()
    devel_writer.close()

    return sess.run(ysoft)
