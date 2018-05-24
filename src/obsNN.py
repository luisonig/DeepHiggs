#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.util import *

def train_obs(gp, ip, x_train, y_train, x_test, y_test):
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


def train_obs_new(ip, x_train, y_train, x_test, y_test):
    """


    """

    #print x_train.shape
    #print y_train.shape

    #print x_test.shape
    #print y_test.shape

    #print "dropout:", ip.dropout

    # Number of training examples 
    m = x_train.shape[1]                 # m: number of examples in the train set
    
    # Input/output layer dimensions
    n_x = x_train.shape[0]
    n_y = y_train.shape[0]
    costs = []                           # To keep track of the cost
    accur_train = []                     # To keep track of the train accuracy
    accur_devel = []                     # To keep track of the devel accuracy
    
    #--> set random seed: only for testing purposes - TO BE REMOVED
    tf.set_random_seed(1)
    seed = 3
    
    #sess = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=8))

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
            cost = tf.reduce_mean(diff)
    tf.summary.scalar('cost', cost)

    with tf.name_scope('optimizer'):
        if ip.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(ip.learning_rate).minimize(cost)
        elif ip.optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(ip.learning_rate).minimize(cost)
        elif ip.optimizer == 'GraDe':
            optimizer = tf.train.GradientDescentOptimizer(ip.learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        with tf.name_scope('y_prob'):
            y_prob = tf.sigmoid(y)
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.round(y_prob), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.name_scope('confmatrix'):
            confmatrix = tf.confusion_matrix(tf.squeeze(tf.cast(y_prob, tf.int32)), tf.squeeze(tf.cast(y_, tf.int32)), num_classes=2)
    tf.summary.scalar('accuracy', accuracy)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    ## Start the session to compute the tensorflow graph
    with tf.Session() as sess:
                
        ## Run the initialization
        sess.run(init)

        # Merge all the summaries and write them out
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(ip.log_dir + '/train/'+ip.run_dir, sess.graph)
        devel_writer = tf.summary.FileWriter(ip.log_dir + '/devel/'+ip.run_dir)
        
        if ip.mode == "train":
            
            ## Do the training loop
            for epoch in range(ip.num_epochs):

                epoch_cost = 0.                              # Defines a cost related to an epoch
                num_minibatches = int(m / ip.minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = random_mini_batches(x_train, y_train, ip.minibatch_size, seed)

                for minibatch in minibatches:

                    ## Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                
                    _ , minibatch_cost = sess.run([optimizer, cost], 
                                                  feed_dict={x: minibatch_X, y_: minibatch_Y, keep_prob: ip.dropout})

                    epoch_cost += minibatch_cost / num_minibatches
        
                ## Print the cost every epoch
                if ip.print_info == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if ip.print_info == True and epoch % 10 == 0:
                    ac_train = accuracy.eval({x: x_train, y_: y_train, keep_prob: 1.0})
                    ac_devel = accuracy.eval({x: x_test, y_: y_test, keep_prob: 1.0})
                    costs.append(epoch_cost)
                    accur_train.append(ac_train)
                    accur_devel.append(ac_devel)
                    

            ## Save model
            save_path = saver.save(sess, ip.log_dir + '/models/'+ip.runname, write_meta_graph=False)
            print("Model saved in path: %s" % save_path)
        
            cm_train = confmatrix.eval({x: x_train, y_: y_train, keep_prob: 1.0})
            cm_devel = confmatrix.eval({x: x_test, y_: y_test, keep_prob: 1.0})
            ac_train = accuracy.eval({x: x_train, y_: y_train, keep_prob: 1.0})
            ac_devel = accuracy.eval({x: x_test, y_: y_test, keep_prob: 1.0})
            yprob    = y_prob.eval({x: x_test, y_: y_test, keep_prob: 1.0})
            accur_train.append(ac_train)
            accur_devel.append(ac_devel)
            
            if ip.print_info == True:
                ## Plot the cost
                plt.figure(1, figsize=(10,5))
                ax=plt.subplot(1, 2, 1)
                ax.set_yscale("log")
                plt.plot(np.squeeze(costs))
                plt.ylabel('cost')
                plt.xlabel('iterations (x10)')                
                plt.title("Learning rate =" + str(ip.learning_rate))
                ax=plt.subplot(1, 2, 2)
                ax.set_yscale("linear")
                plt.plot(np.squeeze(accur_train), label='train')
                plt.plot(np.squeeze(accur_devel), label='devel')
                plt.ylabel('accuracy')
                plt.xlabel('iterations (x10)')                
                plt.title("Learning rate =" + str(ip.learning_rate))
                plt.tight_layout()
                plt.savefig(ip.log_dir + '/'+ip.runname+'.png')
                #plt.show()                
        
            print("Train Accuracy:", ac_train)
            print("Test Accuracy:", ac_devel)
            print("Conf. matrix: T neg.= " + str(cm_devel[0,0]) + ", F neg.= " + str(cm_devel[0,1])
                            + ", F pos.= " + str(cm_devel[1,0]) + ", T pos.= " + str(cm_devel[1,1]))

        elif ip.mode == "eval":
            saver.restore(sess, ip.log_dir + '/models/'+ip.runname)
            
            cm_train = confmatrix.eval({x: x_train, y_: y_train, keep_prob: 1.0})
            cm_devel = confmatrix.eval({x: x_test, y_: y_test, keep_prob: 1.0})
            ac_train = accuracy.eval({x: x_train, y_: y_train, keep_prob: 1.0})
            ac_devel = accuracy.eval({x: x_test, y_: y_test, keep_prob: 1.0})
            yprob    = y_prob.eval({x: x_test, y_: y_test, keep_prob: 1.0})
            
            if ip.print_info == True:
                print("Train Accuracy:", ac_train)
                print("Test Accuracy:", ac_devel)
                print("Conf. matrix: T neg.= " + str(cm_devel[0,0]) + ", F neg.= " + str(cm_devel[0,1])
                                + ", F pos.= " + str(cm_devel[1,0]) + ", T pos.= " + str(cm_devel[1,1]))
        
    #eturn cm_devel, ac_devel, ac_train
    return yprob
