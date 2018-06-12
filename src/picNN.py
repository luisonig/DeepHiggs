#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.util import *

def train_pixels(ip, x_train, y_train, x_devel, y_devel):
    """


    """

    #print x_train.shape
    #print y_train.shape

    #print x_devel.shape
    #print y_devel.shape

    #print "dropout:", ip.dropout

    # Number of training examples
    m = x_train.shape[1]                 # m: number of examples in the train set
    unique, counts = np.unique(y_train, return_counts=True)
    m_vbf_train = counts[1]              # m_vbf_train: number of VBF examples in the train set

    # find number of VBF events in test sample
    unique, counts = np.unique(y_devel, return_counts=True)
    m_vbf_devel = counts[1]               # m_vbf_devel: number of VBF examples in the test set

    # Input/output layer dimensions
    n_x = x_train.shape[0]
    n_y = y_train.shape[0]
    costs_train = []                     # To keep track of the train costs
    costs_devel = []                     # To keep track of the devel costs
    accur_train = []                     # To keep track of the train accuracy
    accur_devel = []                     # To keep track of the devel accuracy
    prec_train = []                      # To keep track of the train precision
    prec_devel = []                      # To keep track of the devel precision
    rec_train  = []                      # To keep track of the train recall
    rec_devel  = []                      # To keep track of the devel recall

    #--> set random seed: only for testing purposes - TO BE REMOVED
    tf.set_random_seed(1)
    seed = 3

    #sess = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=8))

    # Input placeholders
    with tf.name_scope('input'):
        x  = tf.placeholder(tf.float32, [n_x, None], name='x-input')
        y_ = tf.placeholder(tf.float32, [n_y, None], name='y-input')

    # First FC hidden layer with 200 units
    hidden1, weights1 = nn_layer(x, 200, n_x, 'layer1')

    # Second FC hidden layer with 50 units
    hidden2, weights2 = nn_layer(hidden1, 50, 200, 'layer2')

    # Dropout layer
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden2, keep_prob)

    # Do not apply softmax activation yet, see below.
    y, _ = nn_layer(dropped, n_y, 50, 'layer3', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.weighted_cross_entropy_with_logits(targets=tf.transpose(y_), logits=tf.transpose(y), pos_weight=ip.xentropy_w)
        cost = tf.reduce_mean(diff)

        # L2 Regularization
        regularizer = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)
        cost = tf.reduce_mean(cost + ip.lambdal2 * regularizer)
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
            confmatrix = tf.confusion_matrix(tf.squeeze(tf.cast(tf.round(y_prob), tf.int32)), tf.squeeze(tf.cast(y_, tf.int32)), num_classes=2)
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
                    cost_devel = cost.eval({x: x_devel, y_: y_devel, keep_prob: 1.0})
                    cm_train = confmatrix.eval({x: x_train, y_: y_train, keep_prob: 1.0})
                    cm_devel = confmatrix.eval({x: x_devel, y_: y_devel, keep_prob: 1.0})
                    ac_train = accuracy.eval({x: x_train, y_: y_train, keep_prob: 1.0})
                    ac_devel = accuracy.eval({x: x_devel, y_: y_devel, keep_prob: 1.0})
                    precision_train = (cm_train[1,0] + cm_train[1,1])/float(m_vbf_train)
                    precision_devel = (cm_devel[1,0] + cm_devel[1,1])/float(m_vbf_devel)
                    recall_train = (cm_train[1,1])/float(m_vbf_train)
                    recall_devel = (cm_devel[1,1])/float(m_vbf_devel)
                    costs_train.append(epoch_cost)
                    costs_devel.append(cost_devel)
                    accur_train.append(ac_train)
                    accur_devel.append(ac_devel)
                    prec_train.append(precision_train)
                    prec_devel.append(precision_devel)
                    rec_train.append(recall_train)
                    rec_devel.append(recall_devel)


            ## Save model
            save_path = saver.save(sess, ip.log_dir + '/models/'+ip.runname, write_meta_graph=False)
            print("Model saved in path: %s" % save_path)

            cost_devel = cost.eval({x: x_devel, y_: y_devel, keep_prob: 1.0})
            cm_train = confmatrix.eval({x: x_train, y_: y_train, keep_prob: 1.0})
            cm_devel = confmatrix.eval({x: x_devel, y_: y_devel, keep_prob: 1.0})
            ac_train = accuracy.eval({x: x_train, y_: y_train, keep_prob: 1.0})
            ac_devel = accuracy.eval({x: x_devel, y_: y_devel, keep_prob: 1.0})
            yprob    = y_prob.eval({x: x_devel, y_: y_devel, keep_prob: 1.0})
            precision_train = (cm_train[1,0] + cm_train[1,1])/float(m_vbf_train)
            precision_devel = (cm_devel[1,0] + cm_devel[1,1])/float(m_vbf_devel)
            recall_train = (cm_train[1,1])/float(m_vbf_train)
            recall_devel = (cm_devel[1,1])/float(m_vbf_devel)
            costs_train.append(epoch_cost)
            costs_devel.append(cost_devel)
            accur_train.append(ac_train)
            accur_devel.append(ac_devel)
            prec_train.append(precision_train)
            prec_devel.append(precision_devel)
            rec_train.append(recall_train)
            rec_devel.append(recall_devel)

            if ip.print_info == True:
                ## Plot the cost
                plt.figure(1, figsize=(10,5))
                ax=plt.subplot(1, 2, 1)
                ax.set_yscale("log")
                plt.plot(np.squeeze(costs_train), label='train')
                plt.plot(np.squeeze(costs_devel), label='devel')
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
                plt.savefig(ip.log_dir + '/'+ip.runname+'_accuracy.png')
                #plt.show()

            print("Train Accuracy:", ac_train)
            print("Test Accuracy:", ac_devel)
            print("Conf. matrix: T neg.= " + str(cm_devel[0,0]) + ", F neg.= " + str(cm_devel[0,1])
                            + ", F pos.= " + str(cm_devel[1,0]) + ", T pos.= " + str(cm_devel[1,1]))

            plotfile=open('training_plots.csv','w')
            plotfile.write('epoch, cost_train, cost_devel, accuracy_train, accuracy_devel, prec_train, prec_devel, recall_train, recall_devel\n')
            for i in range(len(costs_train)):
                plotfile.write(str(i*10)+ ", " + str(costs_train[i]) + ", " + str(costs_devel[i])
                               + ", " + str(accur_train[i]) + ", " + str(accur_devel[i])
                               + ", " + str(prec_train[i])  + ", " + str(prec_devel[i])
                               + ", " + str(rec_train[i])   + ", " + str(rec_devel[i]) + "\n")
            plotfile.close()


        elif ip.mode == "eval":
            saver.restore(sess, ip.log_dir + '/models/'+ip.runname)

            cm_train = confmatrix.eval({x: x_train, y_: y_train, keep_prob: 1.0})
            cm_devel = confmatrix.eval({x: x_devel, y_: y_devel, keep_prob: 1.0})
            ac_train = accuracy.eval({x: x_train, y_: y_train, keep_prob: 1.0})
            ac_devel = accuracy.eval({x: x_devel, y_: y_devel, keep_prob: 1.0})
            yprob    = y_prob.eval({x: x_devel, y_: y_devel, keep_prob: 1.0})

            if ip.print_info == True:
                print("Train Accuracy:", ac_train)
                print("Test Accuracy:", ac_devel)
                print("Conf. matrix: T neg.= " + str(cm_devel[0,0]) + ", F neg.= " + str(cm_devel[0,1])
                                + ", F pos.= " + str(cm_devel[1,0]) + ", T pos.= " + str(cm_devel[1,1]))

    #eturn cm_devel, ac_devel, ac_train
    return yprob
