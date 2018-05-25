#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from src.analysis import *
import math


class GlobalParameters():
    """
    Global class for data parameters. This includes:

    Methods:
    index_in_epoch -- returns XXX
    epochs_completed -- retuns YYY
    num_examples -- return the number of training examples

    """

    def __init__(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = 0

    def index_in_epoch(self):
        return self._index_in_epoch

    def epochs_completed(self):
        return self._epochs_completed

    def num_examples(self):
        return self._num_examples


def normalize_input(x_vec):
    """
    Normalizes vector by the largest component such that all components are |x|<1

    Arguments:
    x_vec -- vector to normalize

    returns:
    x_vec_norm -- normalized copy of x_vec
    """

    n_obs = x_vec.shape[0]

    x_vec_norm = np.zeros(x_vec.shape)

    for i in range(n_obs):
        obsmax = np.amax(x_vec[i, :])
        x_vec_norm[i, :] = x_vec[i, :]/obsmax

    return x_vec_norm

def cut_probability(x_test, y, y_, ggf_event_count, vbf_event_count):
    """
    Loops over a range of probabilities where events are only accepted if the level of
    confidence of the NN that the event is a signal is large than a given threshold

    Arguments:
    x_test input variables of NN
    y -- output vector of the NN
    y_ -- reference vector (true result)
    ggf_event_count -- number of events in ggf sample (needed for correct XS)
    vbf_event_count -- number of events in vbf sample (needed for correct XS)

    """
    n_events=len(x_test[0])
    print "Total number of events " , x_test.shape[1]
    nr_ggf=0
    nr_vbf=0
    nr_ggf_rec=0
    nr_vbf_rec=0
    for i in range(x_test.shape[1]):
        if y_[0, i] == 1.0:
            nr_vbf +=1
        elif y_[0, i] == 0.0:
            nr_ggf+=1
        if y[0, i] < 0.5:
            nr_ggf_rec+=1
        elif y[0, i] > 0.5:
            nr_vbf_rec+=1

    print "Number of GGF events ", nr_ggf
    print "Number of VBF events ", nr_vbf
    print "Number of reconstructed GGF events ", nr_ggf_rec
    print "Number of reconstructed VBF events ", nr_vbf_rec
    print ""

    plotfile=open('prob_plots.csv','w')
    plotfile.write('prob,n_events,nr_tot_new,nr_ggf,nr_vbf,xs_ggf,xs_vbf,sb,ssqrtb,accuracy\n')

    prob_range = [0.0,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]

    for prob in prob_range:
        x_new=[]
        y_new=[]
        yuscore_new=[]
        n_events_new=0
        for i in range(n_events):
            #if y[0][i] >prob or y[1][i] >prob:
            if y[0][i] > prob:
                x_new.append(x_test.transpose()[i])
                n_events_new +=1
                y_new.append(y.transpose()[i])
                yuscore_new.append(y_.transpose()[i])
        yuscore_new=np.array(yuscore_new)
        y_new=np.array(y_new)
        x_new=np.array(x_new)
        nr_ggf_new=0
        nr_vbf_new=0
        nr_ggf_rec_new=0
        nr_vbf_rec_new=0
        for i in range(len(x_new)):
            if yuscore_new[i][0] == 1.0:
                nr_vbf_new += 1
            elif yuscore_new[i][0] == 0.0:
                nr_ggf_new += 1
            if y_new[i][0] < 0.5:
                nr_ggf_rec_new += 1
            elif y_new[i][0] > 0.5:
                nr_vbf_rec_new += 1

        print ""
        print " Results after cut on probability:", prob
        print "Total number of events ", n_events_new
        print "Number of GGF events ", nr_ggf_new
        print "Number of VBF events ", nr_vbf_new

        if nr_ggf_new > 0:
            xs_ggf, xs_vbf = compute_XS(x_new.transpose(), yuscore_new.transpose(), nr_ggf_new, nr_vbf_new, ggf_event_count, vbf_event_count)
            sb = xs_vbf/xs_ggf
            ssqrtb = xs_vbf/math.sqrt(xs_ggf)
            print "S/B", xs_vbf/xs_ggf
        else:
            xs_ggf='NA'
            xs_vbf='NA'
            sb='NA'
        print "Number of reconstructed GGF events ", nr_ggf_rec_new
        print "Number of reconstructed VBF events ", nr_vbf_rec_new

        #y_new=np.array(y_new)
        #yuscore_new=np.array(yuscore_new)
        if nr_ggf_new >0 and nr_vbf_new >0 :
            correct_prediction = tf.equal(tf.round(y_new), yuscore_new)
            #correct_prediction = tf.equal(tf.argmax(y_new, 1), tf.argmax(yuscore_new, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            sess=tf.InteractiveSession()
            accuracy= sess.run(accuracy)
            print "accuracy_new", accuracy
        else:
            accuracy='NA'
        plotfile.write(str(prob)+','+str(n_events)+','+str(n_events_new)+','+str(nr_ggf_new)
                       +','+str(nr_vbf_new)+','+str(xs_ggf)+','+str(xs_vbf)+','+str(sb)
                       +','+str(ssqrtb)+','+str(accuracy)+'\n')

    plotfile.close()

##--[ Tensorflow functions:
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    #return tf.get_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer(seed = 1))

def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    #return tf.get_variable(name, shape, initializer = tf.zeros_initializer())

def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, output_dim, input_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable("W_"+layer_name, [output_dim, input_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable("b_"+layer_name, [output_dim, 1])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.add(tf.matmul(weights, input_tensor), biases)
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations, weights

def nn_final_layer(input_tensor, output_dim, input_dim, layer_name):
    """
    Reusable code for making a final softmax neural net layer.  It does
    a matrix multiply, bias add, and then uses softmax in axis=0 to
    nonlinearize.  It also sets up name scoping so that the resultant
    graph is easy to read, and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable("W_"+layer_name, [output_dim, input_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable("b_"+layer_name, [output_dim, 1])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.add(tf.matmul(weights, input_tensor), biases)
            tf.summary.histogram('pre_activations', preactivate)
        activations = tf.nn.softmax(preactivate, dim=0, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def next_batch(gp, batch_size, xs, ys):
    start = gp.index_in_epoch()
    if start + batch_size > gp.num_examples():
        #Finished epoch
        gp._epochs_completed +=1
        # Get the rest examples in this epoch
        rest_num_examples = gp.num_examples() - start
        xs_rest_part = xs[:, start:gp.num_examples()]
        ys_rest_part = ys[:, start:gp.num_examples()]

        start = 0
        gp._index_in_epoch = batch_size - rest_num_examples
        end = gp.index_in_epoch()
        xs_new_part = xs[:, start:end]
        ys_new_part = ys[:, start:end]
        xs=np.concatenate((xs_rest_part, xs_new_part), axis=1)
        ys=np.concatenate((ys_rest_part, ys_new_part), axis=1)
        return xs, ys
    else:
        gp._index_in_epoch += batch_size
        end = gp.index_in_epoch()
        return xs[:, start:end], ys[:, start:end]


def random_mini_batches(X, Y, mini_batch_size = 128, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for VBF / 0 for GGF), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

##--] Tensorflow functions
