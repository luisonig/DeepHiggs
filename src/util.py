#!/usr/bin/env python

import numpy as np
import tensorflow as tf


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
        return activations

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
    
##--] Tensorflow functions
    
