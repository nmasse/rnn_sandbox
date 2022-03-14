import numpy as np 
import tensorflow as tf
import argparse

def von_mises(phase, kappa):
    # Von mises distribution w/ 
    x = np.exp(kappa * phase) / np.exp(kappa)
    return x / np.mean(x)

def relu_(h):
    return tf.math.maximum(h, [0.])

def relu_derivative(h):
    return tf.math.maximum(tf.sign(h), [0.])

def softmax_(z):
    return tf.nn.softmax(z)

def softmax_cross_entropy_(y, z, epsilon=0.0001):
    p = softmax_(z)
    p = tf.math.maximum(p, [epsilon]) #Regularize in case p is small
    return -y @ tf.math.log(p)

def softmax_cross_entropy_derivative(y, z):
    return softmax_(z) - y

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def convert(argument):
    return list(map(int, argument.split(',')))