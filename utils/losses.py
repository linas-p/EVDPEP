import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from keras import backend as K
import numpy as np
from IPython import embed


def gaussian_nll(ytrue, ypreds):
    mu = ypreds[:, 0]
    sd = tf.keras.activations.softplus(ypreds[:, 1])
    dist = tfd.Normal(loc=mu, scale=sd)
    return tf.math.reduce_sum(-1.0 * dist.log_prob(ytrue))


def gaussian_nll_transformer(ytrue, ypreds):
    
    mu = ypreds[:, :550]
    logsigma = tf.keras.activations.softplus(ypreds[:, 550:])
    print(mu.shape, ytrue.shape)
    
    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)), axis=1)
    sigma_trace = -K.sum(2*logsigma, axis=1)
    log2pi = -0.5*K.log(2*3.14)
    
    log_likelihood = mse+sigma_trace+log2pi
    
    return K.mean(-log_likelihood)  



def gaussian_nll_lstm(ytrue, ypreds):

    mu = ypreds[:, :, 0]
    logsigma = tf.keras.activations.softplus(ypreds[:, :, 1])
    print(mu.shape, ytrue.shape)
    
    mse = -0.5*K.sum(K.square((ytrue[:,:,0]-mu)/K.exp(logsigma)), axis=1)
    sigma_trace = -K.sum(2*logsigma, axis=1)
    log2pi = -0.5*K.log(2*3.14)
    
    log_likelihood = mse+sigma_trace+log2pi
    
    return K.mean(-log_likelihood)    