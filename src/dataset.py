from tensorflow.examples.tutorials.mnist import input_data as mnist_feeder
import tensorflow as tf
from globals import TRAIN_SET_SIZE, QUERY_SET_SIZE
from tools.feed import feed
import numpy as np

class mnist(object):
    """
    Class for the mnist objects
    
    Args: 
        dir: Directory to cache at

    Class Properties:
    
        These are variables of the class that are available outside. 
        
        *   ``images``: This is the placeholder for images. This needs to be fed in.
        *   ``labels``: This is the placeholder for images. This needs to be fed in.     
        *   ``feed``: This is a feeder from mnist tutorials of tensorflow.      
    """
    def __init__ (self, dir = 'data'):
        """
        Class constructor               
        """
        self.feed = mnist_feeder.read_data_sets (dir, one_hot = True)

        #Placeholders
        with tf.variable_scope('dataset_inputs') as scope:
            self.images = tf.placeholder(tf.float32, shape=[None, 784], name = 'images')
            self.labels = tf.placeholder(tf.float32, shape = [None, 10], name = 'labels') 

class fashion_mnist(object):
    """
    Class for the fashion mnist objects.
    Ensure that data is downloaded from 
    `here <https://github.com/zalandoresearch/fashion-mnist#get-the-data>`_
    
    Args: 
        dir: Directory to cache at

    Attributes:        
        images: This is the placeholder for images. This needs to be fed in using ``feed_dict``.
        labels: This is the placeholder for images. This needs to be fed in using ``feed_dict``.     
        feed: This is a feeder from mnist tutorials of tensorflow. Use this for feeding in data.      
    """
    def __init__ (self, dir = 'data/fashion'):
        """
        Class constructor               
        """
        self.feed = mnist_feeder.read_data_sets (dir, one_hot = True)

        #Placeholders
        with tf.variable_scope('dataset_inputs') as scope:
            self.images = tf.placeholder(tf.float32, shape=[None, 784], name = 'images')
            self.labels = tf.placeholder(tf.float32, shape = [None, 10], name = 'labels')   
            

def sigmoid_to_tanh(input):
    """
    Converts input between [0-1] to [-1,1].

    Args:
        input: Numpy matrix.
    """
    return 2*input - 1
    
class mnist_hash_feeder (object):
    """
    Template for hashing mnist dataset.

    Notes:
        * Use tensorflow example MNIST and create a new dataset.
        * Tensorflow MNIST contains 
            - 5000 validation
            - 55000 train
            - 10000 test
        * We want
            - ``train`` train
            - ``query`` query
            - total - ``query`` gallery - contains train also.

    Attributes:
        query.images: Should contain query images.
        query.labels: Should contain query labels.
        gallery.images: Should contain query images.
        gallery.labels: Should contain query labels.        
        train.next_batch(mini_batch): Should return a mini_batch given
                some mini_batch values.

    Args:
        tuple: number of samples for train, query
                         (Default ``(5000, 1000)``).
    """
    def __init__(self, dataset_size = (TRAIN_SET_SIZE, QUERY_SET_SIZE), dir = 'data'):
        """
        Class constructor
        """
        dataset = mnist_feeder.read_data_sets (dir, one_hot = True)
        images = np.concatenate( (sigmoid_to_tanh(dataset.train.images), 
                                  sigmoid_to_tanh(dataset.test.images), 
                                  sigmoid_to_tanh(dataset.validation.images)), axis = 0)
        labels = np.concatenate( (dataset.train.labels, 
                                  dataset.test.labels, 
                                  dataset.validation.labels), axis = 0)
        self.query = feed(images[0:dataset_size[1]], labels[0:dataset_size[1]])
        self.train = feed(images[dataset_size[1]:dataset_size[1]+dataset_size[0]],
                          labels[dataset_size[1]:dataset_size[0]+dataset_size[1]])
        self.gallery =  feed(images[dataset_size[1]:],
                          labels[dataset_size[1]:])

class mnist_fashion_hash_feeder (object):
    """
    Template for hashing mnist dataset.

    Notes:
        * Use tensorflow example MNIST and create a new dataset.
        * Tensorflow MNIST contains 
            - 5000 validation
            - 55000 train
            - 10000 test
        * We want
            - ``train`` train
            - ``query`` query
            - total - ``query`` gallery - contains train also.

    Attributes:
        query.images: Should contain query images.
        query.labels: Should contain query labels.
        gallery.images: Should contain query images.
        gallery.labels: Should contain query labels.        
        train.next_batch(mini_batch): Should return a mini_batch given
                some mini_batch values.

    Args:
        tuple: number of samples for train, query
                         (Default ``(5000, 1000)``).
    """
    def __init__(self, dataset_size = (TRAIN_SET_SIZE, QUERY_SET_SIZE), dir = 'data/fashion'):
        """
        Class constructor
        """
        dataset = mnist_feeder.read_data_sets (dir, one_hot = True)
        images = np.concatenate( (dataset.train.images, 
                                  dataset.test.images, 
                                  dataset.validation.images), axis = 0)
        labels = np.concatenate( (dataset.train.labels, 
                                  dataset.test.labels, 
                                  dataset.validation.labels), axis = 0)
        self.query = feed(images[0:dataset_size[1]], labels[0:dataset_size[1]])
        self.train = feed(images[dataset_size[1]:dataset_size[1]+dataset_size[0]],
                          labels[dataset_size[1]:dataset_size[0]+dataset_size[1]])
        self.gallery =  feed(images[dataset_size[1]:],
                          labels[dataset_size[1]:])

class hash_mnist(object):
    """
    Class for the mnist hashing objects
    
    Args: 
        dir: Directory to cache at

    Class Properties:
    
        These are variables of the class that are available outside. 
        
        *   ``images``: This is the placeholder for images. This needs to be fed in.
        *   ``labels``: This is the placeholder for images. This needs to be fed in.     
        *   ``feed``: This is a feeder from mnist tutorials of tensorflow.      
    """
    def __init__ (self, dir = 'data'):
        """
        Class constructor               
        """
        self.feed = mnist_hash_feeder(dir = 'data')

        #Placeholders
        with tf.variable_scope('dataset_inputs') as scope:
            self.images = tf.placeholder(tf.float32, shape=[None, 784], name = 'images')
            self.labels = tf.placeholder(tf.float32, shape = [None, 10], name = 'labels') 

class hash_mnist_fashion(object):
    """
    Class for the mnist hashing objects
    
    Args: 
        dir: Directory to cache at

    Class Properties:
    
        These are variables of the class that are available outside. 
        
        *   ``images``: This is the placeholder for images. This needs to be fed in.
        *   ``labels``: This is the placeholder for images. This needs to be fed in.     
        *   ``feed``: This is a feeder from mnist tutorials of tensorflow.      
    """
    def __init__ (self, dir = 'data'):
        """
        Class constructor               
        """
        self.feed = mnist_fashion_hash_feeder(dir = 'data/fashion')

        #Placeholders
        with tf.variable_scope('dataset_inputs') as scope:
            self.images = tf.placeholder(tf.float32, shape=[None, 784], name = 'images')
            self.labels = tf.placeholder(tf.float32, shape = [None, 10], name = 'labels') 

if __name__ == '__main__':
    pass              