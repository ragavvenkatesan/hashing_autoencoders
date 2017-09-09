import tensorflow as tf
from math import sqrt
from third_party import put_kernels_on_grid
import numpy as np
import time 

def profile(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print("Time taken by method " + str(func.__name__) + " " +  \
                                        str(time.time() - start) + " seconds")
        return result
    return wrap

def log(x):
    return tf.log(x + 1e-8)

def one_hot_to_binary(input):
    """
    Converts one-hot vectors to binary

    Args:
        input: ndarray

    Returns:
        ndarry: same shape as input
    """
    int_array = np.asarray([np.argmax(input, 1)], dtype = 'uint8').T
    return np.unpackbits(int_array, axis = 1)[:,-4:]
    
def hamming(a, b, mode = 'tanh'):
    """
    Returns the hamming distance between vectors in ``a`` and ``b``. 

    Args:
        a: numpy nd array
        b: numpy nd array
    
    Returns: 
        ndarray: Hamming disance matrix.
    """
    if mode == 'sigmoid':
        a = 2*a - 1
        b = 2*b - 1
    output = ( -1 * np.dot (a, b.T) + a.shape[1] ) / 2
    # output = np.dot(( 1 - a + np.dot(a.T,b)).T,( 1 - b + np.dot(a.T,b)))
    return output 

def avg(input):
    """
    
    Args:
        input: a vector
    
    Returns:

    """
    num_of_nonzeros = np.sum(input != 0)
    return np.sum(input)/num_of_nonzeros

@profile
def map ( query, database, m = 3):
    """
    Calulates mean average precision of a retrieval.

    Args:
        query: A tuple containing, query features and query labels.
        database: A tuple containing, database features and labels.
        m: Threshold of retrieval. Maximum is``query_features.shape[1]``

    Returns:
        float: mean average precision
    """
    query_features = query[0]
    query_labels = query[1]
    database_features = database[0]
    database_labels = database[1]
    import scipy.io as io
    io.savemat('ourHashes.mat',{'query_features':query_features, 'query_labels':query_labels, 'database_features':database_features,'database_labels':database_labels })
    distances_features = hamming(query_features, database_features)
    similarity_labels = 1-(hamming(query_labels, database_labels, 'sigmoid')/2)
    precisions = []
    # for i in range(m):
    #     hamming_neighbors = distances_features <= i
    #     correctly_retrieved = np.multiply(hamming_neighbors, similarity_labels)
    #     fps = np.sum(np.logical_and(hamming_neighbors == 1, similarity_labels == 0 ))
    #     tps = np.sum(correctly_retrieved)
    #     if tps ==0 and fps == 0:
    #         fps = 1
    #     precision = float(tps)/float((tps+fps))
    #     precisions.append(precision)

    # Code for MAP #Python version of the matlab code - DPSH_IJCAI_ version 1.0_beta23. 
    distances_features =np.argsort(distances_features, axis = 1)
    MAP = 0
    numSucc = 0.
    pos = np.arange(1, distances_features.shape[1]+1)
    for i in range(query_features.shape[0]):
        ngb = np.reshape(similarity_labels[i, distances_features[i, :]], (similarity_labels.shape[1]))
        nRel = np.sum(ngb)
        if nRel > 0:
            prec = np.cumsum(ngb)/pos
            ap = avg(prec*ngb)
            MAP = MAP + ap
            numSucc = numSucc + 1.
    MAP = MAP/numSucc

    return MAP

def _test_map ():
    """
    Unittestfor testing the map method.  
    """
    query_hashes = np.asarray( [ 
                                    [ 1, 0, 0], 
                                    [ 0, 1, 0] 
                                    ] ) 
    query_lables = query_hashes
    gallery_hashes = np.asarray( [ 
                                    [ 0, 0, 1], 
                                    [ 0, 1, 1], 
                                    [ 1, 0, 0 ], 
                                    [ 0, 1, 0]
                                ] )
    gallery_labels = np.asarray( [  [ 0, 0, 1], 
                                    [ 0, 0, 1],
                                    [ 1, 0, 0 ],
                                    [ 0, 1, 0]] )

    print(map ( (query_hashes, query_lables) , (gallery_hashes, gallery_labels) ))

def rmse (a, b):
    """
    Returns the RMSE error between ``a`` and ``b``.

    Args:
        a: a tensor
        b: another tensor

    Returns:
        tensor: RMSE error
    """
    return  tf.reduce_mean(tf.squared_difference(a, b))

def initializer(shape, name = 'xavier'):
    """
    A method that returns random numbers for Xavier initialization.

    Args:
        shape: shape of the initializer.
        name: Name for the scope of the initializer

    Returns:
        float: random numbers from tensorflow random_normal

    """
    with tf.variable_scope(name) as scope:
        stddev = 1.0 / tf.sqrt(float(shape[0]), name = 'stddev')
        inits = tf.truncated_normal(shape=shape, stddev=stddev, name = 'xavier_init')
    return inits

def nhwc2hwnc (nhwc, name = 'nhwc2hwnc'):
    """
    This method reshapes (NHWC) 4D bock to (HWNC) 4D block

    Args:
        nhwc: 4D block in (NHWC) format

    Returns:
        tensorflow tensor: 4D block in (HWNC) format
    """    
    with tf.variable_scope(name) as scope:
        out = tf.transpose(nhwc, [1,2,0,3])
    return out

def nhwc2hwcn (nhwc, name = 'nhwc2hwcn'):
    """
    This method reshapes (NHWC) 4D bock to (HWCN) 4D block

    Args:
        nhwc: 4D block in (NHWC) format

    Returns:
        tensorflow tensor: 4D block in (HWCN) format
    """    
    with tf.variable_scope(name) as scope:
        out = tf.transpose(nhwc, [1,2,3,0])
    return out

def visualize_filters (filters, name = 'conv_filters'):
    """
    This method is a wrapper to ``put_kernels_on_grid``. 
    This adds the grid to image summaries.

    Args:
        tensorflow tensor: A 4D block in (HWNC) format.
    """
    grid = put_kernels_on_grid (filters, name = 'visualizer_' + name) 
    tf.summary.image(name, grid, max_outputs = 1)

def visualize_images (images, name = 'images', num_images = 6):
    """
    This method sets up summaries for images.

    Args:
        images: a 4D block in (NHWC) format.
        num_images: Number of images to display

    Todo:
        I want this to display images in a grid rather than just display using 
        tensorboard's ugly system. This method should be a wrapper that converts 
        images in (NHWC) format to (HWNC) format and makes a grid of the images.
        
        Perhaps a code like this:

        ..code-block:: python
            
            images = images [0:num_images-1]
            images = nhwc2hwcn(images, name = 'nhwc2hwcn' + name)
            visualize_filters(images, name)        
    """
    tf.summary.image(name, images, max_outputs = num_images)

def visualize_1D_filters(filters, name ='dot_product_filters'):
    """
    This method will visualize dot_product layer filter weights. 
    This will do so in a grid by transposing the weights.

    Args:
        tensorflow tensor: A 2D block in (in, out) format.
    """    
    filters = tf.transpose(filters)
    dim = int( sqrt( filters.shape[1].value ) ) 
    filters = tf.reshape(filters, [dim, dim, 1, -1])    
    visualize_filters(filters, name = name)

if __name__ == '__main__':
    pass    