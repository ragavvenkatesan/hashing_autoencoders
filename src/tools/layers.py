import tensorflow as tf
from support import initializer, visualize_filters
import numpy as np

def softmax_layer (input, name = 'softmax', temperature = 1.):
    """
    Creates the softmax normalization

    Args: 
        input: Where is the input of the layer coming from
        temperature: Temperature of the softmax
        name: Name scope of the layer

    Returns:
        tuple: ``(softmax, prediction)``, A softmax output node and prediction output node
    """
    with tf.variable_scope(name) as scope: 
        temperature = tf.constant(temperature, dtype = tf.float32, name ='temperature')
        z = tf.divide(input, temperature, name = 'scale')
        inference = tf.nn.softmax(z, name = 'inference')
        predictions = tf.argmax(inference, 1, name = 'predictions')
        tf.summary.histogram('inference', inference)
        tf.summary.histogram('predictions', predictions)  
    return (inference, predictions)

def dot_product_layer(input, params = None, neurons = 1200, name = 'fc', activation = 'relu'):
    """
    Creates a fully connected layer

    Args:
        input: Where is the input of the layer coming from
        neurons: Number of neurons in the layer.
        params: List of tensors, if supplied will use those params.
        name: name scope of the layer

    Returns:
        tuple: The output node and A list of parameters that are learnanble
    """
    with tf.variable_scope(name) as scope:
        if params is None or params[0] is None:
            weights = tf.Variable(initializer([input.shape[1].value,neurons], name = 'xavier_weights'),\
                                            name = 'weights')
        else:
            weights = params[0]
        
        if params is None or params[1] is None:    
            bias = tf.Variable(initializer([neurons], name = 'xavier_bias'), name = 'bias')
        else:
            bias = params[1]

        dot = tf.nn.bias_add(tf.matmul(input, weights, name = 'dot'), bias, name = 'pre-activation')
        if activation == 'relu':
            activity = tf.nn.relu(dot, name = 'activity' )
        elif activation == 'sigmoid':
            activity = tf.nn.sigmoid(dot, name = 'activity' )            
        elif activation == 'tanh':
            activity = tf.nn.tanh(dot, name = 'activity' )             
        elif activation == 'identity':
            activity = dot                     
        params = [weights, bias]
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('bias', bias)  
        tf.summary.histogram('activity', activity)                              
    return (activity, params)

def conv_2d_layer (input, 
                neurons = 20,
                filter_size = (5,5), 
                stride = (1,1,1,1), 
                padding = 'VALID',
                name = 'conv', 
                params = None,
                activation = 'relu',
                visualize = False):
    """
    Creates a convolution layer

    Args:
        input: (NHWC) Where is the input of the layer coming from
        neurons: Number of neurons in the layer.
        name: name scope of the layer
        filter_size: A tuple of filter size ``(5,5)`` is default.
        stride: A tuple of x and y axis strides. ``(1,1,1,1)`` is default.
        name: A name for the scope of tensorflow
        visualize: If True, will add to summary. Only for first layer at the moment.
        activation: Activation for the outputs.
        padding: Padding to be used in convolution. "VALID" is default.

    Returns:
        tuple: The output node and A list of parameters that are learnanble
    """        
    f_shp = [filter_size[0], filter_size[1], input.shape[3].value, neurons]
    with tf.variable_scope(name) as scope:
        if params is None or params[0] is None:        
            weights = tf.Variable(initializer(  f_shp, 
                                            name = 'xavier_weights'),\
                                            name = 'weights')
        else:
            weights = params[0]
        if params is None or params[0] is None:                    
            bias = tf.Variable(initializer([neurons], name = 'xavier_bias'), name = 'bias')
        else:
            bias = params[1]
    
        c_out = tf.nn.conv2d(   input = input,
                                filter = weights,
                                strides = stride,
                                padding = padding,
                                name = scope.name  )
        c_out_bias = tf.nn.bias_add(c_out, bias, name = 'pre-activation')
        if activation == 'relu':
            activity = tf.nn.relu(c_out_bias, name = 'activity' )
        elif activation == 'sigmoid':
            activity = tf.nn.sigmoid(c_out_bias, name = 'activity' )            
        elif activation == 'tanh':
            activity = tf.nn.tanh(c_out_bias, name = 'activity' )                
        elif activation == 'identity':
            activity = c_out_bias
        params = [weights, bias]
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('bias', bias)  
        tf.summary.histogram('activity', activity) 
        if visualize is True:  
            visualize_filters(weights, name = 'filters_' + name)
    return (activity, params)        

def upsampling_layer (input, size,
                    name = 'upsample', method = tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    """
    Creates a up sampling layer

    Args:
        input: (NHWC) Where is the input of the layer coming from
        size: upsampling image size is a tuple of height and width
        method: refer tensorflow.

    Returns:
        tensor: The output node.
    """            
    with tf.variable_scope(name) as scope:
        output = tf.image.resize_images(input, size=size, method=method)  
    return output

def deconv_2d_layer (input, 
                output_shape = (28,28),
                neurons = 20,
                filter_size = (5,5), 
                stride = (1,1,1,1), 
                padding = 'VALID',
                name = 'conv', 
                params = None,
                activation = 'relu',
                visualize = False):
    """
    Creates a de-convolution layer, Similar to convolutional layers 
    but needs an output shape to work with.

    Args:
        input: (NHWC) Where is the input of the layer coming from
        neurons: Number of neurons in the layer.
        name: name scope of the layer
        output_shape: image output shape in tuple
        filter_size: A tuple of filter size ``(5,5)`` is default.
        stride: A tuple of x and y axis strides. ``(1,1,1,1)`` is default.
        name: A name for the scope of tensorflow
        visualize: If True, will add to summary. Only for first layer at the moment.
        activation: Activation for the outputs.
        padding: Padding to be used in convolution. "VALID" is default.

    Returns:
        tuple: The output node and A list of parameters that are learnanble
    """          
    f_shp = [filter_size[0], filter_size[1], neurons,  input.shape[3].value]
    o_shp = [tf.shape(input)[0], output_shape[0], output_shape[1], neurons]
    with tf.variable_scope(name) as scope:
        if params is None or params[0] is None:        
            weights = tf.Variable(initializer(  f_shp, 
                                            name = 'xavier_weights'),\
                                            name = 'weights')
        else:
            weights = params[0]
        if params is None or params[1] is None:                    
            bias = tf.Variable(initializer([neurons], name = 'xavier_bias'), name = 'bias')
        else:
            bias = params [1]
        c_out = tf.nn.conv2d_transpose(
                                value = input,
                                filter = weights,
                                strides = stride,
                                output_shape = o_shp,
                                padding = padding,
                                name = scope.name  )
        c_out_bias = tf.nn.bias_add(c_out, bias, name = 'pre-activation')
        if activation == 'relu':
            activity = tf.nn.relu(c_out_bias, name = 'activity' )
        elif activation == 'sigmoid':
            activity = tf.nn.sigmoid(c_out_bias, name = 'activity' )      
        elif activation == 'tanh':
            activity = tf.nn.tanh(c_out_bias, name = 'activity' )                    
        elif activation == 'identity':
            activity = c_out_bias
        params = [weights, bias]
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('bias', bias)  
        tf.summary.histogram('activity', activity) 
        if visualize is True:  
            visualize_filters(weights, name = 'filters_' + name)
    return (activity, params)      

def flatten_layer (input, in_shp = None, name = 'flatten'):
    """
    This layer returns the flattened output
    Args:
        input: a 4D node.
        input_shape: Send in an input shape.
        name: name scope of the layer.
    Returns:
        tensorflow tensor: a 2D node.
    """
    with tf.variable_scope(name) as scope:
        if in_shp is None:
            in_shp = input.get_shape().as_list()
        output = tf.reshape(input, [-1, in_shp[1]*in_shp[2]*in_shp[3]])
    return output 

def threshold_layer (input, name = 'threeshold'):
    """
    This layer returns the threshold output
    Args:
        input: a 4D node.
        name: name scope of the layer.
    Returns:
        tensorflow tensor: a 2D node.
    """
    with tf.variable_scope(name) as scope:    
        cond = tf.less(input, tf.zeros(tf.shape(input)))
        output = tf.where(cond, -1*tf.ones(tf.shape(input)), tf.ones(tf.shape(input)))
        tf.summary.histogram('thresholded_codewords', output)        
    return output 

def max_pool_2d_layer  (   input, 
                        pool_size = (1,2,2,1),
                        stride = (1,2,2,1),
                        padding = 'VALID',
                        name = 'pool' ):
    """
    Creates a max pooling layer

    Args:
        input: (NHWC) Where is the input of the layer coming from
        name: name scope of the layer
        pool_size: A tuple of filter size ``(5,5)`` is default.
        stride: A tuple of x and y axis strides. ``(1,1,1,1)`` is default.
        name: A name for the scope of tensorflow
        padding: Padding to be used in convolution. "VALID" is default.

    Returns:
        tensorflow tensor: The output node 
    """       
    with tf.variable_scope(name) as scope:
        output = tf.nn.max_pool (   value = input,
                                    ksize = pool_size,
                                    strides = stride,
                                    padding = padding,
                                    name = name ) 
    return output

def local_response_normalization_layer (input, name = 'lrn'):
    """
    This layer returns the flattened output

    Args:
        input: a 4D node.
        name: name scope of the layer.

    Returns:
        tensorflow tensor: a 2D node.
    """
    with tf.variable_scope(name) as scope:
        output = tf.nn.lrn(input)
    return output

def unflatten_layer (input, channels = 1, name = 'unflatten'):
    """
    This layer returns the unflattened output
    Args:
        input: a 2D node.
        chanels: How many channels are there in the image. (Default = ``1``)
        name: name scope of the layer.

    Returns:
        tensorflow tensor: a 4D node in (NHWC) format that is square in shape.
    """
    with tf.variable_scope(name) as scope:
        dim = int( np.sqrt( input.shape[1].value / channels ) ) 
        output = tf.reshape(input, [-1, dim, dim, channels])
    return output

def dropout_layer (input, prob, name ='dropout'):
    """
    This layer drops out nodes with the probability of 0.5
    During training time, run a probability of 0.5.
    During test time run a probability of 1.0. 
    To do this, ensure that the ``prob`` is a ``tf.placeholder``.
    You can supply this probability with ``feed_dict`` in trainer.

    Args:
        input: a 2D node.
        chanels: How many channels are there in the image. (Default = ``1``)
        name: name scope of the layer.  

    Returns:
        tensorflow tensor: An output node           
    """ 
    with tf.variable_scope (name) as scope:
        output = tf.nn.dropout (input, prob)
    return output

if __name__ == '__main__':
    pass  