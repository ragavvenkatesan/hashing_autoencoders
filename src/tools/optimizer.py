import tensorflow as tf

def apply_adagrad(var_list, obj, learning_rate = 0.1):
    """
    Sets up the Adagrad optimizer

    Args:
        var_list: List of variables to optimizer over.        
        obj: Node of the objective to minimize
    Notes:
        learning_rate: What learning rate to run with. (Default = ``0.01``) Set with ``LR``
    """
    back_prop = tf.train.GradientDescentOptimizer(
                                            learning_rate = learning_rate,
                                            name = 'adagrad' ).minimize(loss = obj, \
                                                    var_list = var_list ) 
    return back_prop

def apply_gradient_descent(var_list, obj, learning_rate = 0.01):
    """
    Sets up the gradient descent optimizer

    Args:
        var_list: List of variables to optimizer over.        
        obj: Node of the objective to minimize
    Notes:
        learning_rate: What learning rate to run with. (Default = ``0.01``) Set with ``LR``
    """
    back_prop = tf.train.GradientDescentOptimizer(
                                            learning_rate = learning_rate,
                                            name = 'gradient_descent' ).minimize(loss = obj, \
                                                    var_list = var_list ) 
    return back_prop

def apply_adam (var_list, obj, learning_rate = 0.0001):
    """
    Sets up the ADAM optmiizer

    Args:
        var_list: List of variables to optimizer over.
        obj: Node of the objective to minimize        
    
    Notes:
        learning_rate: What learning rate to run with. (Default = ``0.01``) Set with ``LR``
    """      
    back_prop = tf.train.AdamOptimizer(
                                        learning_rate = learning_rate,
                                        name = 'adam' ).minimize(loss = obj, \
                                            var_list = var_list) 
    return back_prop                                                               

def apply_rmsprop( var_list, obj, learning_rate = 0.0001 ):
    """
    Sets up the RMS Prop optimizer

    Args:
        var_list: List of variables to optimizer over.
        obj: Node of the objective to minimize        

    Notes:
        * learning_rate: What learning rate to run with. (Default = ``0.001``). Set  ``LR``
        * momentum: What is the weight for momentum to run with. (Default = ``0.7``). Set ``MOMENTUM``
        * decay: What rate should learning rate decay. (Default = ``0.95``). Set ``DECAY``            
    """    
    back_prop = tf.train.RMSPropOptimizer(
                                        learning_rate = learning_rate,
                                        # decay = 0.95,
                                        # momentum = MOMENTUM,
                                        name = 'rmsprop' ).minimize(loss = obj, \
                                        var_list = var_list) 
    return back_prop

def apply_weight_decay (var_list, coeff, name = 'weight_decay'):
    """
    This method applys L2 Regularization to all weights and adds it to the ``objectives`` 
    collection. 
    
    Args:
        name: For the tensorflow scope.
        var_list: List of variables to apply.
    
    Notes:
        What is the co-efficient of the L2 weight? Set ``WEIGHT_DECAY_COEFF``.( Default = 0.0001 )
    """                              
    for param in var_list:
        norm = coeff * tf.nn.l2_loss(param)
        tf.summary.scalar('l2_' + param.name, norm)                  
        tf.add_to_collection(name + '_objectives', norm)

def apply_l1 ( var_list, coeff, name = 'l1'):
    """
    This method applys L1 Regularization to all weights and adds it to the ``objectives`` 
    collection. 
    
    Args:
        var_list: List of variables to apply l1
        name: For the tensorflow scope.
    
    Notes:
        What is the co-efficient of the L1 weight? Set ``L1_COEFF``.( Default = 0.0001 )
    """                              
    for param in var_list:
        norm = coeff * tf.reduce_sum(tf.abs(param, name = 'abs'), name = 'l1')
        tf.summary.scalar('l1_' + param.name, norm)                  
        tf.add_to_collection(name + '_objectives', norm)

def process_params(params, name):
    """
    This method adds the params to two collections.
    The first element is added to ``regularizer_worthy_params``.
    The first and second elements are is added to ``trainable_parmas``.

    Args:
        params: List of two.
        name: For the scope
    """
    if len(params) == 2:
        tf.add_to_collection(name + '_trainable_params', params[0])
        tf.add_to_collection(name + '_trainable_params', params[1])         
    else:
        tf.add_to_collection(name + '_trainable_params', params[0])        
    tf.add_to_collection(name + '_regularizer_worthy_params', params[0])       

def apply_regularizer (name, var_list, l2_coeff = 0., l1_coeff = 0.):
    """
    This method applys Regularization to all weights and adds it to the ``objectives`` 
    collection. 
    
    Args:
        var_list: List of variables to apply l1
        name: For the tensorflow scope.
    
    Notes:
        What is the co-efficient of the L1 weight? Set ``L1_COEFF``.( Default = 0.0001 )
    """       
    with tf.variable_scope(name + '_weight-decay') as scope:
        if l2_coeff > 0:
            apply_weight_decay(name = name + 'weight_decay', var_list = var_list, coeff = l2_coeff )

    with tf.variable_scope(name + '_l1-regularization') as scope:
        if l1_coeff > 0:
            apply_l1(name =name + '_weight_decay',  var_list = var_list, coeff = l1_coeff)