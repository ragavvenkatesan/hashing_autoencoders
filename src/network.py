import tensorflow as tf
import numpy as np
import sys 
sys.path.append('../')

from tools.layers import *
from tools.support import visualize_images, visualize_1D_filters, log, rmse
from tools.optimizer import *
from globals import *

def process_hash_regularizer(input, coeff = 0.001, name ='hash'):
    """
    This method adds the params to hash collections.
    The first and second elements are is added to ``hash_params``.

    Args:
        params: List of two.
        name: For the scope
    """
    with tf.variable_scope(name) as scope:
        if coeff > 0:
            hash_regularizer = tf.nn.l2_loss( tf.abs(input) - 1 )            
            tf.summary.scalar(name + '_hash_penalty', hash_regularizer)            
            hash_regularizer_obj = coeff * hash_regularizer
            tf.add_to_collection(name + '_objectives', hash_regularizer_obj)
            tf.summary.scalar(name + '_weighted_hash_penalty', hash_regularizer_obj)

    for i in range(CODEWORD_LENGTH):
        tf.summary.histogram(name + '_hash_code' + str(i),input[:,i])

def process_codeword_normalization_regularizer(input, coeff = 0.001, tolerance = 3, name ='codeword'):
    """
    Penalizes the network for learning codewords greater than a tolerance.

    Args:
        params: List of two.
        tolerance: 10 is default.
        name: For the scope
    """
    with tf.variable_scope(name) as scope:
        if coeff > 0:
            cond = tf.greater (tf.abs(input), tolerance*tf.ones(tf.shape(input)))
            regularize_params = tf.where(cond, input, tf.zeros(tf.shape(input)))
            codeword_regularizer = tf.reduce_mean(tf.abs( regularize_params ))     
            tf.summary.scalar(name + '_codeword_penalty', codeword_regularizer)            
            codeword_regularizer_obj = coeff * codeword_regularizer
            tf.add_to_collection(name + '_objectives', codeword_regularizer_obj)
            tf.summary.scalar(name + '_weighted_codeword_penalty', codeword_regularizer_obj)

    for i in range(CODEWORD_LENGTH):
        tf.summary.histogram(name + '_codeowrd_code' + str(i),input[:,i])

            
class network (object):
    """
    Abstract Template class. Inherit this for building other networks.

    Args:
        images: Placeholder for images feed.
        target: Placeholder for target labels feed. 
                If ``None`` or not provided, will create a dummy.
        name: Name of the network.  

    Class Properties:
        These are variables of the class that are available outside. 
        
        *   ``images``: This is the placeholder for images. This needs to be fed in.
        *   ``name``: The name scope of the network. 
        *   ``dropout_prob``: This is also a placeholder for dropout probability. 
              This needs to be fed in.    
        *   ``back_prop``: Running this will make take one back prop step. 
        *   ``obj``: Is a cumulative objective tensor.
        *   ``params`` : Parameters of the model.
        *   ``targets`` : Labels placeholder to be fed in.
    """
    def __init__ (  self,
                    images = None,
                    targets = None,
                    name = 'network' ):
        """
        Class constructor. Creates the model and all the connections. 
        """
        with tf.variable_scope(name + '_feed') as scope:
            self.images = images
            self.name = name    
            self.back_prop = None
            # Unflatten Layer

            images_square = unflatten_layer ( self.images )
            visualize_images(images_square)
            # Placeholder probability for dropouts.
            self.dropout_prob = tf.placeholder_with_default(
                                                input = tf.constant(1.0, dtype = tf.float32),
                                                shape = None,
                                                name = 'dropout_probability')
            self.targets = targets
            if self.targets is None:
                self.targets = tf.placeholder(tf.float32, shape = [None, 10], name = 'targets')

        self.obj = 0.
        self.parmas = []
        self._build() # This method needs to be written in the new class.

    def _cook_optimizer(self, 
                        lr = 0.01, 
                        optimizer = 'sgd',
                        l1_coeff = 0.00001,
                        l2_coeff = 0.00001):
        """
        This is a template for building an optimizer.
        
        Args:
            lr: Learning rate.
            optimizer: Option for what optimizer to use.
                    * ``'sgd'`` - Stochastic Gradient Descent.
                    * ``'rmsprop'`` - RMSPROP
                    * ``'adam'`` - ADAM
                    * ``'adagrad'`` - ADAGRAD
            l1_coeff: Weight for the L1 regularizer of parameters.
            l2_coeff: Weight for the L2 regularizer of parameters.

        """
        with tf.variable_scope (self.name + '_train') as scope:
            apply_regularizer (name = self.name, var_list = tf.get_collection(
                                                    self.name + '_regularizer_worthy_params'), 
                                                    l1_coeff = l1_coeff,
                                                    l2_coeff = l2_coeff  )
            self.obj = tf.add_n(tf.get_collection( self.name + '_objectives'), name='objective')
            tf.summary.scalar('total_objective', self.obj)

            # Change (supply as arguments) parameters here directly in the code.
            if optimizer == 'sgd':                                                                              
                self.back_prop = apply_gradient_descent(var_list = tf.get_collection(
                                        self.name + '_trainable_params'),
                                        obj = self.obj,  learning_rate = lr )
            elif optimizer == 'adagrad':                                                                              
                self.back_prop = apply_adagrad(var_list = tf.get_collection(
                                        self.name + '_trainable_params'),
                                        obj = self.obj,  learning_rate = lr )                                        
            elif optimizer == 'rmsprop':
                self.back_prop = apply_rmsprop(var_list = tf.get_collection(
                                        self.name + '_trainable_params') ,
                                        obj = self.obj,  learning_rate = lr)
            elif optimizer == 'adam':
                self.back_prop = apply_adam (var_list = tf.get_collection(
                                        self.name + '_trainable_params') ,
                                        obj = self.obj,  learning_rate = lr )
            else:
                raise Error('Invalid entry to optimizer')   

class gan_generator(network):
    """
    Definition of the Generator class of networks.
    Inherited from :class:`network.network`.
    
    Args:
        name: Name of the network.

    Class Properties:
        These are new variables of the class that are available outside. 
        

    """
    def __init__ (  self,
                    images = None,
                    targets = None,
                    z = None,
                    input_params = None,
                    name = 'network' ):
        """
        Class constructor. Creates the model and all the connections. 
        """
        with tf.variable_scope(name + '_feed') as scope:
            self.name = name    
            self.back_prop = None
            # Unflatten Layer
            # Placeholder probability for dropouts.
            self.dropout_prob = tf.placeholder_with_default(
                                                input = tf.constant(1.0, dtype = tf.float32),
                                                shape = None,
                                                name = 'dropout_probability')
            self.targets = targets
            if self.targets is None:
                self.targets = tf.placeholder(tf.float32, shape = [None, 10], name = 'dummy')

        self.obj = 0.
        self.parmas = []

        self._build(z = z, targets = targets, input_params = input_params) # This method needs to be written in the new class.

    def _build(self, z, targets= None, input_params = None):
        """
        This method builds the network architecture.

        Args:
            input_params: list of paramers.
        """
        with tf.variable_scope (self.name + '_architecture') as scope:
            with tf.variable_scope ( 'generator') as scope:
                self.z = z
                self.targets =targets

                # Dot Product Layer 1
                par = input_params[0]
                process_params(par, name = self.name)    
                par[0] = tf.transpose(par[0])  
                # Decoder ... 
                decoder_1_out, params = dot_product_layer  (  input = self.z, 
                                                        neurons = HIDDEN_2,
                                                        params = par,
                                                        name = 'gen_decoder_dot_1')
                g1_params = params
 
                dec_1_out_dropout = dropout_layer ( input = decoder_1_out,
                                                prob = self.dropout_prob,
                                                name = 'gen_dropout_1')

                # Dot Product Layer 2                 
                par = input_params[1]
                process_params(par, name = self.name)    
                par[0] = tf.transpose(par[0])  
                decoder_2_out, params = dot_product_layer  (  input = dec_1_out_dropout, 
                                                        neurons = HIDDEN_1,
                                                        params = par,
                                                        name = 'gen_decoder_dot_2')
                g2_params = params

                dec_2_out_dropout = dropout_layer ( input = decoder_2_out,
                                            prob = self.dropout_prob,
                                            name = 'gen_dropout_2')

                # Dot Product Layer 3                 
                par = input_params[2]
                process_params(par, name = self.name)    
                par[0] = tf.transpose(par[0])  
                decoder_3_out, params = dot_product_layer  (  input = dec_2_out_dropout, 
                                                        neurons = 1250,
                                                        params = par,
                                                        name = 'gen_decoder_dot_0')
                g3_params = params
                dec_3_square = unflatten_layer ( decoder_3_out, channels = CONV_2_N,
                                                name = 'gen_unflatten'  )
                upsample_1 = upsampling_layer (dec_3_square, size = (10,10), 
                                                name = 'gen_upsampling_1')

                # Convolution layer 1
                par = input_params[3]
                process_params(par, name = self.name)                               
                deconv1_out, params =  deconv_2d_layer (    input = upsample_1,
                                                        neurons = CONV_1_N,
                                                        filter_size = CONV_2_FILT,
                                                        output_shape = (12,12),
                                                        params = par,                                                    
                                                        name = 'deconv_1' )

                g4_params = params

                # DeConv Layer 2
                par = input_params[4]
                process_params(par, name = self.name)
                upsample_2 = upsampling_layer (deconv1_out, size = (24,24), 
                                                name = 'gen_upsampling_2')
                generated_images_square, params =  deconv_2d_layer (    
                                                        input = upsample_2,
                                                        neurons = 1,
                                                        filter_size = CONV_1_FILT,
                                                        stride = (1,1,1,1),
                                                        output_shape = (28,28),
                                                        params = par,   
                                                        activation = 'tanh',                                                 
                                                        name = 'conv_2' )
  
                g5_params = params
                self.generation = flatten_layer (generated_images_square, in_shp = [-1, 28, 28, 1])
                visualize_images(generated_images_square, name = 'generated_images')
            self.params = [g1_params, g2_params, g3_params, g4_params, g5_params]            

    def cook(self, fake):
        """ Supply fake from discriminator

        Args:
            fake: tensor node from discriminator
        """
        
        with tf.variable_scope (self.name + '_objectives') as scope:        
            with tf.variable_scope( self.name + 'discriminator_obj') as scope: 
                # if targets is none, then think of this as simple GAN.
                if self.targets is None:          
                    # generator_obj = - 0.5 * tf.reduce_mean(log(fake))
                    generator_obj = 0.5 * tf.reduce_mean( (fake-1) **2 )
                else:
                    generator_obj = tf.nn.softmax_cross_entropy_with_logits(labels = self.targets, 
                                                            logits = fake, name = self.name)
                tf.summary.scalar('generator_obj', generator_obj)
                tf.add_to_collection( self.name + '_objectives', generator_obj )   
        with tf.variable_scope (self.name + '_probabilites') as scope:                                                                                                                                            
            tf.summary.scalar('fake_probability', tf.reduce_mean(fake))

        self._cook_optimizer( 
                            lr = GEN_GAN_LR, 
                            optimizer = GEN_GAN_OPTIMIZER,
                            l1_coeff = GEN_GAN_L1_COEFF,
                            l2_coeff = GEN_GAN_WEIGHT_DECAY_COEFF)

class gan_discriminator(network):
    """
    Definition of the Generator class of networks.
    Inherited from :class:`network.network`.
    
    Args:
        name: Name of the network.

    Class Properties:
        These are new variables of the class that are available outside. 
        

    """
    def __init__ (  self,
                    images = None,
                    targets = None,
                    generation = None,
                    name = 'network' ):
        """
        Class constructor. Creates the model and all the connections. 
        """
        with tf.variable_scope(name + '_feed') as scope:
            self.images = images
            self.name = name    
            self.back_prop = None
            # Unflatten Layer
            images_square = unflatten_layer ( self.images )
            visualize_images(images_square)
            # Placeholder probability for dropouts.
            self.dropout_prob = tf.placeholder_with_default(
                                                input = tf.constant(1.0, dtype = tf.float32),
                                                shape = None,
                                                name = 'dropout_probability')
            self.targets = targets
            if self.targets is None:
                self.targets = tf.placeholder(tf.float32, shape = [None, 10], name = 'dummy')

        self.obj = 0.
        self.parmas = []
        self._build(generation = generation) # This method needs to be written in the new class.

    def _build(self, generation):
        """
        This method builds the network architecture.

        Args:
            input_params: list of paramers.
        """
        with tf.variable_scope ('discriminator') as scope:
 
            image_unflatten = unflatten_layer ( self.images )
            gen_unflatten = unflatten_layer ( generation )

            # Conv Layer 1 - image
            conv1_out_image, params =  conv_2d_layer (
                                                input = image_unflatten,
                                                neurons = CONV_1_N,
                                                filter_size = CONV_1_FILT,
                                                name = 'conv_1_img',
                                                visualize = True )            
            pool1_out_img = max_pool_2d_layer ( input = conv1_out_image, name = 'pool_1_img')
            lrn1_out_img = local_response_normalization_layer (pool1_out_img, name = 'lrn_1_img' ) 
            
            # Conv Layer 1 - gen
            conv1_out_gen, params =  conv_2d_layer (
                                                input = gen_unflatten,
                                                neurons = CONV_1_N,
                                                filter_size = CONV_1_FILT,
                                                params = params,
                                                name = 'conv_1_gen',
                                                visualize = False )

            pool1_out_gen = max_pool_2d_layer ( input = conv1_out_gen, name = 'pool_1_gen')
            lrn1_out_gen = local_response_normalization_layer (pool1_out_gen, name = 'lrn_1_gen' )                                          
            process_params(params, name = self.name)
            c1_params = params





            # Conv Layer 2 - image
            conv2_out_image, params =  conv_2d_layer (
                                                input = lrn1_out_img,
                                                neurons = CONV_2_N,
                                                filter_size = CONV_2_FILT,
                                                name = 'conv_2_img' )

            pool2_out_img = max_pool_2d_layer ( input = conv2_out_image, name = 'pool_2_img')
            lrn2_out_img = local_response_normalization_layer (pool2_out_img, name = 'lrn_2_img' ) 


            # Conv Layer 2 - gen
            conv2_out_gen, params =  conv_2d_layer (
                                                input = lrn1_out_gen,
                                                neurons = CONV_2_N,
                                                filter_size = CONV_2_FILT,
                                                params = params,
                                                name = 'conv_2_gen' )

            pool2_out_gen = max_pool_2d_layer ( input = conv2_out_gen, name = 'pool_2_gen')
            lrn2_out_gen = local_response_normalization_layer (pool2_out_gen, name = 'lrn_2_gen' )                                          
            process_params(params, name = self.name)
            c2_params = params

            # Dropout Layer
            flat_gen = flatten_layer(lrn2_out_gen)
            flat_img = flatten_layer(lrn2_out_img)

            flat_gen_dropout = dropout_layer ( input = flat_gen,
                                            prob = self.dropout_prob,
                                            name = 'dropout_1_gen')                  

            flat_img_dropout = dropout_layer ( input = flat_img,
                                            prob = self.dropout_prob,
                                            name = 'dropout_1_img')  



            # Dot Product Layer 1 -img
            fc1_out_img, params = dot_product_layer  (  input = flat_img_dropout,
                                                    neurons = HIDDEN_1,
                                                    name = 'image_disc_dot_1')
            # Dot Product Layer 1 - gen
            fc1_out_gen, params = dot_product_layer  (  input = flat_gen_dropout,
                                                    params = params,
                                                    neurons = HIDDEN_2,
                                                    name = 'gen_disc_dot_1')

            process_params(params, name = self.name)
            d1_params = params
            
            ##
            fc1_out_gen_dropout = dropout_layer ( input = fc1_out_gen,
                                            prob = self.dropout_prob,
                                            name = 'dropout_2_gen')                                          
            fc1_out_img_dropout = dropout_layer ( input = fc1_out_img,
                                            prob = self.dropout_prob,
                                            name = 'dropout_2_img')

            # Dot Product Layer 2 -img
            fc2_out_img, params = dot_product_layer  (  input = fc1_out_img_dropout,
                                                    neurons = HIDDEN_2,
                                                    name = 'image_disc_dot_2')
            # Dot Product Layer 2 - gen
            fc2_out_gen, params = dot_product_layer  (  input = fc1_out_gen_dropout,
                                                    params = params,
                                                    neurons = HIDDEN_2,
                                                    name = 'gen_disc_dot_2')
            process_params(params, name = self.name)
            d2_params = params

            ##
            fc2_out_gen_dropout = dropout_layer ( input = fc2_out_gen,
                                            prob = self.dropout_prob,
                                            name = 'dropout_3_gen')                                          
            fc2_out_img_dropout = dropout_layer ( input = fc2_out_img,
                                            prob = self.dropout_prob,
                                            name = 'dropout_3_img')

            # Dot Product Layer 1 -img
            self.real, params = dot_product_layer  (  input = fc2_out_img_dropout,
                                                    neurons = 1,
                                                    activation = 'sigmoid',
                                                    name = 'real')
            # Dot Product Layer 1 -gen
            self.fake, params = dot_product_layer  (  input = fc2_out_gen_dropout,
                                                    params = params,
                                                    neurons = 1,
                                                    activation = 'sigmoid',
                                                    name = 'fake')

            process_params(params, name = self.name)
            d3_params = params
            self.params = [c1_params, c2_params, d1_params, d2_params, d3_params]            


        with tf.variable_scope (self.name + '_objectives') as scope:        
            with tf.variable_scope( self.name + 'discriminator_obj') as scope:           
                # discriminator_obj = - 0.5 * tf.reduce_mean(log(self.real)) - \
                #                             0.5 * tf.reduce_mean(log(1-self.fake))
                discriminator_obj =  0.5 * tf.reduce_mean ((self.real-1)**2) + \
                                            0.5 * tf.reduce_mean ((self.fake)**2)
                tf.summary.scalar('discriminator_obj', discriminator_obj)
                tf.add_to_collection( self.name + '_objectives', discriminator_obj ) 

        with tf.variable_scope (self.name + '_probabilites') as scope:                                                                           
            tf.summary.scalar('fake_probability', tf.reduce_mean(self.fake))
            tf.summary.scalar('real_probability', tf.reduce_mean(self.real))
                
        self._cook_optimizer( 
                            lr = DIS_GAN_LR, 
                            optimizer = DIS_GAN_OPTIMIZER,
                            l1_coeff = DIS_GAN_L1_COEFF,
                            l2_coeff = DIS_GAN_WEIGHT_DECAY_COEFF)

class autoencoder(network):
    """
    Definition of the Autoencoder class of networks.
    Inherited from :class:`network.network`.
    
    Args:
        images: Placeholder for images
        name: Name of the network.

    Class Properties:
        These are new variables of the class that are available outside. 
        
        *   ``decoded``: Output of the decoder node.
        *   ``codeword`` : The codewords of images supplied.
    """
    def _build(self):
        """
        This method builds the network architecture.
        """
        with tf.variable_scope (self.name + '_architecutre') as scope:
            images_square = unflatten_layer ( self.images )
            visualize_images(images_square)

            # Conv Layer 1
            conv1_out, params =  conv_2d_layer (    input = images_square,
                                                    neurons = CONV_1_N,
                                                    filter_size = CONV_1_FILT,
                                                    name = 'enc_conv_1',
                                                    visualize = True )
            process_params(params, name = self.name)
            e1_params = params
            pool1_out = max_pool_2d_layer ( input = conv1_out, name = 'enc_pool_1')
            # lrn1_out = local_response_normalization_layer (pool1_out, name = 'lrn_1' )

            # Conv Layer 2
            conv2_out, params =  conv_2d_layer (    input = pool1_out,
                                                    neurons = CONV_2_N,
                                                    filter_size = CONV_2_FILT,
                                                    name = 'enc_conv_2' )
            process_params(params, name = self.name)
            e2_params = params
            pool2_out = max_pool_2d_layer ( input = conv2_out, name = 'enc_pool_2')
            # lrn2_out = local_response_normalization_layer (pool2_out, name = 'lrn_2' )

            flattened = flatten_layer(pool2_out)

            # Dropout Layer 1 
            flattened_dropout = dropout_layer ( input = flattened,
                                                prob = self.dropout_prob,
                                                name = 'enc_dropout_1')                                          

            # Dot Product Layer 1
            fc1_out, params = dot_product_layer  (  input = flattened_dropout,
                                                    neurons = HIDDEN_1,
                                                    name = 'enc_dot_1')
            process_params(params, name = self.name)
            e3_params = params 

            # Dropout Layer 2 
            fc1_out_dropout = dropout_layer ( input = fc1_out,
                                            prob = self.dropout_prob,
                                            name = 'enc_dropout_2')
            # Dot Product Layer 2
            fc2_out, params = dot_product_layer  (  input = fc1_out_dropout, 
                                                    neurons = HIDDEN_2,
                                                    name = 'enc_dot_2')
            process_params(params, name = self.name)
            e4_params = params 

            # Dropout Layer 3 
            fc2_out_dropout = dropout_layer ( input = fc2_out,
                                            prob = self.dropout_prob,
                                            name = 'enc_dropout_3')
            
            # Dot Product Layer 2
            self.codeword, params = dot_product_layer  (  input = fc2_out_dropout, 
                                                    neurons = CODEWORD_LENGTH,
                                                    activation = CODE_ACTIVATION,
                                                    name = 'enc_dot_2')
            process_params(params, name = self.name)
            process_codeword_normalization_regularizer(self.codeword, 
                                            coeff = AUTOENCODER_CODEWORD_COEFF,
                                            name = self.name)
            e5_params = params 
            # tf.summary.histogram('codewords', self.codeword)
            # self.hash = threshold_layer ( input = self.codeword,
            #                                 name = 'hash')
            # process_hash_regularizer(self.codeword, coeff = AUTOENCODER_HASH_COEFF,
            #                                name = self.name)

            # Decoder ... 
            decoder_1_out, params = dot_product_layer  (  input = self.codeword, 
                                                    neurons = HIDDEN_2,
                                                    params = [tf.transpose(e5_params[0]), None],
                                                    name = 'decoder_dot_1')
            d1_params = params
            process_params([params[1]], name = self.name)
 
            dec_1_out_dropout = dropout_layer ( input = decoder_1_out,
                                            prob = self.dropout_prob,
                                            name = 'dec_dropout_1')

            decoder_2_out, params = dot_product_layer  (  input = dec_1_out_dropout, 
                                                    neurons = HIDDEN_1,
                                                    params = [tf.transpose(e4_params[0]), None],
                                                    name = 'decoder_dot_2')
            d2_params = params
            process_params([params[1]], name = self.name)
            
            # dropout 2
            dec_2_out_dropout = dropout_layer ( input = decoder_2_out,
                                            prob = self.dropout_prob,
                                            name = 'dec_dropout_2')

            decoder_3_out, params = dot_product_layer  (  input = dec_2_out_dropout, 
                                                    neurons = 1250,
                                                    params = [tf.transpose(e3_params[0]), None],
                                                    name = 'decoder_dot_3')
            d3_params = params
            process_params([params[1]], name = self.name)

            # DeConv Layer 1
            # The output shapes need to be changed according to architecture.

            dec_3_square = unflatten_layer ( decoder_3_out, channels = CONV_2_N  )
            upsample_1 = upsampling_layer (dec_3_square, size = (10,10), name = 'dec_upsampling_1')

            deconv1_out, params =  deconv_2d_layer (    input = upsample_1,
                                                    neurons = CONV_1_N,
                                                    filter_size = CONV_2_FILT,
                                                    output_shape = (12,12),
                                                    # n_outs = MINI_BATCH_SIZE,
                                                    stride = (1,1,1,1),                                                    
                                                    params = [e2_params[0], None],                                                    
                                                    name = 'dec_deconv_1' )

            process_params([params[1]], name = self.name)
            d4_params = params

            # DeConv Layer 2
            upsample_2 = upsampling_layer (deconv1_out, size = (24,24), name = 'dec_upsampling_2')
            decoded_images_square, params =  deconv_2d_layer (    input = upsample_2,
                                                    neurons = 1,
                                                    filter_size = CONV_1_FILT,
                                                    stride = (1,1,1,1),
                                                    output_shape = (28,28),
                                                    # n_outs = MINI_BATCH_SIZE,                                                    
                                                    params = [e1_params[0], None], 
                                                    activation = 'tanh',                                                   
                                                    name = 'dec_deconv_2' )
  
            process_params([params[1]], name = self.name)
            d5_params = params            
                 
            self.decoded = flatten_layer (decoded_images_square, in_shp = [-1, 28, 28, 1])
            visualize_images(decoded_images_square, name = 'decoded')
            # This is because transpose don't initialize.
            self.params = [     [e5_params[0], d1_params[1] ],
                                [e4_params[0], d2_params[1] ],
                                [e3_params[0], d3_params[1] ],
                                [e2_params[0], d4_params[1] ],
                                [e1_params[0], d5_params[1] ]    ]

        with tf.variable_scope (self.name + '_objectives') as scope:        
            with tf.variable_scope( self.name + '_decoder_error') as scope:
                reconstruction_error =  rmse(self.images, self.decoded)                                         
                tf.add_to_collection( self.name + '_objectives', reconstruction_error )                                                    
                tf.summary.scalar('reconstruction_error', reconstruction_error)

            self._cook_optimizer( 
                                lr = AUTOENCODER_LR, 
                                optimizer = AUTOENCODER_OPTIMIZER,
                                l1_coeff = AUTOENCODER_L1_COEFF,
                                l2_coeff = AUTOENCODER_WEIGHT_DECAY_COEFF)
        
class hashing_autoencoder(network):
    """
    Definition of the expert class of networks.

    Notes:
    
    Args:
        images: Placeholder for images
        name: Name of the network.

    Class Properties:
        * autoencoder: An Autoencoder network.
        * gan: A GAN network.
    """
    def _build(self):
        """
        This builds the networks
        """
        self.autoencoder = autoencoder(
                                    images = self.images,
                                    name = 'autoencoder')
        # self.hash = self.autoencoder.hash
        self.z = tf.placeholder(tf.float32, shape = [None, CODEWORD_LENGTH], 
                                    name = 'z-layer')         
        with tf.variable_scope('latent_space') as scope:
            for i in range(CODEWORD_LENGTH):
                tf.summary.histogram('z_' + str(i) ,self.z[:,i])                                    
                tf.summary.histogram('codeword_' + str(i), self.autoencoder.codeword[:,i])
        self.generator = gan_generator( z = self.z,
                        input_params = self.autoencoder.params,
                        name = 'gan_generator')
        self.discriminator = gan_discriminator ( images = self.images,
                            generation = self.generator.generation,
                            name = 'gan_discriminator' )
        self.generator.cook(fake = self.discriminator.fake)
        mean_z, var_z = tf.nn.moments(self.z, axes=[0])
        mean_codeword, var_codeword = tf.nn.moments(self.autoencoder.codeword, axes = [0])

        with tf.variable_scope ('divergence') as scope:
            tf.summary.scalar( tensor = tf.nn.l2_loss(mean_z-mean_codeword), name = 'mean divergence')
            tf.summary.scalar( tensor = tf.nn.l2_loss(var_z-var_codeword), name = 'variance divergence')
        # divergence = self.hash 

if __name__ == '__main__':
    pass                    