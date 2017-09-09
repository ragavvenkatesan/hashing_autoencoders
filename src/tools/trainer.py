import tensorflow as tf
import numpy as np
from support import hamming, map, profile, one_hot_to_binary

def sample_random(m, n, labels = None, dist = 'gaussian'):
    """
    Returns random numbers concatenated with labels. 
    of size m, n

    Args:
        m: number of samples
        n: Number of dimensions per sample.
        labels: if supplied will be concatenated horizontally.
        dist: Which distribution to sample from.
    """
    if labels is None:
        if dist == 'bernoulli':
            tmp = np.random.randint(low=0, high =2, size=[m, n])
            cond = np.less(tmp, 0.5*np.ones(tmp.shape))
            output = np.where(cond, -1*np.ones(tmp.shape), np.ones(tmp.shape))              
        elif dist == 'gaussian':
            return np.random.normal(loc = 1, scale = 1, size = [m, n])   
        elif dist == 'dirichlet':
            return np.random.dirichlet(s = (3), size = [m, n]) 
    else:
        assert m == labels.shape[0]
        if dist == 'bernoulli':            
            rand = np.random.randint(low=0, high =2, size=[m, n - labels.shape[1]])
            cond = np.less(output, 0.5*np.ones(output.shape))
            output = np.where(cond, -1*np.ones(output.shape), np.ones(output.shape))              
        elif dist == 'gaussian':
            rand = np.random.normal(loc =0, scale = 1, size = [m, n - labels.shape[1]])
        output = np.concatenate ((rand, labels) , axis = 1 )
     
    return output

class trainer(object):
    """
    Trainer for Two networks trained simultaneously

    Args:
        nets: A List of networks.
        dataset: A tensorflow dataset object        
        session: Supply a session to run on, if nothing is provided,
            a new session will be opened. 
        init_vars: if ``True`` will initialize all global variables.            
      
    """
    def __init__ (self, hashing_encoder, dataset, session = None, dropout_prob = 0.5,
                    init_vars = True, tensorboard = 'tensorboard',
                    dist = 'bernoulli'):
        """
        Class constructor
        """
        self.hae = hashing_encoder
        self.dataset = dataset 
        if session is None:
            self.session = tf.InteractiveSession()        
        else:
            self.session = session
        if init_vars is True:
            tf.global_variables_initializer().run()
        self.summaries(name = tensorboard)   
        self.cw = 32 
        self.dropout_prob = dropout_prob
        self.dist = dist
        print dist
        
    def summaries(self, name = "tensorboard"):
        """
        Just creates a summary merge bufer
        Args:
            name: a name for the tensorboard directory
        """
        self.summary = tf.summary.merge_all()
        self.tensorboard = tf.summary.FileWriter(name)
        self.tensorboard.add_graph(self.session.graph)

    def bp_step(self, net, mini_batch_size = 500):
        """
        Sample a minibatch of data and run one step of BP.

        Args:
            mini_batch_size: Integer
            ind: Supply the list index of the net to update.
        
        Returns: 
            tuple of tensors: total objective and cost of that step
        """
        x, y = self.dataset.train.next_batch(mini_batch_size)
        _, obj = self.session.run(  
                           fetches = [  net.back_prop, 
                                        net.obj ], \
                           feed_dict = {self.hae.images:x,
                                        self.hae.targets:y, \
                                        net.dropout_prob: self.dropout_prob,
                                        self.hae.z: sample_random(m = mini_batch_size, n = self.cw,
                                           # labels = one_hot_to_binary(y) 
                                           dist = self.dist,
                                           labels = None)})
        return obj


    def _get_hashes (self, dataset, batch_mode = False, mini_batch_size = 500):
        """
        Return hashes for images

        Args:
            dataset: some feed
            batch_mode: If ``True``, will run the code in mini batches.
            mini_batch_size: Needed if batch mode.

        Returns:
            nd array: hashes           
        """
        if batch_mode is False:
            output = self.session.run (
                            fetches = self.hae.hash, 
                            feed_dict = {self.hae.images:dataset.images})           
        else:
            for ind in range(dataset.images.shape[0]/mini_batch_size):
                tmp = self.session.run (
                                fetches = self.hae.hash, 
                                feed_dict = {self.hae.images:dataset.next_batch(mini_batch_size)[0]})      

                if ind == 0:                
                    output = tmp
                else:
                    output = np.vstack([output, tmp])
        return output 

    #@profile
    def precision (self, top_k=5000, mini_batch_size = 500):
        """
        Return mean average precision       
        Args:
            top_k: top somethiung considered.
            mini_batch_size: Provide mini batch size

        Returns:
            float: precision            
        """
        query_hashes = self._get_hashes (dataset = self.dataset.query)
        gallery_hashes = self._get_hashes (dataset = self.dataset.gallery, 
                                                batch_mode = True,
                                                mini_batch_size = mini_batch_size)
        query_labels = self.dataset.query.labels
        gallery_labels = self.dataset.gallery.labels     
        return map( (query_hashes, query_labels), (gallery_hashes, gallery_labels), m = 3 )

    def write_summary (self, iter = 0, mini_batch_size = 500):
        """
        This method updates the summaries
        
        Args:
            iter: iteration number to index values with.
            mini_batch_size: Mini batch to evaluate on.
        """
 
        x = self.dataset.train.images[:mini_batch_size]
        y = self.dataset.train.labels[:mini_batch_size]
        s = self.session.run(self.summary,   
                                feed_dict = {self.hae.images:x,
                                             self.hae.targets:y, \
                                             self.hae.z: sample_random(m = mini_batch_size, n = self.cw,
                                             #labels = one_hot_to_binary(y)[:mini_batch_size]
                                             dist = self.dist,
                                             labels = None )}  )
        self.tensorboard.add_summary(s, iter)

    def _get_tensor (self, input, net, mini_batch_size = 500, 
                            dataset = None, batch_mode = False):
        """
        Given a input tensor, returns the value of the 
        tensor.

        Args:
            input: Tensorflow tensor
            ind: network index
            dataset: image data
            batch_mode: will run mini_batch wise.
            mini_batch_size: if batch_mode it will run these many mini batches.

        Returns:
            numpy ndarray: tensor output
        """
        if dataset is None:
            dataset = self.dataset.query
        
        if batch_mode is False:
            output = self.session.run(input,   feed_dict = {self.hae.images:dataset.images,
                                        net.dropout_prob: self.dropout_prob,
                                        self.hae.z: sample_random(m = dataset.images.shape[0],
                                        n = self.cw,
                            #labels = one_hot_to_binary(dataset.labels)[:dataset.images.shape[0]]
                            dist = self.dist,
                            labels = None )})
        else:
            for ind in range(dataset.images.shape[0]/mini_batch_size):
                x, y = dataset.next_batch(mini_batch_size)
                tmp = self.session.run(input,   
                                    feed_dict = {self.hae.images:x,
                                        net.dropout_prob: self.dropout_prob,
                                        self.hae.z: sample_random(m = mini_batch_size,
                                        n = self.cw,
                                        #labels = one_hot_to_binary(y)[:mini_batch_size] 
                                        dist = self.dist,
                                        labels = None)})   
                if ind == 0:                
                    output = tmp
                else:
                    output = np.vstack([output, tmp])                                                 
        return output
    
    @profile
    def _unique (self, input, net, dataset = None, batch_mode = False, mini_batch_size = 500):
        """
        Given ``input`` is a matrix of b X d, 
        outputs number of unique vectors (row-wise).        

        Args:
            input: Tensorflow tensor
            ind: network index
            dataset: image data
            batch_mode: If the size is large, all data might not fit in a GPU.
                        using batch mode will save memory.
            mini_batch_size: mini batch size for batch mode.
        
        Returns:
            int: Count of unique rows
        """
        if dataset is None:
            dataset = self.dataset.gallery
        tensor = self._get_tensor(input = input, net = net, dataset = dataset, 
                                    mini_batch_size = mini_batch_size, batch_mode = True)
        unique_tensors =  np.unique(tensor, axis =0)
        return unique_tensors.shape[0]

    def train ( self, 
                k = [1, 1, 1],
                begin_after = [1, 1, 1],
                iter= 10000, 
                mini_batch_size = 500, 
                update_after_iter = 100, 
                print_unique_hashes = True,
                print_precision = False,
                summarize = True):
        """
        Run backprop for ``iter`` iterations

        Args:   
            iter: number of iterations to run
            k: Only update if the iteration is factored by.
            begin_after: Only update net after these many iterations.
            mini_batch_size: Size of the mini batch to process with
            update_after_iter: This is the iteration for validation
            summarize: Tensorboard operation
        """

        obj = [0] * 3
        cost = [0] * 3
        train_acc  = [0] * 3
        acc = [0] * 3
        self.nets = [ self.hae.autoencoder, self.hae.discriminator, self.hae.generator ]
        updates = [0] * 3        
        for it in range(iter):
            for ind in xrange(3):
                if it % k[ind] == 0 and it >= begin_after[ind]:                        
                    obj[ind] = self.bp_step(mini_batch_size = mini_batch_size, net = self.nets[ind])                                            
                    updates [ind] = updates [ind] + 1
            if it % update_after_iter == 0:      
                print ("\n\n")
                print( " Iter " + str(it) )
                for ind in xrange(3):
                    print(" Objective " + str(obj[ind]) )
                if print_unique_hashes is True:
                    unique_hashes = self._unique(self.hae.hash, net = self.hae.autoencoder, 
                                        batch_mode = True, mini_batch_size = mini_batch_size)
                    print ( "Unique Hashes :" + str(unique_hashes)) 
                if print_precision is True:           
                    print ( "Precision " +str(self.precision( mini_batch_size = mini_batch_size) ) )
                if summarize is True:               
                    self.write_summary(iter = it, mini_batch_size = mini_batch_size)
                    print ("Updates [" + str(updates[0]) + " " + str(updates[1]) + " " + str(updates[2]) + "]")
        
        # print ("Final Test Precision "+ str(self.precision(mini_batch_size = mini_batch_size)))    
        self.tensorboard.close()

if __name__ == '__main__':
    pass