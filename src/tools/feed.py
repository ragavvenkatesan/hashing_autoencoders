class feed (object):
    """
    Template for datasets

    Attributes:
        images: Contains data
        labels: Contains data
        next_batch: Provides the next batch when supplied with next batch.
        current_ind: Int that keeps track of current index of dataset.
        self.dataset_size: int that checks the size of the dataset.
    """
    def __init__(self, images, labels):
        """
        Class Constructor
        """
        self.images = images
        self.labels = labels
        self.current_ind = 0        
        self.dataset_size = len(self.labels)        

    def next_batch(self, mini_batch_size):
        """
        Returns mini batch

        Args:
            mini_batch_size: Int
        
        Returns:
            tuple: ``(images, labels)`` of size ``mini_batch_size``.
        """
        assert ( mini_batch_size <= self.dataset_size )            
        if ( self.current_ind + mini_batch_size )  > self.dataset_size:
            self.current_ind = mini_batch_size
        else:
            self.current_ind = self.current_ind + mini_batch_size                    
        x = self.images [self.current_ind - mini_batch_size : self.current_ind ]
        y = self.labels [self.current_ind - mini_batch_size : self.current_ind ]                
        return (x ,y) 