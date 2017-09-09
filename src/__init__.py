from tools.trainer import trainer

from dataset import hash_mnist, hash_mnist_fashion
from network import hashing_autoencoder    
from globals import *


if __name__ == '__main__':
    # dataset = hash_mnist_fashion()
    dataset = hash_mnist()
    hashing_nets = hashing_autoencoder(
                        images = dataset.images,
                        targets = dataset.labels,
                        name = 'hashing_autoencoder')

    run = trainer (     hashing_encoder = hashing_nets,
                        dataset = dataset.feed,
                        dropout_prob = DROPOUT_PROBABILITY,
                        dist = DISTRIBUTION,
                        tensorboard = 'autoencoder')
    run.train(  k = [5, 1, 3],
                begin_after = [1, 1, 1],
                print_unique_hashes = False,
                print_precision = False,
                summarize = True  ) 