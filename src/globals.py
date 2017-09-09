# Some Global Defaults
# Architecture
CONV_1_FILT = (5,5)
CONV_1_N = 20
CONV_2_FILT = (3,3)
CONV_2_N = 50
HIDDEN_1 = 800
HIDDEN_2 = 800
HIDDEN_0 = 1250 # This is calculated as the output of the conv layer going in to 
                # the dot product layers. Do this manually.
                # Make sure you also update the output shapes in the code.
CODEWORD_LENGTH = 32
# Update also on trainer class.

IMAGE_SHAPE = 784
NUM_CLASSES = 10
DROPOUT_PROBABILITY = 0.5

# Optimizer properties of autoencoder
AUTOENCODER_LR = 0.001  # Learning rate 
# 0.001
AUTOENCODER_WEIGHT_DECAY_COEFF = 0.0001 # Co-Efficient for weight decay
AUTOENCODER_L1_COEFF = 0.0001 # Co-Efficient for L1 Norm
# MOMENTUM = 0.7 # Momentum rate 
AUTOENCODER_OPTIMIZER = 'rmsprop' # Optimizer (options include 'adam', 'adagrad', 'sgd',
                               # 'rmsprop') Easy to upgrade if needed. rmsprop
CODE_ACTIVATION = 'identity'

# DECAY = 0.95
AUTOENCODER_HASH_COEFF = 0.
AUTOENCODER_CODEWORD_COEFF = 1e-2
DISTRIBUTION = 'gaussian'

# Optimizer properties of GAN
DIS_GAN_LR = 0.01  # Learning rate 
DIS_GAN_WEIGHT_DECAY_COEFF = 0.0001 # Co-Efficient for weight decay
DIS_GAN_L1_COEFF = 0.0001 # Co-Efficient for L1 Norm
DIS_GAN_OPTIMIZER = 'sgd' # Optimizer (options include 'adam', 'adagrad', 'sgd',
                               # 'rmsprop') Easy to upgrade if needed.


GEN_GAN_LR = 0.01  # Learning rate 
GEN_GAN_WEIGHT_DECAY_COEFF = 0.0001 # Co-Efficient for weight decay
GEN_GAN_L1_COEFF = 0.0001 # Co-Efficient for L1 Norm
GEN_GAN_OPTIMIZER = 'sgd' # Optimizer (options include 'adam', 'adagrad', 'sgd',
                               # 'rmsprop') Easy to upgrade if needed.


# Dataset sizes
TRAIN_SET_SIZE = 50000
QUERY_SET_SIZE = 1000

# Train options
MINI_BATCH_SIZE = 100 # Mini batch size 
UPDATE_AFTER_ITER = (TRAIN_SET_SIZE / MINI_BATCH_SIZE ) * 10 # Update after these many iterations.
ITER = (TRAIN_SET_SIZE / MINI_BATCH_SIZE )  * 100 # Total number of iterations to run

if __name__ == '__main__':
    pass