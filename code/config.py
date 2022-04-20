"""
This files contains the parameters used during training.
"""

# path to glove embeddings
path_to_embeddings = '../pretrained/glove.840B.300d.txt'

dataset = 'snli'

#model settings
vector_length = 300
lstm_hidden_dim = 2048
lstm_layers = 1
dropout = 0
nli_hidden_dim = 512

#training parameters
batch_size = 64
num_epochs = 20
lr = 0.1
lr_decay = 0.99 
min_lr = 0.01 

seed = 42