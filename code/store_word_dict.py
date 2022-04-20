import mutils
from datasets import load_dataset
import config
import json

'''
This file stores the tokens in the training data in a dict that is saved in a json output file `train_word_dict.json`.
The purpose of this is that these tokens can be loaded when initializing the embedding model 
for evaluating on the test data
'''

print('load dataset')
dataset = load_dataset(config.dataset)

print('start loading in training data', flush=True)
U_train, V_train, Y_train = mutils.get_UVY(dataset['train'])

print('preprocess sentences', flush=True)
U_train = mutils.preprocess_sentence_data(U_train)
V_train = mutils.preprocess_sentence_data(V_train)

print('load word dict', flush=True)
train_sents = U_train + V_train
word_dict = mutils.load_word_dict(train_sents)

#store word dict in json output file
with open('train_word_dict.json', 'w') as fp:
    json.dump(word_dict, fp,  indent=4)

