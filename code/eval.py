import numpy as np

from datasets import load_dataset
import config
import models
import mutils

import sys
import torch
import json

"""
This file can be used to evaluate a trained model on the NLI testset
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(argv=None):
    '''
    Main function that evaluates the given model and returns the dev and test accuracy
    :param model: the name (str) of the model to be evaluated. this can be 'base', 'lstm', 'bilstm' or 'bilstmpool'
    :param nli_path: the path (str) to the nli model
    :param encoder_path: the path (str) to the encoder. This is None if the selected model is 'base'

    :returns dev_acc: the dev accuracy (float)
    :returns eval_acc : the test accuracy (float)
    '''
    if argv is None:
        argv = sys.argv
    
    model = argv[1]
    nli_path = argv[2]
    encoder_path = argv[3]

    print(f"Start evalutating {model} model")
    #Load model
    nli, encoder = mutils.load_model(model, nli_path, encoder_path)

    #for the base model, we take the mean of the embeddings
    if model == 'base':
        take_mean = True
    else:
        take_mean = False

    #dev acc is already computed during training, so extract it
    nli_checkpoint = torch.load(nli_path, map_location=device)
    dev_acc = nli_checkpoint['dev_accuracy']

    #load the data
    dataset = load_dataset(config.dataset)

    #open train vocabulary; load embeddings of tokens in training data
    with open('train_word_dict.json') as json_file:
        word_dict = json.load(json_file)
    embedding_model = mutils.load_embeddings(word_dict)

    #load and preprocess test data
    U_test, V_test, Y_test = mutils.get_UVY(dataset['test'])

    U_test = mutils.preprocess_sentence_data(U_test)
    V_test = mutils.preprocess_sentence_data(V_test)

    #start evaluation
    correct = 0.
    for i in range(0, len(U_test), config.batch_size):
        #batching
        s1 = mutils.get_batch(embedding_model, U_test[i:i + config.batch_size], device, take_mean)
        if len(s1) != config.batch_size:
            continue
        s2 = mutils.get_batch(embedding_model, V_test[i:i + config.batch_size], device, take_mean)
        y_true = torch.tensor(Y_test[i:i + config.batch_size], dtype=torch.long, device=device)

        #base model does not use an encoder
        if model != 'base':
            #pad sentences and make predictions
            (u, u_lengths), (v, v_lengths) = mutils.pad(s1,s2)
            batch_output = nli(u, v, u_lengths, v_lengths)
        else:
            #make predictions
            u = torch.stack(s1)
            v = torch.stack(s2)
            batch_output = nli(u, v)

        #evaluate predictions
        batch_pred = batch_output.data.max(1)[1]
        correct += (batch_pred == y_true).sum().item()

    #compute accuracy
    eval_acc = round(100 * correct / len(U_test),2)
    print(f"dev accuracy : {dev_acc}")
    print(f"test accuracy : {eval_acc}")
    return dev_acc, eval_acc

if __name__ == '__main__':
    main()
