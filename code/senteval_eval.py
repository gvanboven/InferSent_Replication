import sys
import logging

import torch
import mutils

import json
import models

device = "cpu"
print(f"Using {device} device", flush=True)

GLOVE_PATH = "../pretrained/glove.840B.300d.txt"
PATH_TO_DATA = "../SentEval/data" #set to SentEval data path
PATH_TO_SENTEVAL = "../SentEval" # Set to SentEval path

#import senteval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# define transfer tasks
transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC',
                  'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14']
# define the tasks that have accuracy as performance measure
accuracy_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC',
                  'SICKEntailment', 'MRPC']

# define senteval params
params_senteval = {'usepytorch': True,
                           'task_path': PATH_TO_DATA,
                           'seed': 1111, 'kfold': 5,
                           'batch_size' : 64}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def prepare(params, samples):
    ''''
    Load the embedding model with the tokens from the training data
    '''
    print('load word dict')
    with open('train_word_dict.json') as json_file:
        word_dict = json.load(json_file)

    print('load embedding model', flush=True)
    params['embedding_model'] = mutils.load_embeddings(word_dict)


def batcher(params, batch):
    """
    batch input sentences, apply padding and encode inputs 
    """
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = mutils.get_batch(params['embedding_model'], batch, 'cpu', params['take_mean'])
    if params['model'] != 'base':
        padded_embeddings, embedding_lengths = mutils.pad_singlebatch(embeddings)
        embeddings = params['encoder'](padded_embeddings, embedding_lengths)
    else:
        embeddings = torch.stack(embeddings)

    return embeddings.detach().numpy()

def main(argv=None):
    '''
    Main function that carries out SentEval tasks for a given model and stores the results in an output json
    :param model: the name (str) of the model to be evaluated. this can be 'base', 'lstm', 'bilstm' or 'bilstmpool'
    :param encoder_path: the path (str) to the encoder. This is None if the selected model is 'base'
    :param output_file: the path (str) to the json file in which the results should be stored
    '''

    if argv is None:
        argv = sys.argv

    params_senteval['model'] = argv[1]
    encoder_path = argv[2]
    output_file = argv[3]

    #load encoder model
    if params_senteval['model'] == 'base':
        take_mean = True
        encoder = None
    else:
        take_mean = False
        encoder_checkpoint = torch.load(encoder_path, map_location=device)
        if params_senteval['model'] == 'lstm':
            print("load lstm encoder")
            encoder = models.LSTM(device)

        elif params_senteval['model'] == 'bilstm':
            print("load bilstm encoder")
            encoder = models.BiLSTM(device)

        elif params_senteval['model'] == 'bilstmpool':
            print("load bilstm maxpool encoder")
            encoder = models.BiLSTMpool(device)

        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        encoder.eval()

    params_senteval['take_mean'] = take_mean
    params_senteval['encoder'] = encoder

    # execute sent evaluation for defined tasks
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)

    #store the scores for the tasks with accuracy as its measure in an output json file
    accuracy_results = {k:v for k,v in results_transfer.items() if k in accuracy_tasks}
    with open(output_file, 'w') as fp:
        json.dump(accuracy_results, fp,  indent=4)

if __name__ == "__main__":
    main()