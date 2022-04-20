import numpy as np
import config
from nltk.tokenize import word_tokenize
import torch
from torch import nn
import models

device = "cuda" if torch.cuda.is_available() else "cpu"

output_dict = {0: 'entailment', 1: 'neutral', 2:'contradiction'}

def load_embeddings(word_dict):
    """
    loads the embeddings for all the tokens in the training data as strores them in a dict
    :param word_dict: dict containing all tokens in the training data

    :returns embeddings_model: dict with the embeddings for all tokens in the training data
    """
    embedding_model = {}
    #Load  embeddings
    with open(config.path_to_embeddings, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(' ', maxsplit=1)
            coefs = np.fromstring(coefs, sep=" ")
            if word in word_dict:
                embedding_model[word] = np.array(coefs)
    return embedding_model

def get_UVY(dataset):
    """
    splits a given train/dev or test dataset into the premises (U), hypotheses (V) and labels (Y)
    """
    U = dataset['premise']
    V = dataset['hypothesis']
    Y = [np.abs(label) for label in dataset['label']] #assert there are no accidental negative numbers in the labels
    return U, V, Y

def preprocess_sentence_data(dataset):
    """
    preprocesses a list of strings by tokenizing the sentences and lowercasing all the tokens
    """
    dataset = [word_tokenize(sent.lower()) for sent in dataset]
    return dataset


def get_word_embedding(embedding_model, token):
    """
    returns the word embedding for a given token. If the given token is unknown, return a vector of zeros
    :param embedding_model: a dict containing all word verctors for tokens in the training data
    :param token: a string for the current token

    :returns vector: returns the embedding vector
    :type vector: numpy array
    """
    if token in embedding_model:
        vector = embedding_model[token]
        if len(vector) != config.vector_length:
            print(token, vector)
    else:
        vector = np.zeros(config.vector_length)
    return vector

def get_sent_embedding(embedding_model, tokenized_sent, device, take_mean = False):
    '''
    returns the embedding for a sentence  

    :param embedding_model: a dict containing all word verctors for tokens in the training data
    :param tokenized_sent: list of tokens (strings)
    :param take_mean: bool, indicating whether the mean of the token embeddings should be taken
                        as the senence embedding
    
    :returns sent_embedding: the embedding for the sentence. 
    :type sent_embedding: tensor.float
    '''
    sent_embedding = []
    for token in tokenized_sent:
        # [sent_len x embedding dim]
        sent_embedding.append(get_word_embedding(embedding_model, token))
    if take_mean:
        # [embedding dim]
        sent_embedding = np.mean(sent_embedding, 0)
    return torch.tensor(np.array(sent_embedding), dtype=torch.float, device=device)

def get_batch(embedding_model, batch, device, take_mean = False):
    """
    Creates a batch by extracting the sentence embedding for each sentence in the batch
    :param embedding_model: a dict containing all word verctors for tokens in the training data
    :param batch: list of strings containing current batch
    :param take_mean: bool, indicating whether the mean of the token embeddings should be taken
                        as the senence embedding

    :returns batch_embeddings: sentence vectors for entire batch
    :type batch_embedidngs: list of tensors
    """
    batch_embeddings = []
    for sent in batch:
        #tensor [sent_len x embedding dim] if take_mean is false else [embedding dim]
        sent_embedding = get_sent_embedding(embedding_model, sent, device, take_mean)
        batch_embeddings.append(sent_embedding)

    return batch_embeddings


def create_batches(embedding_model, data, device, sentences = False, take_mean = False):
    """
    :returns batches: list of tensors
    """
    batches = []
    #create batches splits:
    for i in range(0, len(data), config.batch_size):
        batch = data[i:i + config.batch_size]
        # retrieve embeddings if data contains sentences
        if sentences:
            batch_embedding = get_batch(embedding_model, batch, device, take_mean)
            batches.append(batch_embedding)
        else:
            # convert the targets to tensors directly
            batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)
            batches.append(batch_tensor)
    
    #skip last batch if it is incomplete
    if len(batches[-1]) != config.batch_size:
        batches = batches[:-1]
  
    return batches

def load_word_dict(preprocessed_data):
    '''
    Create a dict containing all words that should be present in the embeddings as keys
    '''
    word_dict = {}
    for sent in preprocessed_data:
        for token in sent:
            if token not in word_dict:
                word_dict[token] = ''
    #make sure to add BOS and EOS tokens
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict

def pad(s1, s2, batch_size=config.batch_size):
    '''
    Zero Pad two lists of preprocessed sentences to the longest sequence lengths in both lists, 
    then splits the padded sequences again into two tensors of [seq len x batch size x embedding dim]
    returns for both sequence tensors both the tensors and the length of the sequences
    '''

    #extract sequence lengths
    s1_lengths = np.array([len(x) for x in s1])
    s2_lengths = np.array([len(x) for x in s2])
    #combine hypotheses and premises to pad consistently
    padded_sequences = nn.utils.rnn.pad_sequence(s1+s2)
    #and split again
    s1_padded = padded_sequences[:, :batch_size, :]
    s2_padded = padded_sequences[:, batch_size:, :]

    return (s1_padded, s1_lengths), (s2_padded, s2_lengths)

def pad_singlebatch(batch):
    '''
    Zero pads one lists of preprocessed sentences to the longest sequence length in the lists, 
    returns the tensors ([seq len x batch size x embedding size]) and the length of the sequences
    '''
    #extract sequence lengths
    s1_lengths = np.array([len(x) for x in batch])
    #combine hypotheses and premises to pad consistently
    padded_sequences = nn.utils.rnn.pad_sequence(batch)

    return (padded_sequences, s1_lengths)


def load_model(model, nli_path, encoder_path):
    '''
    function that loads and returns a trained NLI and encoder model.
    :param model: the name (str) of the model to be evaluated. this can be 'base', 'lstm', 'bilstm' or 'bilstmpool'
    :param nli_path: the path (str) to the nli model
    :param encoder_path: the path (str) to the encoder. This is None if the selected model is 'base'

    :returns nli: the nli model
    :returns encoder: the encoder model (lstm/bilstm/bilstmpool)
    '''
    if model == 'base':
        encoder = None
        nli_dim = config.vector_length
        
    else:
        if model == 'lstm':
            encoder = models.LSTM(device)
            nli_dim = config.lstm_hidden_dim
        elif model == 'bilstm':
            print("load bilstm encoder")
            encoder = models.BiLSTM(device)
            nli_dim = 2 * config.lstm_hidden_dim
        elif model == 'bilstmpool':
            print("load bilstm maxpool encoder")
            encoder = models.BiLSTMpool(device)
            nli_dim = 2 * config.lstm_hidden_dim
        encoder_checkpoint = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        encoder.eval()
        
    nli = models.NLI(device, nli_dim, encoder)
    nli_checkpoint = torch.load(nli_path, map_location=device)
    nli.load_state_dict(nli_checkpoint['model_state_dict'])
    nli.eval()

    return nli, encoder

def predict(model, embedding_model, nli, s1, s2, batch_size=1):
    '''
    Function that returns the prediction of a given nli model for a (set of/)single sentence pair(/s)
    :param model: the name (str) of the model to be evaluated. this can be 'base', 'lstm', 'bilstm' or 'bilstmpool'
    :param embedding_model: a dict containing the embeddings for in-vocabulary tokens
    :param nli: loaded and trained nli model
    :param s1: list of premises (strings)
    :param s2: list of hypotheses (string)
    :param batch_size: the number of premise - hypothesis pairs (int)

    :returns batch_pred: predictions
    '''
    #set model in eval mode
    nli.eval()

    #for the base model, we take the mean of the embeddings
    if model == 'base':
        take_mean = True
    else:
        take_mean = False
    
    #preprocess and batch sentences
    s1 = preprocess_sentence_data([s1])
    s2 = preprocess_sentence_data([s2])

    s1 = get_batch(embedding_model, s1, device, take_mean)
    s2 = get_batch(embedding_model, s2, device, take_mean)

    with torch.no_grad():

        #base model does not use an encoder
        if model != 'base':
            #pad sentences and make predictions
            (u, u_lengths), (v, v_lengths) = pad(s1,s2, batch_size)
            batch_output = nli(u, v, u_lengths, v_lengths)
        else:
            #make predictions
            u = torch.stack(s1)
            v = torch.stack(s2)
            batch_output = nli(u, v)

        #return the prediction
        batch_pred = batch_output.data.max(1)[1][0].item()
        return batch_pred

def encode(model, embedding_model, sents, device, encoder):
    '''
    preprocess and encode a list of sentences (string)
    the embedding model and the encoder should already be loaded
    '''
    #for the base model, we take the mean of the embeddings
    if model == 'base':
        take_mean = True
    else:
        take_mean = False
    #preprocess and batch sentences
    sents = preprocess_sentence_data(sents)
    embeddings = get_batch(embedding_model, sents, device, take_mean)

    if model != 'base':
        padded_embeddings, embedding_lengths = pad_singlebatch(embeddings)
        embeddings = encoder(padded_embeddings, embedding_lengths)
    else:
        embeddings = torch.stack(embeddings)
    return embeddings