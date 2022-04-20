import numpy as np

from datasets import load_dataset
import config
import models
import mutils
import datetime

import sys
import torch
from torch import nn
from torch import optim

#load tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#load device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device", flush=True)

def eval(embedding_model, nli, U_dev, V_dev, Y_dev, model, take_mean, epoch):
    '''
    Evaluate on dev data
    '''
    correct = 0.
    nli.eval()
    for i in range(0, len(U_dev), config.batch_size):
        #batch inputs
        s1 = mutils.get_batch(embedding_model, U_dev[i:i + config.batch_size], device, take_mean)
        if len(s1) != config.batch_size:
            continue
        s2 = mutils.get_batch(embedding_model, V_dev[i:i + config.batch_size], device, take_mean)
        y_true = torch.tensor(Y_dev[i:i + config.batch_size], dtype=torch.long, device=device)
        
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
    eval_acc = round(100 * correct / len(U_dev),2)
    print(f"Epoch {epoch} Validation accurary : {eval_acc}")
    return eval_acc

def main(argv=None):
    '''
    Main function that trains the  model and stores the checkpoints for this model
    :param model: the name (str) of the model to train. this can be 'base', 'lstm', 'bilstm' or 'bilstmpool'
    :param nli_path: the path (str) where to store  the nli model checkpoint
    :param encoder_path: the path (str) where to store the encoder checkpoint. 
                        This should be None if the selected model is 'base'
    '''
    if argv is None:
        argv = sys.argv
    
    model = argv[1]
    nli_path = argv[2]
    encoder_path = argv[3]

    #set up tensorboard
    log_dir = "../runs/" + model + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)

    print(f"start training {model} model")

     #for the base model, we take the mean of the embeddings and do not use an encoder
    if model == 'base':
        take_mean = True
        encoder = None
        nli_dim = config.vector_length
    else:
        #for the other models we do use an encoder
        take_mean = False
        if model == 'lstm':
            print("load lstm encoder")
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

    nli = models.NLI(device, nli_dim, encoder)

    #load data
    dataset = load_dataset(config.dataset)
    
    print('start loading in data', flush=True)
    U_train, V_train, Y_train = mutils.get_UVY(dataset['train'][:100])
    U_dev, V_dev, Y_dev = mutils.get_UVY(dataset['validation'][:100])

    print('datasets loaded, start shuffling', flush=True)
    #shuffle train data
    np.random.seed(config.seed)
    shuffle_permutation = np.random.permutation(len(U_train))

    U_train = np.array(U_train)[shuffle_permutation]
    V_train = np.array(V_train)[shuffle_permutation]
    Y_train = np.array(Y_train)[shuffle_permutation]

    #shuffle dev data
    np.random.seed(config.seed)
    shuffle_permutation = np.random.permutation(len(U_dev))

    U_dev = np.array(U_dev)[shuffle_permutation]
    V_dev = np.array(V_dev)[shuffle_permutation]
    Y_dev = np.array(Y_dev)[shuffle_permutation]

    print('preprocess sentences', flush=True)
    U_train = mutils.preprocess_sentence_data(U_train)
    V_train = mutils.preprocess_sentence_data(V_train)
    U_dev = mutils.preprocess_sentence_data(U_dev)
    V_dev = mutils.preprocess_sentence_data(V_dev)

    print('load word dict', flush=True)
    train_sents = U_train + V_train
    word_dict = mutils.load_word_dict(train_sents)

    print('load embedding model', flush=True)
    embedding_model = mutils.load_embeddings(word_dict)

    # initialize loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(nli.parameters(), lr=config.lr)

    best_acc = 0

    print('start training', flush=True)
    print(f"Train for {config.num_epochs} epochs; {len(U_train) / config.batch_size} batches", flush =True)

    for epoch in range(config.num_epochs):
        nli.train()

  
        if epoch > 1 : #learning rate decay
            opt.param_groups[0]['lr'] = config.lr_decay * opt.param_groups[0]['lr']

        running_loss = 0.0
        correct = 0.

        #go through all batches
        for batch_number, i in enumerate(range(0, len(U_train), config.batch_size)):
            #batching
            s1 = mutils.get_batch(embedding_model, U_train[i:i + config.batch_size], device, take_mean)
            if len(s1) != config.batch_size:
                continue
            s2 = mutils.get_batch(embedding_model, V_train[i:i + config.batch_size], device, take_mean)
            y_true = torch.tensor(Y_train[i:i + config.batch_size], dtype=torch.long, device=device)

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
            
            #compute loss
            batch_loss = loss_fn(batch_output, y_true)
            #gradient computation
            batch_loss.backward() 
            #optimize
            opt.step()
            #reset gradients
            opt.zero_grad()
            
            #print loss every 2000 batches
            running_loss += batch_loss.item()
            if batch_number % 2000 == 199:
                print(f"[{epoch + 1}, {batch_number + 1:5d}] loss: {running_loss / 100:.3f}", flush=True)
                running_loss = 0.0

        #compute train accuracy of epoch
        train_acc = round(100 * correct / len(U_train),2)
        print(f"Epoch {epoch} Train accurary : {train_acc}", flush=True)
        #compute dev accuracy of epoch
        dev_acc = eval(embedding_model, nli, U_dev, V_dev, Y_dev, model, take_mean, epoch)
        #write accuracies to tensorboard
        writer.add_scalars(f'train/dev accuracy', {
                                            'train_acc': train_acc,
                                            'dev_acc': dev_acc,
                                        }, epoch)
        #if this is the best current model, store it
        if dev_acc > best_acc:
            best_acc = dev_acc

            print(f"Save model at epoch {epoch}")
            #save model
            torch.save({'epoch': epoch,
                        'model_state_dict' : nli.state_dict(),
                        'optimizer_state_dict' : opt.state_dict(),
                        'train_accuracy': train_acc,
                        'dev_accuracy': dev_acc}, nli_path)
            if model != 'base' :
                torch.save({'epoch': epoch,
                        'model_state_dict' : encoder.state_dict(),
                        'optimizer_state_dict' : opt.state_dict(),
                        'train_accuracy': train_acc,
                        'dev_accuracy': dev_acc}, encoder_path)
        #if this model is not the best model, devide lr by 5
        else:
            print(f"reduce lr by /5 at epoch {epoch}", flush = True)
            opt.param_groups[0]['lr'] = opt.param_groups[0]['lr'] / 5

        #stop if lr is smaller than minimum lr
        if opt.param_groups[0]['lr'] < config.min_lr : 
            break

if __name__ == '__main__':
    main()