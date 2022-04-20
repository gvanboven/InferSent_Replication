import torch
from torch import nn
import config
import numpy as np

class NLI(nn.Module):
    
    def __init__(self, device :str, vector_input_dim:int, encoder =None):
        super(NLI, self).__init__()
        self.device = device
        
        self.input_dimension = 4*vector_input_dim
        self.hidden_dimension = config.nli_hidden_dim
        self.n_output_classes = 3
        self.encoder = encoder
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dimension, self.hidden_dimension, device=self.device),
            nn.Linear(self.hidden_dimension, self.hidden_dimension, device=self.device),
            nn.Linear(self.hidden_dimension, self.n_output_classes, device=self.device)
            ).to(self.device)

    def forward(self, u, v, u_len=None, v_len=None):
        if self.encoder != None:
            u = self.encoder(u, u_len)
            v = self.encoder(v, v_len)
        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output

class LSTM(nn.Module):

    def __init__(self, device :str):
        super(LSTM, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size=config.vector_length,
                            hidden_size=config.lstm_hidden_dim,
                            num_layers=config.lstm_layers,
                            bidirectional=False,
                            dropout=config.dropout,
                            device = self.device)

    def forward(self, batch, batch_len):
        #sort in descending order
        seq_lengths, perm_idx = np.sort(batch_len)[::-1], np.argsort(-batch_len)
        #sort sequences in batch; [seq len x batch size x embedding dim]
        seq_tensor = torch.index_select(batch, 1, torch.tensor(perm_idx, dtype=torch.long, device=self.device))
        #pack 
        seq_packed = nn.utils.rnn.pack_padded_sequence(seq_tensor, seq_lengths.copy())

        #[batch size x hidden dim]
        encoded_batch = self.lstm(seq_packed)[1][0].squeeze(0)

        idx_unsort = np.argsort(perm_idx)
        #unsort sequences
        # [batch size x hidden dim]
        encoded_batch = torch.index_select(encoded_batch, 0, torch.tensor(idx_unsort, dtype=torch.long, device=self.device))
        return encoded_batch

class BiLSTM(nn.Module):

    def __init__(self, device :str):
        super(BiLSTM, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size=config.vector_length,
                            hidden_size= config.lstm_hidden_dim,
                            num_layers=config.lstm_layers,
                            bidirectional=True,
                            dropout=config.dropout,
                            device = self.device)

    def forward(self, batch, batch_len):
        #sort in descending order
        seq_lengths, perm_idx = np.sort(batch_len)[::-1], np.argsort(-batch_len)
        #sort sequences in batch; [seq len x batch size x embedding dim]
        seq_tensor = torch.index_select(batch, 1, torch.tensor(perm_idx, dtype=torch.long, device=self.device))
        #pack 
        seq_packed = nn.utils.rnn.pack_padded_sequence(seq_tensor, seq_lengths.copy())
                
        # [2 (forward + backward last hidden state) x batch size x hidden dim ]
        encoded_batch = self.lstm(seq_packed)[1][0]
        # concat forward and backward hidden state to :[batch size x 2 * hidden dim]
        encoded_batch = torch.cat((encoded_batch[0],encoded_batch[1]), 1)

        #unsort sequences; [batch size x 2* hidden dim]
        idx_unsort = np.argsort(perm_idx)
        encoded_batch = torch.index_select(encoded_batch, 0, torch.tensor(idx_unsort, dtype=torch.long, device=self.device))

        return encoded_batch

class BiLSTMpool(nn.Module):

    def __init__(self, device :str):
        super(BiLSTMpool, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size=config.vector_length,
                            hidden_size= config.lstm_hidden_dim,
                            num_layers=config.lstm_layers,
                            bidirectional=True,
                            dropout=config.dropout,
                            device = self.device)

    def forward(self, batch, batch_len):
        #sort in descending order
        seq_lengths, perm_idx = np.sort(batch_len)[::-1], np.argsort(-batch_len)
        #sort sequences in batch
        seq_tensor = torch.index_select(batch, 1, torch.tensor(perm_idx, dtype=torch.long, device=self.device))
        #pack 
        seq_packed = nn.utils.rnn.pack_padded_sequence(seq_tensor, seq_lengths.copy())
                
        #packed  [seq len x batch size x hidden dim * 2 (forward + backward last hidden state)]
        encoded_batch = self.lstm(seq_packed)[0]

        #unpack [batch size x seq len x 2*hidden dim]
        encoded_batch = nn.utils.rnn.pad_packed_sequence(encoded_batch, batch_first=True)[0]

        idx_unsort = np.argsort(perm_idx)
        encoded_batch = torch.index_select(encoded_batch, 0, torch.tensor(idx_unsort, dtype=torch.long, device=self.device))
        
        # zero padding must be removed for max pooling
        # list of length batch_size, each element is [seqlen x 2*hid]
        batch_output = [sent[:sent_len] for sent, sent_len in zip(encoded_batch, batch_len)]
        #take max in each dimension; list len batch size; [2*hidden dim]
        emb = [torch.max(x, 0)[0] for x in batch_output]
        # [batch size x 2* hidden dim ]
        emb = torch.stack(emb, 0)
        return emb