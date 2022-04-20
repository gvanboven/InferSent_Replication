In this folder I do a partial replication of the InferSent embeddings by Conneau et al. (2017). I create create sentence embeddings through training a Natural Language Inference Tasl. Specifically I train and evaluate four models:
* `base` : a base model which takes the mean of the GloVe embeddings of the sentence tokens as its representations.
* `lstm` takes a unidirectional LSTM as its sentence encoder.
* `bilstm` takes a bidirectional LSTM as its sentence encoder.
* `bilstmpool` has a bidirectional LSTM that applies max pooling over the hidden states of the tokens. 

The models are evaluated both on the NLI task and on a variety of SentEval tasks. I replicate the finding by Conneau et al. that the best results are obtained for the `bilstmpool` model. 

This project was carried out for the course Advanced Topics in Computational Semantics at the University of Amsterdam, taught by Ekaterina Shutova, Alina Leidinger and Rochelle Choenni.

# Dependencies

# training and evaluation

# repo structure
For more info on the code, see the ReadMe in the code folder

# Demo 