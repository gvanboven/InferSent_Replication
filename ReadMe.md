# InferSent Replication
In this folder I do a partial replication of the InferSent embeddings by Conneau et al. (2017). I create create sentence embeddings through training on a Natural Language Inference Task (the [SNLI dataset](https://nlp.stanford.edu/projects/snli/) (Bowman et al., 2015)). Specifically I train and evaluate four models:
* `base` : a baseline model which takes the mean of the [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) (Pennington et al., 2014) of the tokens in the sentence as the sentence representation.
* `lstm` takes a unidirectional LSTM as its sentence encoder.
* `bilstm` takes a bidirectional LSTM as its sentence encoder.
* `bilstmpool` has a bidirectional LSTM that applies max pooling over the hidden states of the tokens. 
After extracting the sentence embeddings, all models concatenate the embedding of the hypothesis and the premise following Conneau et al., process the embedding through a small MLP and finally make the prediction in a 3-way softmax (entailment, neutral or contradiciton).

The models are evaluated both on the NLI task and on a variety of SentEval tasks. I replicate the finding by Conneau et al. that the best results are obtained for the `bilstmpool` model. 

This project was carried out for the course Advanced Topics in Computational Semantics at the University of Amsterdam, taught by Ekaterina Shutova, Alina Leidinger and Rochelle Choenni.

The checkpoints of the models I trained and the tensorboard files can be found [here](https://drive.google.com/drive/folders/18EWKTYv4CsF8mxgE7K4Ym6zHtqR6w6fF?usp=sharing).

## Dependencies
The code is written in python. The dependencies are:
* Python 3 (recent version)
* Pytorch (recent version)
* [Senteval](https://github.com/facebookresearch/SentEval) 
* Datasets 2.0
* NLTK tokenize
* Numpy
* Sys 
* Json

## Training and evaluation

## Repo structure
For more info on the code, see the ReadMe in the code folder

# Demo 

# References