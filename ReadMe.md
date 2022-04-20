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

All code should be run from the `code` folder. 

The training of the model can be done by running the following command:     
`python train.py [model_name] [nli_path] [encoder_path]`      
Here the `[model_name]` is the name of the selected model: `base`, `lstm`, `bilstm` or `bilstmpool`.    
`[nli_path]` is the path to which the NLI model checkpoint should be stored and `[encoder]` is the path where the encoder should be stored. The `base` model does not train an encoder and in this case the `encoder_path` should be `None`.

## Repo structure
* The folder `code` contains all code used to train and evaluate models. This folder also contains a notebook `error_analysis.ipnb` that contains a demo of the models and an error analysis (see below). 
* The folder `eval_results` contains my evaluation results for all four models, both on the NLI task and on the SentEval tasks.
* `pretrained` should contain the pretrained GloVe model (`glove.840B.300d.txt`), but this was currently taken out. In order for the code to run, this model should be placed in this folder


## Demo 

## References