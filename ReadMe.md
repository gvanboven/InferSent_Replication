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

Training of a model can be done by running the following command:     
`python train.py [model_name] [nli_path] [encoder_path]`      
Here the `[model_name]` is the name of the selected model: `base`, `lstm`, `bilstm` or `bilstmpool`.    
`[nli_path]` is the path to which the NLI model checkpoint should be stored and `[encoder_path]` is the path where the encoder should be stored. The `base` model does not train an encoder and in this case the `encoder_path` should be `None`.

A model can be evaluated on the NLI task by running:    
`python eval.py [model_name] [nli_path] [encoder_path]`    
Which takes all the same parameters as training, only now the paths should refer to the checkpoints of a trained model. Again `[encoder_path]`  should be `None` for the `base` model.

To evaluate a model on SentEval run:       
`python senteval_eval.py [model_name] [nli_path] [output_path]`      
Where `[output_path]` should be a json file to which the evaluation results will be stored.

## Repository structure
* The folder `code` contains all code used to train and evaluate models. This folder also contains a notebook `error_analysis.ipnb` that contains a demo of the models and an error analysis (see below). 
* The folder `eval_results` contains my evaluation results for all four models, both on the NLI task and on the SentEval tasks.
* `pretrained` should contain the pretrained GloVe model (`glove.840B.300d.txt`), but this was currently taken out. In order for the code to run, this model should be placed in this folder


## Demo 
The file `code/error_analysis` contains a demonstration on how to make NLI predictions with a trained model for a hypothesis - premise pair. Continuing, this file contains an error analysis for the NLI task, and an analysis of the information that is represented in the sentence embeddings that are formed by the different models.

## References

S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning. A large annotated corpus for learning
natural language inference. _arXiv preprint arXiv:1508.05326_, 2015.

A. Conneau, D. Kiela, H. Schwenk, L. Barrault, and A. Bordes. Supervised learning of univer-
sal sentence representations from natural language inference data. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_, pages 670–680, Copenhagen,
Denmark, September 2017. Association for Computational Linguistics.

J. Pennington, R. Socher, & C. D. Manning. Glove: Global vectors for word representation. In _Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)_ pages 1532-1543, October 2014.