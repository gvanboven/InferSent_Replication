## Code

This folder contains the code I used to train and evaluate the NLI models. 

* `config.py` contains configuration parameters
* `error_analysis/ipynb` contains a demo of the models as well as an error analysis
* `eval_models.py` can be run to evaluate all models and store the evaluation results in two output json files
* `eval.py` evaluates a single model on the NLI task
* `models.py` contains the code for the NLI model and the various LSTM encoders
* `mutils.py` contains util functions
* `senteval.py` can be used to evaluate a model on the SentEval tasks
* `store_word_dict` stores the tokens in the training data to a json file (`train_word_dict.json`), that can be used in order to only loaded those token embeddings during evaluation
* `train.py` can be used to train a model