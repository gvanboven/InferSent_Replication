import eval
import json

"""
This file computes and stores the performance scores for the four models (base, lstm, bilst, bilstmpool) in a dict, 
which is stored in an output json file `eval_results/final_model_results.json`.
The purpose of this file is that the results can easily be printed in the error analysis notebook

The senteval performance scores are extracted from a separate json per model, and the micro/macro averages are computed per model.

The nli performance scores are computed through `eval.py`

Finally, the per-task SentEval performance scores pfor all models are stored in a new json file 'eval_results/final_task_results.json'
"""

eval_results_path = '../eval_results/'
model_path = '../model_checkpoints/'

#dict for averaged results
nli_results = {'base': {}, 
                'lstm' : {},
                'bilstm' : {},
                'bilstmpool' : {}}

#dict for per-task results
sent_eval_scores = {'base' : {},
                   'lstm' : {},
                   'bilstm' : {},
                   'bilstmpool': {}}

def load_senteval_scores(sent_eval_scores, model, modelpath):
    '''
    Store the dev accuracy per task
    '''
    with open(modelpath) as json_file:
        task_scores = json.load(json_file)
        for task, scores in task_scores.items():            
            sent_eval_scores[model][task] = scores['devacc']
        return sent_eval_scores

def get_senteval_scores(json_path):
    '''
    compute the macro and the micro and macro average for the senteval tasks dev accuracies
    '''
    #open SentEval scores
    with open(json_path) as json_file:
        results = json.load(json_file)
    #macro average
    ntasks = len(results.keys())
    scores_sum = sum([int(results[task]['devacc']) for task in results.keys() ])
    macro = round(scores_sum / ntasks,2)
    #micro average
    weighted_sum = sum([int(results[task]['devacc']) * int(results[task]['ndev']) for task in results.keys()])
    nsamples = sum([int(results[task]['ndev']) for task in results.keys()])
    micro = round(weighted_sum / nsamples,2)
    return macro, micro

#compute and store SentEval scores
nli_results['base']['macro'], nli_results['base']['micro']  = get_senteval_scores(eval_results_path + 'base_senteval_results.json')
nli_results['lstm']['macro'], nli_results['lstm']['micro']  = get_senteval_scores(eval_results_path + 'lstm_senteval_results.json')
nli_results['bilstm']['macro'], nli_results['bilstm']['micro']  = get_senteval_scores(eval_results_path + 'bilstm_senteval_results.json')
nli_results['bilstmpool']['macro'], nli_results['bilstmpool']['micro']  = get_senteval_scores(eval_results_path + 'bilstmpool_senteval_results.json')

#compute and score NLI scores
nli_results['base']['dev_acc'], nli_results['base']['test_acc']  = eval.main([None ,'base', model_path + 'base_model_final', None])
nli_results['lstm']['dev_acc'], nli_results['lstm']['test_acc'] = eval.main([None ,'lstm', model_path + 'lstm_nli_model_final', model_path + 'lstm_lstm_model_final'])
nli_results['bilstm']['dev_acc'], nli_results['bilstm']['test_acc'] = eval.main([None ,'bilstm', model_path + 'bilstm_nli_model_final', model_path + 'bilstm_lstm_model_final'])
nli_results['bilstmpool']['dev_acc'], nli_results['bilstmpool']['test_acc'] = eval.main([None ,'bilstmpool', model_path + 'bilstmpool_nli_model_final', model_path + 'bilstmpool_lstm_model_final'])


sent_eval_scores = load_senteval_scores(sent_eval_scores, 'base', eval_results_path + 'base_senteval_results.json')
sent_eval_scores = load_senteval_scores(sent_eval_scores, 'lstm', eval_results_path + 'lstm_senteval_results.json')
sent_eval_scores = load_senteval_scores(sent_eval_scores, 'bilstm', eval_results_path + 'bilstm_senteval_results.json')
sent_eval_scores = load_senteval_scores(sent_eval_scores, 'bilstmpool', eval_results_path + 'bilstmpool_senteval_results.json')

#store results dict to output file
with open(eval_results_path + 'final_model_results.json', 'w') as fp:
    json.dump(nli_results, fp,  indent=4)

with open(eval_results_path + 'final_task_results.json', 'w') as fp:
    json.dump(sent_eval_scores, fp,  indent=4)