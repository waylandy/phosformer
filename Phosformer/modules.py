import sys
import time
import warnings

import numpy as np
from scipy.special import softmax

import torch

def batch_job(kinases, substrates, model=None, tokenizer=None, input_order='SK', output_hidden_states=False, output_attentions=False, batch_size=20, device='cpu', threads=1, verbose=False):
    """
    Parameters
    ----------
    kinases : list of str
        protein kinase domain sequences
    substrates : list of str
        11-mer peptide sequences
    model :  PreTrainedModel
        The pre-loaded Phosformer model.
    tokenizer : Tokenizer
        The pre-loaded Phosformer tokenizer.
    input_order : str, default='SK'
        The order to provide the sequence input. "S" stands for substrate
        and "K" stands for kinase. The default, "SK" makes it such that the
        substrate comes first, then the kinase. Do not change this parameter.
    output_hidden_states : bool, default=False
        If true, includes the final layer embedding vector in the output.
    output_attentions : bool, default=False
        If true, includes the final layer attention matrix in the output.
    batch_size : int, default=20
        Batch size for running predictions.
    device : str, default='cpu'
        Torch device for running predictions. Options may include "cpu", 
        "cuda", "cuda:0", etc.
    threads : int, default=1
        Torch threads for running predictions.
    verbose : bool, default=False
        If true, reports progress in stderr.
    
    Yields
    ------
    output : dict
        Contains prediction results and other additional requested outputs.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    if not callable(model) or not callable(tokenizer):
        raise Exception('Please provide the pre-loaded Phosformer "model" and "tokenizer".')

    torch.set_num_threads = threads
    torch.cuda.empty_cache()
    model.eval()

    ############
    token_dic   = {'.':'<mask>', '-':'<pad>'}
    format_one  = lambda x: ''.join(token_dic[i] if i in token_dic else i.upper() for i in x if not i.isspace())
    format_many = lambda x: list(map(format_one, x))

    ############
    start = time.time()
    
    total = len(kinases)
    kinase_buffer, substrate_buffer = [], []
    for n, (kinase, substrate) in enumerate(zip(kinases, substrates)):
        kinase_buffer += [kinase]
        substrate_buffer += [substrate]
        if (n == total - 1) or ((n + 1) % batch_size == 0):
            
            ############
            if input_order == 'SK':
                ids = tokenizer(format_many(substrate_buffer), format_many(kinase_buffer), truncation=False, padding=True)
                
            if input_order == 'KS':
                ids = tokenizer(format_many(kinase_buffer), format_many(substrate_buffer), truncation=False, padding=True)
            
            input_ids      = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask'], dtype=torch.bool).to(device)
            
            ############
            model = model.to(device)
            with torch.no_grad():
                result    = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states, output_attentions=output_attentions)

            if output_hidden_states:
                embedding = [i[j].cpu().numpy() for i, j in zip(result['hidden_states'][-1], attention_mask)]
            else:
                embedding = batch_size * [None]

            if output_attentions:
                attention = [i[:,j][:,:,j].cpu().numpy() for i, j in zip(result['attentions'][-1], attention_mask)]
            else:
                attention = batch_size * [None]
            
            tokens = [i[j].cpu().numpy() for i, j in zip(input_ids, attention_mask)]
            pred = softmax(result['logits'].cpu(), axis=1)[:,1].numpy()
            
            ############
            outputs = zip(kinase_buffer, substrate_buffer, pred, tokens, embedding, attention)
            for _kinase, _substrate, _pred, _tokens, _embedding, _attention in outputs:
                output = {
                    'kinase'    : _kinase,
                    'substrate' : _substrate,
                    'pred'      : _pred,
                    'tokens'    : _tokens,
                }

                if output_hidden_states:
                    output['embedding'] =  _embedding
                    
                if output_attentions:
                    output['attention'] =  _attention

                if input_order == 'SK':
                    output['token_labels'] = np.array(((len(_substrate) + 2)*['S']) + ((len(_kinase) + 2)*['K']))
                    
                if input_order == 'KS':
                    output['token_labels'] = np.array(((len(_kinase) + 2)*['K']) + ((len(_substrate) + 2)*['S']))

                yield output
            
            ##########
            kinase_buffer = []
            substrate_buffer = []
            
            ##########
            elapsed = time.time() - start
            if verbose:
                sys.stderr.write(f'Progress: {n+1}, Elapsed: {elapsed:.2f} ({(1+n)/elapsed:.2f} / s)\r')
    
    if verbose:
        sys.stderr.write('\n')

def predict_one(kinase, substrate, **kwargs):
    """
    Parameters
    ----------
    kinase : str
        a single protein kinase domain sequence
    substrate : str
        a single 11-mer peptide sequence
    **kwargs : `batch_job` parameters

    Returns
    -------
    output : float
        prediction value
    """
    kinase_allowed = {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'}
    substrate_allowed = {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-'}
    phosphosite_allowed = {'S','T','Y'}

    assert type(kinase) == str
    assert all(i in kinase_allowed for i in kinase)

    assert type(substrate) == str
    assert all(i in substrate_allowed for i in substrate)
    assert len(substrate) == 11
    assert substrate[5] in phosphosite_allowed
    
    return next(batch_job([kinase], [substrate], **kwargs))['pred']

def predict_many(kinases, substrates, **kwargs):
    """
    Parameters
    ----------
    kinase : list of str
        a list of protein kinase domain sequences
    substrate : list of str
        a list of 11-mer peptide sequences
    **kwargs : `batch_job` parameters

    Returns
    -------
    output : np.ndarray
        prediction values
    """
    kinase_allowed = {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'}
    substrate_allowed = {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-'}
    phosphosite_allowed = {'S','T','Y'}
    
    for kinase in kinases:
        assert type(kinase) == str
        assert all(i in kinase_allowed for i in kinase)
        
    for substrate in substrates:
        assert type(substrate) == str
        assert all(i in substrate_allowed for i in substrate)
        assert len(substrate) == 11
        assert substrate[5] in phosphosite_allowed
    
    return np.array([i['pred'] for i in batch_job(kinases, substrates, **kwargs)])

