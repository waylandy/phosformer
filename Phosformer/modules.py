import sys
import time
import warnings

import numpy as np
from scipy.special import softmax

import torch

def batch_job(kinases, substrates, model=None, tokenizer=None, input_order='SK', output_hidden_states=False, output_attentions=False, batch_size=20, device='cpu', threads=1, verbose=False):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    if not callable(model) or not callable(tokenizer):
        raise Exception('"model" and "tokenizer" must be provided as keyword arguments.')

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

