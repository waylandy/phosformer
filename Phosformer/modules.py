import sys
import time
import warnings
from itertools import groupby

import numpy as np
from scipy.special import softmax

import torch

def batch_job(kinases, peptides, model=None, tokenizer=None, input_order='SK', output_hidden_states=False, output_attentions=False, batch_size=20, device='cpu', threads=1, verbose=False):
    """
    Parameters
    ----------
    kinases : list of str
        protein kinase domain sequences
    peptides : list of str
        11-mer peptide sequences
    model :  PreTrainedModel
        The pre-loaded Phosformer model.
    tokenizer : Tokenizer
        The pre-loaded Phosformer tokenizer.
    input_order : str, default='SK'
        The order to provide the sequence input. "S" stands for substrate peptide
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
    kinase_buffer, peptide_buffer = [], []
    for n, (kinase, peptide) in enumerate(zip(kinases, peptides)):
        kinase_buffer += [kinase]
        peptide_buffer += [peptide]
        if (n == total - 1) or ((n + 1) % batch_size == 0):
            
            ############
            if input_order == 'SK':
                ids = tokenizer(format_many(peptide_buffer), format_many(kinase_buffer), truncation=False, padding=True)
                
            if input_order == 'KS':
                ids = tokenizer(format_many(kinase_buffer), format_many(peptide_buffer), truncation=False, padding=True)
            
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
            outputs = zip(kinase_buffer, peptide_buffer, pred, tokens, embedding, attention)
            for _kinase, _peptide, _pred, _tokens, _embedding, _attention in outputs:
                output = {
                    'kinase'    : _kinase,
                    'peptide'   : _peptide,
                    'pred'      : _pred,
                    'tokens'    : _tokens,
                }

                if output_hidden_states:
                    output['embedding'] =  _embedding
                    
                if output_attentions:
                    output['attention'] =  _attention

                if input_order == 'SK':
                    output['token_labels'] = np.array(((len(_peptide) + 2)*['S']) + ((len(_kinase) + 2)*['K']))
                    
                if input_order == 'KS':
                    output['token_labels'] = np.array(((len(_kinase) + 2)*['K']) + ((len(_peptide) + 2)*['S']))

                yield output
            
            ##########
            kinase_buffer = []
            peptide_buffer = []
            
            ##########
            elapsed = time.time() - start
            if verbose:
                sys.stderr.write(f'Progress: {n+1}, Elapsed: {elapsed:.2f} ({(1+n)/elapsed:.2f} / s)\r')
    
    if verbose:
        sys.stderr.write('\n')

def predict_one(kinase, peptide, **kwargs):
    """
    Parameters
    ----------
    kinase : str
        a single protein kinase domain sequence
    peptide : str
        a single 11-mer peptide sequence
    **kwargs : `batch_job` parameters

    Returns
    -------
    output : float
        prediction value
    """
    kinase_allowed = {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'}
    peptide_allowed = {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-'}
    phosphosite_allowed = {'S','T','Y'}

    if type(kinase) != str:
        raise Exception('Kinase inputs must be of type string.')
    if not all(i in kinase_allowed for i in kinase):
        raise Exception('Illegal character in kinase input. Please only use uppercase amino acids.')
    
    if type(peptide) != str:
        raise Exception('Peptide inputs must be of type string.')
    if not all(i in peptide_allowed for i in peptide):
        raise Exception('Illegal character in peptide input. Please only use uppercase amino acids and dash.')
    if len(peptide) != 11:
        raise Exception('Peptide input must be exactly 11 characters long.')
    if peptide[5] not in phosphosite_allowed:
        raise Exception('The middle character of the peptide must be a phosphorylated amino acid (S, T, or Y).')
    
    return next(batch_job([kinase], [peptide], **kwargs))['pred']

def predict_many(kinases, peptides, **kwargs):
    """
    Parameters
    ----------
    kinase : list of str
        a list of protein kinase domain sequences
    peptide : list of str
        a list of 11-mer peptide sequences
    **kwargs : `batch_job` parameters

    Returns
    -------
    output : np.ndarray
        prediction values
    """
    kinase_allowed = {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'}
    peptide_allowed = {'A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','-'}
    phosphosite_allowed = {'S','T','Y'}
    
    for kinase in kinases:
        if type(kinase) != str:
            raise Exception(f'Kinase inputs must be of type string.\nOffending kinase input: "{kinase}"')
        if not all(i in kinase_allowed for i in kinase):
            raise Exception(f'Illegal character in kinase input. Please only use uppercase amino acids.\nOffending kinase input: "{kinase}"')

    for peptide in peptides:
        if type(peptide) != str:
            raise Exception(f'Peptide inputs must be of type string.\nOffending peptide input: "{peptide}"')
        if not all(i in peptide_allowed for i in peptide):
            raise Exception(f'Illegal character in peptide input. Please only use uppercase amino acids and dash.\nOffending peptide input: "{peptide}"')
        if len(peptide) != 11:
            raise Exception(f'Peptide input must be exactly 11 characters long.\nOffending peptide input: "{peptide}"')
        if peptide[5] not in phosphosite_allowed:
            raise Exception(f'The middle character of the peptide must be a phosphorylated amino acid (S, T, or Y).\nOffending peptide input: "{peptide}"')
    
    return np.array([i['pred'] for i in batch_job(kinases, peptides, **kwargs)])

def get_potential_phosphosites(substrate_sequence, size=11):
    """
    Parameters
    ----------
    substrate_sequence : str
        a single protein sequence
    size : int, default=11
        length of the output peptides
    
    Returns
    -------
    peptides : list of tuple 
        each tuple is (residue_number, peptide_sequence)
    """
    if size % 2 != 1:
        raise Exception('Size must be an odd number.')
    
    flanking = int((size - 1) / 2)
    padded_substrate = flanking * '-' + substrate_sequence + flanking * '-'
    
    peptide_generator = ((index + 1, padded_substrate[index:index+size]) for index in range(len(substrate_sequence)))
    peptides = list((resnum, peptide) for resnum, peptide in peptide_generator if peptide[flanking] in 'STY')
    
    return peptides

def read_fasta(file):
    """
    Parameters
    ----------
    file : str
        the fasta file
    
    Yields
    ------
    output : tuple
        a single entry in the fasta file (header, sequence)
    """
    is_header = lambda x: x.startswith('>')
    compress  = lambda x: ''.join(_.strip() for _ in x)
    reader    = iter(groupby(open(file), is_header))
    reader    = iter(groupby(open(file), is_header)) if next(reader)[0] else reader
    for key, group in reader:
        if key:
            for header in group:
                header = header[1:].strip()
        else:
            sequence = compress(group)
            if sequence != '':
                yield header, sequence