# Introduction

This repository contains the code and datasets for the manuscript "Phosformer: An interpretable Transformer model for kinase-specific phosphosite prediction".


# Installing dependencies

### Downloading this repository

```
# Download this repository
git clone https://github.com/waylandy/phosformer

cd phosformer
```

### Installing dependencies with `conda`

If you do not have `conda` installed, you can download from the [Anaconda website](https://www.anaconda.com/).

```
conda env create -f environment.yaml
conda activate phosformer
```

If you want to exit the environment:

```
conda deactivate
```

If you want to start over, you can also delete the `conda` environment:

```
conda env remove -n phosformer
```

**Notes on PyTorch:** The Phosformer model uses the PyTorch library which can benefit from GPU acceleration. You can monitor your GPU and check the your CUDA version by running `nvidia-smi -l 1`. PyTorch should always work on CPU, however additional steps may be required to enable CUDA, depending on your GPU model. For more information, see the installation guide on the [PyTorch website](https://pytorch.org/). 


# How to generate predictions

The section describes how to load Phosformer and use it for kinase-specific phosphosite predictions. Example code can be found in under `example.ipynb`. The computational notebook can be viewed using JupyterLab which is included in our environment. You can run it using `jupyter lab`.

### Loading Phosformer

The section describes how to load the model and tokenizer. If this is your first time running Phosformer, the model will automatically be downloaded from Hugging Face Hub and cached for subsequent uses.

```
import Phosformer

# authentication tokens are provided for reviewers
# this will be removed upon acceptance & the model will be made fully public
auth_token = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# load Phosformer model and tokenizer
model = Phosformer.RobertaForSequenceClassification.from_pretrained('waylandy/phosformer', use_auth_token=auth_token)
tokenizer = Phosformer.RobertaTokenizer.from_pretrained('waylandy/phosformer', use_auth_token=auth_token)

# disables dropout for deterministic results
model.eval()
```

### Running predictions on kinase-peptide pairs

Phosformer generates predictions based on **two inputs**:

1. a kinase domain sequence
    - must be a string composed of the 20 amino acid, in uppercase
    - the sequence should **not** contain gap characters
    - the sequence can be any length (although kinase domains are usually ~300 residues long)
    - sequences for all human kinases can be found under `data/human_kinases.csv`
2. a peptide sequence
    - must be a string composed of the 20 amino acid, in uppercase
    - must be exactly 11 residues long
    - the center residue is defined as the potential phosphosite and must be either S, T, or Y
    - if the potential phosphosite occurs within 5 residues of the N or C-terminal, pad the sequence with gap characters

For each pair of inputs, Phosformer generates **one output**:

1. probability of phosphoryllation
    - this is a value in the range (0, 1) where values greater than 0.5 are positive predictions

Unlike previous methods in phosphosite prediction, Phosformer does not utilize kinase or substrate labels as input. Phosformer's understanding of the kinase and substrate is based soley off of primary sequence information. Consequently the Phosformer architecture is capable of representing any kinase-substrate combination — new labels do not need to be added to accommodate the addition of new kinases.

Here is a basic example showing how to run a single prediction.

```
# Example prediction using ERK2 kinase domain (UniProt: P28482)
kinase  = 'YTNLSYIGEGAYGMVCSAYDNVNKVRVAIKKISPFEHQTYCQRTLREIKILLRFRHENIIGINDIIRAPTIEQMKDVYIVQDLMETDLYKLLKTQHLSNDHICYFLYQILRGLKYIHSANVLHRDLKPSNLLLNTTCDLKICDFGLARVADPDHDHTGFLTEYVATRWYRAPEIMLNSKGYTKSIDIWSVGCILAEMLSNRPIFPGKHYLDQLNHILGILGSPSQEDLNCIINLKARNYLLSLPHKNKVPWNRLFPNADSKALDLLDKMLTFNPHKRIEVEQALAHPYL'
peptide = 'LLKLASPELER'

Phosformer.predict_one(kinase, peptide, model=model, tokenizer=tokenizer)
```

Here is the same example, except the kinase domain sequence is retrieved from an included csv file.

```
import pandas as pd

kinases       = pd.read_csv('data/human_kinases.csv')
kinase_domain = kinases[kinases['uniprot']=='P28482']['sequence'].item()
peptide       = 'LLKLASPELER'

Phosformer.predict_one(kinase_domain, peptide, model=model, tokenizer=tokenizer)
```

Here is a basic example showing how to run a single prediction.

```
import pandas as pd

kinases       = pd.read_csv('data/example_input.csv')
kinase_list   = kinases['kinase domain sequence'].values
peptide_list  = kinases['peptide sequence'].values

Phosformer.predict_many(
    kinase_list, peptide_list,
    model=model, tokenizer=tokenizer,
    batch_size=20, # how many samples to load at once, if you're running out of memory, you can set this number lower
    device='cpu',  # either "cpu" or "cuda"
    threads=1      # specify how many threads to use, can help speed up if running on cpu
)
```




