
import torch 
import numpy as np
import pandas as pd
from collections import Counter


def preprocess(word2id, text, max_len):
    seq = []
    # Encode into IDs
    for word in text[:max_len]:
        seq.append(word2id.get(word, word2id['<unk>']))
    # Pad to maximal length
    if len(seq) < max_len:
        seq += [word2id['<pad>'] for _ in range(max_len - len(seq))]
    # Convert list into tensor
    seq = torch.from_numpy(np.array(seq))
    # One-hot encode    
    one_hot_seq = torch.nn.functional.one_hot(seq.long(), num_classes=len(word2id), ) 
    # Permute channel (one-hot) dim first
    one_hot_seq = one_hot_seq.permute(1,0)
    return one_hot_seq
    
def build_vocab(data): 
    # Build the vocabulary
    voc = set()
    rare_AAs = {'X', 'U', 'B', 'O', 'Z'}
    for sequence in data:
        voc.update(sequence)
    unique_AAs = sorted(voc - rare_AAs)
    # Build the mapping
    word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    return word2id

def build_labels(targets):
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0
    return fam2label

def get_amino_acid_frequencies(data):

    aa_counter = Counter()
    for sequence in data:
        aa_counter.update(sequence)
    return pd.DataFrame({'AA': list(aa_counter.keys()), 'Frequency': list(aa_counter.values())})