import os
from build_fun import build_labels, preprocess, build_vocab
from torch.utils.data import  DataLoader, Dataset

import pandas as pd
import numpy as np



class FamDataset(Dataset):
    def __init__(self, data_path,max_len):
        self.data, self.label = [], []
        self.max_len = max_len
        self.classes = 0
        self.fam2label = {}
        self.word2id = {}

        data_raw = []
        for file_name in os.listdir(data_path):

            with open(os.path.join(data_path, file_name)) as file:
                data_raw.append(pd.read_csv(file, index_col=None, usecols=["sequence", "family_accession"]))
        all_data = pd.concat(data_raw)  
        self.data, self.label =  all_data["sequence"], all_data["family_accession"]

        #mapping
        self.fam2label = build_labels(self.label)
        self.classes = len(self.fam2label)
        self.word2id = build_vocab(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = preprocess(word2id=self.word2id,  text=self.data.iloc[index], max_len=self.max_len) 
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])

        return  {'sequence': seq, 'target' : label}


        
#Data loaders

def load_data(dataset_path,max_len=120,  num_workers=os.cpu_count(),  pin_memory=True, batch_size=250, **kwargs):
    dataset = FamDataset( dataset_path, max_len,**kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size,pin_memory=pin_memory, shuffle=True)

def load_valdata(dataset_path,max_len=120,  num_workers=os.cpu_count(),  pin_memory=True, batch_size=250, **kwargs):
    dataset = FamDataset( dataset_path, max_len,**kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size,pin_memory=pin_memory, shuffle=True)  

if __name__ == '__main__':
    from .plot_fun import distrib_fam_sizes, distr_sequ_length, distrib_AA_freq
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--data_path', type=str , default='random_split/train', help = '# path of data to analyse')
    parser.add_argument('-ml', '--max_lenght', type=int , default=120, help = 'max length of sequence')
    args = parser.parse_args()
    data_path = args.data_path
    max_len =args.max_lenght
    #load data
    dataset = FamDataset(data_path, max_len)

    # Plot the distribution of family sizes
    distrib_fam_sizes(dataset)

    # Plot the distribution of sequences' lengths
    distr_sequ_length(dataset)

    # Plot the distribution of AA frequencies
    distrib_AA_freq(dataset)