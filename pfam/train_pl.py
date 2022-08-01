from model import ProtCNN
from utils import   FamDataset, load_valdata, load_data
from pl_setup import Light

from os import path
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def train(args):

    #hyperparameters
    num_gpus = args.num_gpus
    batch_size =args.batch_size
    num_epochs = args.num_epochs
    max_len = args.max_len
    lr = args.lr
    weight_decay = args.weight_decay
    milestones = args.milestones
    gamma = args.gamma
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path

    #model setup
    famdata  = FamDataset(train_data_path, max_len=max_len)
    classes = famdata.classes
    model = ProtCNN(classes)
    light = Light(model,  milestones, gamma, lr, weight_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=1, 
        monitor="train_loss",
        mode="min",
        filename="best")
    trainer = pl.Trainer(gpus=num_gpus, max_epochs=num_epochs,callbacks=[checkpoint_callback])

    #load data
    train_data =load_data(train_data_path, max_len=max_len, batch_size=batch_size)
    val_data = load_valdata(val_data_path , max_len=max_len, batch_size=batch_size)

    #Train
    pl.seed_everything(0)
    trainer.fit(light,train_data,val_data)
   




if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-td', '--train_data_path', type=str , default='random_split/train', help = '# path of trainng data')
    parser.add_argument('-vd', '--val_data_path', type=str , default='random_split/dev', help = '# path of trainng data')
    parser.add_argument('-g', '--num_gpus', type=int , default=torch.cuda.device_count(), help = '# of gpus')
    parser.add_argument('-b', '--batch_size', type=int, default=250, help = 'batch size')
    parser.add_argument('-e', '--num_epochs', type=int , default=5, help = '# of epochs')
    parser.add_argument('-m', '--max_len', type=int , default=120, help = 'max len of sequence')
    parser.add_argument('-p', '--lr', type=float , default=1e-2, help = 'optimization parameter for lr sgd')
    parser.add_argument('-wd', '--weight_decay', type=float , default=1e-2, help = 'optimization parameter for weight decay sgd')
    parser.add_argument('-ms', '--milestones', nargs='+', type=int, default=[5,8,10,12,14,16,18,20], help = 'list of epoch indices for MultiStepLR scheduler, separate by space')
    parser.add_argument('-ga', '--gamma', type=float , default=0.8, help = 'optimization parameter for Scheduler, decrease percentage lr')
    args = parser.parse_args()
    train(args)