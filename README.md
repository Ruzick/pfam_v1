# pfam_v1

This app will help you identify the protein family from a sequence of aminoacids given. 

Instructions:

## For Docker:
obtain the image uploaded into dockerhub via commandline:  
```
docker pull miladock/pfam_v1:pfam_v1
```  
run the inference.py from the  image locally in interactive mode by:
```
docker run -i miladock/pfam_v1:pfam_v1
```
## For Local:
Use the following folder structure
``` diff
pfam_v1
├── pfam
│   ├── requirements.txt
│   ├── __init__.py
│   ├── build_fun.py
│   ├── inference.py
│   ├── model.py
│   ├── pl_setup.py
│   ├── plot_fun.py
│   ├── train_pl.py
│   └── utils.py
│   └── source
├── images
│  
├── checkpoints 
├── random_split
│   ├── train
│   ├── dev
│   └── test
├── .gitignore
├── README.md
└── Dockerfile

```
Environment requirements:
```
python==3.7.4
matplotlib==3.4.1
numpy==1.18.5
pandas==1.2.3
pytorch-lightning==1.5.3
seaborn==0.11.1
tensorboard==2.2.2
torch==1.8.1
torchmetrics==0.6.0
```

After downloading the dataset :
https://www.kaggle.com/googleai/pfam-seed-random-split

For inference on a pretrained model run:
```diff
python ./pfam/inference.py
```
with the following available arguments:
```diff
--data_path [str default: 'random_split/train'] 
--model_path [str default: 'checkpoints/best.ckpt'] *this is the path of your pre-trained model checkpoint
--max_lenght [int default: 120] defines the max length of the aminoacid sequence 
--map_location [str default: 'cpu'], use 'gpu' if available 
```

To explore a dataset run:
```
python ./pfam/utils.py
```
available arguments:
```diff
--data_path [str default: 'random_split/train'] 
--max_lenght [int default: 120] defines the max length of the aminoacid sequence 
```
To train a model:
```
python ./pfam/train_pl.py
```
arguments:
```diff
--train_data_path [str default: 'random_split/train'] 
--val_data_path [str default: 'random_split/dev'] 
--max_len [int default: 120] an int, defines the max length of the aminoacid sequence 
--num_gpus [int default: torch.cuda.device_count()]
--batch_size [int default:250]
--num_epochs [int default:5]
--weight_decay [float default: 1e-2] weight_decay parameter for SGD
--lr [float default: 1e-2] learning rate parameter for SGD
--milestones [nargs+(list of ints separate entries by space) default:[5,8,10,12,14,16,18,20]] list of epoch indices for MultiStepLR scheduler
--gamma [float default:0.8] optimization parameter for Scheduler, decrease lr
```

```diff
+ notes on the loss function
```
Some aminoacids with very low frequency in families were dropped entirely from the dataset in order to make the data balanced and use out of the shelf cross-entropy loss. One can instead keep all the amino acids and use a balanced cross-entropy by defining the class weights or a focal loss.
