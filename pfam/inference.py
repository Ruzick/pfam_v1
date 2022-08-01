
from utils import FamDataset
from model import ProtCNN
from build_fun import preprocess
from pl_setup import Light
import torch




def infer(args):
    model_path = args.model_path
    max_len =args.max_lenght
    data_path = args.data_path
    map_location = args.map_location
    test = FamDataset(data_path, max_len=max_len)
    prot_cnn =  ProtCNN(test.classes)
    checkpoint =  torch.load(model_path,map_location=torch.device(map_location) )
    lr = checkpoint['lr_schedulers'][0]['base_lrs'][0]
    weight_decay = checkpoint['optimizer_states'][0]['param_groups'][0]['weight_decay']
    milestones = list(checkpoint['lr_schedulers'][0]['milestones'].keys())
    gamma = checkpoint['lr_schedulers'][0]['gamma']

    #load hyper parameters
    model = Light(prot_cnn,  milestones, gamma, lr, weight_decay)
    model.load_state_dict(checkpoint["state_dict"] )
    model.eval()

    #get input
    user_input = input("Enter the protein sequence you would like to infer the family of:  ")
    one_coded = preprocess(test.word2id, "".join(user_input.upper().split()), max_len)

    #infer
    label = model(one_coded.unsqueeze(0)).argmax().item()
    #map
    label2fam = {y: x for x, y in test.fam2label.items()}
    print(f"This sequence  belongs to the {label2fam[label]} protein family.")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--model_path', type=str , default='checkpoints/best.ckpt', help = '# path of pretrained model')
    parser.add_argument('-d', '--data_path', type=str , default='random_split/train', help = '# path of pretrained model')
    parser.add_argument('-ml', '--max_lenght', type=int , default=120, help = 'max length of sequence')
    parser.add_argument ('-mpl', '--map_location', type=str , default='cpu', help = "'cpu' or 'gpu' ")

    args = parser.parse_args()

   
    infer(args)
