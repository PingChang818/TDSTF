import argparse
import torch
import datetime
import json
import yaml
import os
from diff import TDSTF
from dataset import get_dataloader
from exe import train, evaluate

parser = argparse.ArgumentParser(description='')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--modelfolder', type=str, default='')
parser.add_argument('--nsample', type=int, default=100)
args = parser.parse_args()
print(args)
config = yaml.safe_load(open('config/base.yaml', 'r'))
print(json.dumps(config, indent=4))
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
foldername = './save/attention_' + current_time + '/'
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
data_path = 'preprocess/data/dataset.pkl'
var_path = 'preprocess/data/var.pkl'
size = config['diffusion']['size']
train_loader, valid_loader, test_loader = get_dataloader(data_path, var_path, size)
model = TDSTF(config, args.device).to(args.device)

if args.modelfolder == '':
    train(
        model,
        config,
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load('./save/' + args.modelfolder + '/model.pth'))

print('test')
CRPS, MSE = evaluate(1, model, test_loader, nsample=args.nsample, foldername=foldername)
print('CRPS: {}'.format(CRPS))
print('MSE: {}'.format(MSE))
