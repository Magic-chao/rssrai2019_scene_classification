# -*- coding: utf-8 -*-
# @Author   : Magic
# @Time     : 2019/7/4 12:02
# @File     : test.py

import torch
import argparse, os
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data import get_test_data, get_val_data

parser = argparse.ArgumentParser(description='test scene data...')
parser.add_argument('--gpus', default='0', type=str, help='ordinates of gpus to use, can be "0,1,2,3" ')
parser.add_argument('--batch-size', default=1, type=int, help='batch size')
parser.add_argument('--num-workers', default=4, type=int, help='the num of threads to load data')
parser.add_argument('--resume', type=str, default='senet_model_29.pth')

args = parser.parse_args()
os.environ['CUDA_VISBLE_DEVICES'] = args.gpus

# Load Datasets
val_data = get_val_data()
test_data = get_test_data()

val_loader = DataLoader(val_data, num_workers=args.num_workers, batch_size=128, shuffle=False)
test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=1, shuffle=False)

# Build Model
model = models.resnext50_32x4d()
in_feature = model.fc.in_features
model.fc = torch.nn.Linear(in_feature, 45)
model_dict = torch.load(args.resume)['model']
pred_dict = {}
for k,v in model_dict.items():
    pred_dict[k.replace('module.','')] = v

#model_dict.update(pred_dict)
model.load_state_dict(pred_dict)
model.to('cuda:0')
model.eval()
res = []


def val():
    with torch.no_grad():
        rightN = 0
        for idx, batch in enumerate(val_loader, 1):
            img, label = Variable(batch[0]), Variable(torch.from_numpy(np.array(batch[1])).long())
            #print(img.shape, label.shape)
            if torch.cuda.is_available():
                img = img.to('cuda:0')
                label = label.to('cuda:0')
            pred = model(img)

            # cal acc
            pred = np.argmax(pred.data.cpu().numpy(), axis=1)
            gt = label.squeeze().cpu().numpy()
            rightN += (pred == gt).sum()
        print('==> Complete. Acc: {:.4f} '.format(rightN / len(val_loader.dataset)))


def eval():
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            #print(batch[0], batch[1])
            pred = model(batch[2])
            pred_label = pred.argmax(dim=1)
            result = '{:05d}.jpg {}'.format(idx+1, int(pred_label.item()) + 1)
            print(result + ' is done!')
            
            #print('\n')
            res.append(result)

    with open('submit.txt', 'w') as f:
        for line in res:
            f.write(line + '\n')


if __name__ == '__main__':
    val()
    #eval()
    # print('Submit.txt is Finished!')

