# -*- coding: utf-8 -*-
# @Author   : Magic
# @Time     : 2019/7/4 12:02
# @File     : train.py

import torch, time
import torchvision.models as models
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from Nadam import Nadam
from model import senet52
from data import get_train_data, get_val_data
from utils import save_checkpoint
import utils


def parse_args():
    parser = argparse.ArgumentParser(description='train scene data')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--num-workers', default=4, type=int ,help='the num of threads to load data')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs, Default 500')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--num-classes', default=45, type=int, help='the number of classes')
    parser.add_argument('--gpus', default='0,1', type=str, help='ordinates of gpus to use, can be "0,1,2,3" ')
    parser.add_argument('--seed', default=666, type=int, help='random seed to use, Default=666')

    parser.add_argument('--begin-epoch', default=0, type=int, help='begin epoch')
    parser.add_argument('--lr-factor', default=0.1, type=float, help='the ratio to reduce lr on each step')
    parser.add_argument('--lr-step-epochs', default='20,45,60,80', type=str, help='the epochs to reduce the lr')
    parser.add_argument('--save-model-prefix', default='resnext', type=str, help='model prefix')
    parser.add_argument('--save-model-step', type=int, default=1, help='snapshot step (epoch num)')
    
    parser.add_argument('--net-params', type=str, default=None, help='resume the training')
    parser.add_argument('--log-dir', type=str, default='log', help='the directory of the log')
    parser.add_argument('--log-file', type=str, default='log.txt', help='log file path')

    return parser.parse_args()

args = parse_args()
writer = SummaryWriter(log_dir='logs_board/resnext')
devs = [int(x) for x in args.gpus.split(',')]
lr_step_epochs = [int(x) for x in args.lr_step_epochs.split(',')]
if args.log_dir:
    utils.create_dir(args.log_dir)
    logger = utils.Logger(os.path.join(args.log_dir, args.log_file))

train_data = get_train_data()
train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

model = models.resnext50_32x4d(pretrained=True)
num_fc = model.fc.in_features
model.fc = nn.Linear(num_fc, args.num_classes)

if args.net_params:
    print('=> Loading checkpoint... ')
    resume_model = torch.load(args.net_params)
    model_dict = resume_model['model']
    args.begin_epoch = resume_model['epoch']
    pred_dict = {}
    for k,v in model_dict.items():
        pred_dict[k.replace('module.','')] = v
    model.load_state_dict(pred_dict)
    
if args.begin_epoch:
    for i in lr_step_epochs:
        if args.begin_epoch>=i:
            args.lr = args.lr*0.1
print('Learning rate is ', args.lr)

model = nn.DataParallel(model, device_ids=devs)
model.to('cuda:0')
criterion = nn.CrossEntropyLoss()
optimizer = Nadam(model.parameters(), lr=args.lr)

val_data = get_val_data()
val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

def train(epoch):
    epoch_loss, rightN = 0, 0
    model.train()
    for idx, batch in enumerate(train_loader, 1):
        img, label = Variable(batch[0], requires_grad=True), Variable(torch.from_numpy(np.array(batch[1])).long())
        if torch.cuda.is_available():
            img = img.to('cuda:0')
            label = label.to('cuda:0')

        optimizer.zero_grad()
        t0 = time.time()
        pred = model(img)
        #print(pred.shape, label.shape, label.squeeze().shape)
        loss = criterion(pred, label.squeeze())
        
        #cal acc
        pred = np.argmax(pred.data.cpu().numpy(), axis=1)
        gt = label.squeeze().cpu().numpy()
        rightN += (pred==gt).sum()
        
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        writer.add_scalar('scalar/loss',loss.item(), epoch*len(train_loader)+idx)
        msg = '==> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.'.format(epoch, idx, len(train_loader),\
                                                                                loss.item(), time.time()-t0)
        logger.write(msg)
    msg = '==> Epoch {} Complete. Train Acc: {:.4f} || Avg Loss: {:.4f}'.format(epoch, rightN/len(train_loader.dataset), epoch_loss/len(train_loader))
    logger.write(msg)
    writer.add_scalar('scalar/train_acc', rightN/len(train_loader.dataset), epoch)

def val(epoch):
    model.eval()
    with torch.no_grad():
        count = 0
        for idx, batch in enumerate(val_loader, 1):
            img, label = Variable(batch[0]), Variable(torch.from_numpy(np.array(batch[1])).long())
            if torch.cuda.is_available():
                img = img.to('cuda:0')
                label = label.to('cuda:0')
            pred = model(img)
        
            #cal acc
            pred = np.argmax(pred.data.cpu().numpy(), axis=1)
            gt = label.squeeze().cpu().numpy()
            count += (pred==gt).sum()
        msg = '==> Train{}: Complete. Val Acc: {:.4f} '.format(epoch, count/len(val_loader.dataset))
        logger.write(msg)
        writer.add_scalar('scalar/val_acc', count/len(val_loader.dataset), epoch)

if __name__ == '__main__':
    for epoch in range(args.begin_epoch, args.epochs + 1):
        train(epoch)
        val(epoch)
        if epoch in lr_step_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_factor
            print('Learning rate decay : lr = {}'.format(optimizer.param_groups[0]['lr']))

        if (epoch+1) % args.save_model_step == 0:
            save_checkpoint(model, epoch, args.save_model_prefix)

writer.close()
