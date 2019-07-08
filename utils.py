# -*- coding: utf-8 -*-
# @Author   : Magic
# @Time     : 2019/7/4 10:49
# @File     : utils.py

import os
import torch
import numpy as np
from config import config_dict
#file_path = 'F:\\ai_competition\\rssrai2019_scene_classification\\ClsName2id.txt'
file_path = config_dict['name_to_id']

def map_label(file_path=file_path):
    chinese_to_english = {}
    label_map = {}
    with open(file_path, encoding='utf-8') as f:
        for line in  f.readlines():
            cn_name, en_name, label = line.strip().split(':')
            chinese_to_english[cn_name] = en_name
            label_map[en_name] = label
    return chinese_to_english, label_map

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('Create dir failed! try again.')
            raise

def cuda(x):
    if torch.cuda.is_available():
        if isinstance(x, (list, tuple)):
            return [_x.cuda() for _x in x]
        else:
            return x.cuda()

def save_checkpoint(model, epoch, prefix):
    output_path = 'checkpoint/' + prefix + '_model_{}.pth'.format(epoch)
    if not os.path.exists('checkpoint/'):
        os.mkdir('checkpoint/')
    state = {'epoch': epoch, 'model':model.state_dict()}
    torch.save(state, output_path)
    print('Checkpoint save to {}'.format(output_path))

class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.log_file = open(output_name, 'w')
        self.info = {}

    def append(self, key, value):
        vals = self.info.setdefault(key, [])
        vals.append(value)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.info.items():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.info = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)




