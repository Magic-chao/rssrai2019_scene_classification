# -*- coding: utf-8 -*-
# @Author   : Magic
# @Time     : 2019/7/4 12:04
# @File     : preprocess.py

import os
import utils
from config import config_dict

# transfrom cn_name to en_name
def transform_name(path):
    cn_to_en, _ = utils.map_label()
    for ro, di, fi in os.walk(path):
        dirname = os.path.dirname(ro)
        name = ro.split('/')[-1]
        print(dirname, name)
        if  name in cn_to_en:
            os.rename(os.path.join(dirname, name), os.path.join(dirname, cn_to_en[name]))
    print('Rename Success!')

def calc_trainset_mean_std():
    pass

if __name__ == '__main__':
    transform_name(config_dict['data_dir'])