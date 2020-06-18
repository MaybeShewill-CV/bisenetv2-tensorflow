#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 下午5:46
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : train_bisenetv2_cityscapes.py
# @IDE: PyCharm
"""
Train bisenetv2 on cityscapes dataset
"""
from trainner.cityscapes import cityscapes_bisenetv2_single_gpu_trainner as single_gpu_trainner, \
    cityscapes_bisenetv2_multi_gpu_trainner as multi_gpu_trainner
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

LOG = init_logger.get_logger('train_bisenetv2_cityscapes')
CFG = parse_config_utils.cityscapes_cfg_v2


def train_model():
    """

    :return:
    """
    if CFG.TRAIN.MULTI_GPU.ENABLE:
        LOG.info('Using multi gpu trainner ...')
        worker = multi_gpu_trainner.BiseNetV2CityScapesMultiTrainer()
    else:
        LOG.info('Using single gpu trainner ...')
        worker = single_gpu_trainner.BiseNetV2CityScapesTrainer()

    worker.train()
    return


if __name__ == '__main__':
    """
    main function
    """
    train_model()
