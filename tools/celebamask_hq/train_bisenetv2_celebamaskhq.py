#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 下午5:46
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : train_bisenetv2_celebamaskhq.py
# @IDE: PyCharm
"""
Train bisenetv2 on celebamaskhq dataset
"""
from trainner.celebamask_hq import celebamask_hq_bisenetv2_single_gpu_trainner as single_gpu_trainner
from trainner.celebamask_hq import celebamask_hq_bisenetv2_multi_gpu_trainner as multi_gpu_trainner
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

LOG = init_logger.get_logger('train_bisenetv2_celebamaskhq')
CFG = parse_config_utils.celebamask_hq_cfg


def train_model():
    """

    :return:
    """
    if CFG.TRAIN.MULTI_GPU.ENABLE:
        LOG.info('Using multi gpu trainner ...')
        worker = multi_gpu_trainner.BiseNetV2CelebamaskhqMultiTrainer()
    else:
        LOG.info('Using single gpu trainner ...')
        worker = single_gpu_trainner.BiseNetV2CelebamaskhqTrainer()

    worker.train()
    return


if __name__ == '__main__':
    """
    main function
    """
    train_model()
