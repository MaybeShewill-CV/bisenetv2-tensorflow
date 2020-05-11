#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 下午5:46
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : train_bisenet_pascal_voc.py
# @IDE: PyCharm
"""
Train bisenet on city spaces dataset
"""
from trainner import cityscapes_bisenet_trainner as trainner
from local_utils.log_util import init_logger

LOG = init_logger.INIT_LOG


def train_model():
    """

    :return:
    """
    worker = trainner.BiseNetCityScapesTrainer()
    worker.train()

    return


if __name__ == '__main__':
    """
    main function
    """
    train_model()
