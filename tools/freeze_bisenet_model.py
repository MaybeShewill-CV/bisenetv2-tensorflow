#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/1/15 上午10:49
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : freeze_bisenet_model.py
# @IDE: PyCharm
"""
Freeze bisenet model
"""
import argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util

from bisenet_model import bisenet


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights_path', type=str, help='The ckpt file path')
    parser.add_argument('--pb_file_path', type=str, help='The output frozen pb file path',
                        default='./checkpoint/bisenetv2_cityscapes.pb')

    return parser.parse_args()


def load_graph_from_ckpt_file(weights_path):
    """

    :param weights_path:
    :return:
    """
    # construct compute graph
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 512, 1024, 3], name='input_tensor')
    net = bisenet.BiseNet(phase='test')
    prediction = net.inference(
        input_tensor=input_tensor,
        name='BiseNet',
        reuse=False
    )
    prediction = tf.squeeze(prediction, axis=0, name='final_output')
    prediction = tf.identity(prediction, name='final_output')

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.allocator_type = 'BFC'

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(0.9995)
        variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    # create a session
    sess = tf.Session(config=sess_config)

    # import best model
    saver.restore(sess, weights_path)  # variables

    # get graph definition
    gd = graph_util.remove_training_nodes(sess.graph_def)

    return gd, sess, prediction


def freeze_model(output_pb_file_path, sess, graph_def):
    """

    :param output_pb_file_path:
    :param sess
    :param graph_def:
    :return:
    """
    converted_graph_def = graph_util.convert_variables_to_constants(
        sess, graph_def, ["final_output"])
    tf.train.write_graph(converted_graph_def, './', output_pb_file_path, as_text=False)

    return


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    bisenet_gd, bisenet_sess, _ = load_graph_from_ckpt_file(args.weights_path)

    freeze_model(
        output_pb_file_path=args.pb_file_path,
        sess=bisenet_sess,
        graph_def=bisenet_gd
    )
