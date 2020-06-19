#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/1/15 上午10:49
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : freeze_cityscapes_bisenetv2_model.py
# @IDE: PyCharm
"""
Freeze bisenetv2 model
"""
import argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib

from bisenet_model import bisenet_v2
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.cityscapes_cfg_v2


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights_path', type=str, help='The ckpt file path')
    parser.add_argument('--frozen_pb_file_path', type=str, help='The output frozen pb file path',
                        default='./checkpoint/bisenetv2_cityscapes_frozen.pb')
    parser.add_argument('--optimized_pb_file_path', type=str, help='The output frozen pb file path',
                        default='./checkpoint/bisenetv2_cityscapes_optimized.pb')

    return parser.parse_args()


def load_graph_from_ckpt_file(weights_path):
    """

    :param weights_path:
    :return:
    """
    # construct compute graph
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 512, 1024, 3], name='input_tensor')
    net = bisenet_v2.BiseNetV2(phase='test', cfg=CFG)
    prediction = net.inference(
        input_tensor=input_tensor,
        name='BiseNetV2',
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


def optimize_inference_model(frozen_pb_file_path, output_pb_file_path):
    """

    :param frozen_pb_file_path:
    :param output_pb_file_path:
    :return:
    """
    input_graph = tf.GraphDef()
    with tf.gfile.GFile(frozen_pb_file_path, "rb") as f:
        data2read = f.read()
        input_graph.ParseFromString(data2read)

    optimized_graph = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def=input_graph,
        input_node_names=['input_tensor'],
        output_node_names=['final_output'],
        placeholder_type_enum=tf.float32.as_datatype_enum
    )

    with tf.gfile.GFile(output_pb_file_path, 'w') as f:
        f.write(optimized_graph.SerializeToString())
    return


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    bisenetv2_gd, bisenetv2_sess, _ = load_graph_from_ckpt_file(args.weights_path)

    freeze_model(
        output_pb_file_path=args.frozen_pb_file_path,
        sess=bisenetv2_sess,
        graph_def=bisenetv2_gd
    )

    # optimize_inference_model(
    #     frozen_pb_file_path=args.frozen_pb_file_path,
    #     output_pb_file_path=args.optimized_pb_file_path
    # )
