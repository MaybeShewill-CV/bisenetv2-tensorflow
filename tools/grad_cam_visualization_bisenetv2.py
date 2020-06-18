#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/14 下午5:20
# @Author  : LuoYao
# @Site    : ICode
# @File    : grad_cam_visualization_bisenetv2.py
# @IDE: PyCharm
"""
Grad_CAM visualization result
"""
import argparse
import os.path as ops

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from bisenet_model import bisenet_v2


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    """

    """
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))


def init_args():
    """

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_image_path',
        type=str,
        help='The input image file path',
        default='./data/test_image/test_01.png'
    )
    parser.add_argument(
        '--weights_path',
        type=str,
        help='The weight file path',
        default='./model/cityscapes/bisenetv2/cityscapes.ckpt'
    )

    return parser.parse_args()


def build_model():
    """

    """
    graph = tf.Graph()
    with graph.as_default():
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 512, 1024, 3], name='input_tensor')
        label_tensor = tf.placeholder(dtype=tf.int32, shape=[1, 512, 1024], name='input_label')
        net = bisenet_v2.BiseNetV2(phase='train')
        loss = net.compute_loss(
            input_tensor=input_tensor,
            label_tensor=label_tensor,
            name='BiseNetV2',
            reuse=False
        )

        final_output_tensor = graph.get_tensor_by_name(
            'BiseNetV2/logits/segmentation_head_logits:0'
        )
        stage_1_output_tensor = graph.get_tensor_by_name(
            'BiseNetV2/detail_branch/stage_1/conv_block_2_repeat_1/3x3_conv/relu:0'
        )
        stage_2_output_tensor = graph.get_tensor_by_name(
            'BiseNetV2/detail_branch/stage_2/conv_block_2_repeat_2/3x3_conv/relu:0'
        )
        stage_3_output_tensor = graph.get_tensor_by_name(
            'BiseNetV2/detail_branch/stage_3/conv_block_2_repeat_1/3x3_conv/conv/conv:0'
        )

        y_c = tf.reduce_sum(tf.multiply(final_output_tensor, label_tensor), axis=1)

    return graph


def fetch_target_layer(graph, layer_names):
    """

    """
    returned_layer = []
    for layer_name in layer_names:
        try:
            layer = graph.get_tensor_by_name(layer_name)
        except Exception as err:
            print(err)
            layer = None
        returned_layer.append(layer)

    return dict(zip(layer_names, returned_layer))


def vis_grad_cam(input_image_path, weights_path):
    """

    """
    graph = build_model()

    for vv in graph.as_graph_def().node:
        print(vv.name)

    return


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    vis_grad_cam(
        input_image_path=args.input_image_path,
        weights_path=args.weights_path
    )
