#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/7 下午2:37
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : visualize_bisenetv2_attention_map.py
# @IDE: PyCharm
"""
Visualize attention map of Bilateral Guided Aggregation module in BisenetV2
"""
import os.path as ops
import argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.cityscapes_cfg_v2


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_image_path', type=str, help='The input image file path')
    parser.add_argument('-w', '--pb_file_path', type=str, help='The tensorflow pb model file path')

    return parser.parse_args()


def _load_tensors_from_pb(graph, pb_file, return_elements):
    """
    :param graph:
    :param pb_file:
    :param return_elements:
    :return:
    """
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())
    with graph.as_default():
        return_elements = tf.import_graph_def(
            frozen_graph_def,
            return_elements=return_elements
        )
    return return_elements


def _load_graph_from_frozen_pb_file(frozen_pb_file_path):
    """
    通过frozen模型权重文件加载模型计算图
    :param frozen_pb_file_path:
    :return:
    """
    # 解析pb文件
    with tf.gfile.GFile(frozen_pb_file_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 修复一些node attr
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
    # 加载计算图定义到默认的计算图
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def _visualize_attention_map(attention_map_tensor, name):
    """

    :param attention_map_tensor:
    :param name
    :return:
    """
    with tf.variable_scope(name_or_scope=name):
        max_val = tf.reduce_max(attention_map_tensor, axis=[1, 2], keepdims=True)
        min_val = tf.reduce_min(attention_map_tensor, axis=[1, 2], keepdims=True)
        vis_ret = (attention_map_tensor - min_val) * 255.0 / (max_val - min_val)
    return vis_ret


def visualize_model_attention_map(src_input_image, pb_file_path):
    """

    :param src_input_image:
    :param pb_file_path:
    :return:
    """
    assert ops.exists(src_input_image), '{:s} not exist'.format(src_input_image)
    assert ops.exists(pb_file_path), '{:s} not exist'.format(pb_file_path)

    sess_graph = _load_graph_from_frozen_pb_file(pb_file_path)
    for vv in sess_graph.as_graph_def().node:
        print(vv.name)
        pass

    sess_input_node = sess_graph.get_tensor_by_name('prefix/input_tensor:0')
    sess_ga_detail_input_node = sess_graph.get_tensor_by_name(
        'prefix/BiseNetV2/aggregation_branch/guided_aggregation_block/detail_branch/avg_pooling_block:0'
    )
    sess_ga_detail_output_node = sess_graph.get_tensor_by_name(
        'prefix/BiseNetV2/aggregation_branch/guided_aggregation_block/aggregation_features/guided_semantic_features:0'
    )
    sess_ga_detail_large_input_node = sess_graph.get_tensor_by_name(
        'prefix/BiseNetV2/aggregation_branch/guided_aggregation_block/detail_branch/1x1_conv_block/Conv2D:0'
    )
    sess_ga_detail_large_output_node = sess_graph.get_tensor_by_name(
        'prefix/BiseNetV2/aggregation_branch/guided_aggregation_block/aggregation_features/guided_detail_features:0'
    )
    sess_ga_final_output_node = sess_graph.get_tensor_by_name(
        'prefix/BiseNetV2/aggregation_branch/guided_aggregation_block/'
        'aggregation_features/aggregation_feature_output/relu:0'
    )

    with tf.Session(graph=sess_graph) as sess:
        src_image = cv2.imread(src_input_image, cv2.IMREAD_COLOR)
        src_image = src_image[:, :, (2, 1, 0)]
        input_image_block = cv2.resize(src_image, dsize=(1024, 512), interpolation=cv2.INTER_LINEAR)
        input_image_block_rescale = input_image_block.astype('float32') / 255.0
        img_mean = np.array(CFG.DATASET.MEAN_VALUE).reshape((1, 1, len(CFG.DATASET.MEAN_VALUE)))
        img_std = np.array(CFG.DATASET.STD_VALUE).reshape((1, 1, len(CFG.DATASET.STD_VALUE)))
        input_image_block_rescale -= img_mean
        input_image_block_rescale /= img_std
        channel_index = 24

        ga_detail_input_node_value, ga_detail_output_node_value = sess.run(
            [sess_ga_detail_input_node, sess_ga_detail_output_node],
            feed_dict={sess_input_node: [input_image_block_rescale]}
        )
        ga_detail_large_input_node_value, ga_detail_large_output_node_value = sess.run(
            [sess_ga_detail_large_input_node, sess_ga_detail_large_output_node],
            feed_dict={sess_input_node: [input_image_block_rescale]}
        )
        ga_final_output_node_value = sess.run(
            sess_ga_final_output_node,
            feed_dict={sess_input_node: [input_image_block_rescale]}
        )

        # origin_feature_map = np.array(ga_detail_input_node_value[0, :, :, channel_index], dtype=np.float32)
        # guided_feature_map = np.array(ga_detail_output_node_value[0, :, :, channel_index], dtype=np.float32)
        origin_feature_map = np.array(ga_detail_large_input_node_value[0, :, :, channel_index], dtype=np.float32)
        guided_feature_map = np.array(ga_detail_large_output_node_value[0, :, :, channel_index], dtype=np.float32)
        final_fused_feature_map = np.array(ga_final_output_node_value[0, :, :, channel_index], dtype=np.float32)

        plt.figure('src_input_image')
        plt.imshow(input_image_block[:, :, (2, 1, 0)])
        plt.figure('origin_feature_map')
        plt.imshow(origin_feature_map, cmap='jet')
        plt.figure('guided_feature_map')
        plt.imshow(guided_feature_map, cmap='jet')
        plt.figure('final_fused_feature_map')
        plt.imshow(final_fused_feature_map, cmap='jet')
        plt.show()

    return


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()
    visualize_model_attention_map(
        src_input_image=args.input_image_path,
        pb_file_path=args.pb_file_path
    )
