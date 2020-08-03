#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/10 下午5:55
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : timeprofile_cityscapes_bisenetv2.py
# @IDE: PyCharm
"""
Test tensorrt bisenetv2 inference time consuming
"""
import argparse
import os.path as ops
import time

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from local_utils.config_utils import parse_config_utils

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
CFG = parse_config_utils.cityscapes_cfg_v2

PB_FILE_PATH = './checkpoint/cityscapes/bisenetv2_cityscapes_frozen.pb'
ONNX_MODEL_FILE_PATH = './checkpoint/cityscapes/bisenetv2_cityscapes_frozen.onnx'
TRT_ENGINE_FILE_PATH = './checkpoint/cityscapes/bisenetv2_cityscapes_frozen.trt'

LABEL_CONTOURS = [(0, 0, 0),  # 0=road
                  # 1=sidewalk, 2=building, 3=wall, 4=fence, 5=pole
                  (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  # 6=traffic light, 7=traffic sign, 8=vegetation, 9=terrain, 10=sky
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                  # 11=person, 12=rider, 13=car, 14=truck, 15=bus
                  (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                  # 16=train, 17=motorcycle, 18=bicycle
                  (0, 64, 0), (128, 64, 0), (0, 192, 0)]


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


def _decode_prediction_mask(mask):
    """

    :param mask:
    :return:
    """
    mask_shape = mask.shape
    mask_color = np.zeros(shape=[mask_shape[0], mask_shape[1], 3], dtype=np.uint8)

    unique_label_ids = [v for v in np.unique(mask) if v != 0 and v != 255]

    for label_id in unique_label_ids:
        idx = np.where(mask == label_id)
        mask_color[idx] = LABEL_CONTOURS[label_id]

    return mask_color


def init_args():
    """

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pb_file_path',
        type=str,
        help='The frozen tensorflow pb file path',
        default=PB_FILE_PATH
    )
    parser.add_argument(
        '--onnx_file_path',
        type=str,
        help='The converted onnx model file path',
        default=ONNX_MODEL_FILE_PATH
    )
    parser.add_argument(
        '--output_trt_file_path',
        type=str,
        help='The output tensorrt engien file path',
        default=TRT_ENGINE_FILE_PATH
    )
    parser.add_argument(
        '--input_image_path',
        type=str,
        help='The input testing image file path'
    )

    return parser.parse_args()


def time_profile_tensorflow_graph(image_file_path, pb_file_path):
    """

    """
    assert ops.exists(pb_file_path), '{:s} not exist'.format(pb_file_path)

    sess_graph = _load_graph_from_frozen_pb_file(pb_file_path)
    input_tensor = sess_graph.get_tensor_by_name('prefix/input_tensor:0')
    output_tensor = sess_graph.get_tensor_by_name('prefix/final_output:0')

    image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    image_feed = image[:, :, (2, 1, 0)]
    image_feed = cv2.resize(image_feed, (1024, 512), interpolation=cv2.INTER_LINEAR)
    image_feed = image_feed.astype('float32') / 255.0
    img_mean = np.array(CFG.DATASET.MEAN_VALUE).reshape((1, 1, len(CFG.DATASET.MEAN_VALUE)))
    img_std = np.array(CFG.DATASET.STD_VALUE).reshape((1, 1, len(CFG.DATASET.STD_VALUE)))
    image_feed -= img_mean
    image_feed /= img_std

    loops = 5001
    with tf.Session(graph=sess_graph) as sess:
        t_start = time.time()
        tmp_cost_time = 0.0
        for i in range(loops):
            if i == 0:
                tmp_t_start = time.time()
            ret = sess.run(
                output_tensor,
                feed_dict={input_tensor: [image_feed]}
            )
            if i == 0:
                tmp_cost_time = time.time() - tmp_t_start
        t_cost_time = (time.time() - t_start - tmp_cost_time) / (loops - 1)
        print('Origin tensorflow graph inference cost time: {:.5f}'.format(t_cost_time))
        print('Origin tensorflow graph inference fps: {:.5f}'.format(1 / t_cost_time))

    mask_color = _decode_prediction_mask(ret)
    mask_color = cv2.resize(mask_color, (2048, 1024))

    plt.figure('decode_result')
    plt.imshow(mask_color[:, :, (2, 1, 0)])
    plt.figure('src')
    plt.imshow(image[:, :, (2, 1, 0)])
    plt.show()

    return


def convert_onnx_into_tensorrt_engine(onnx_model_file_path, trt_engine_output_file):
    """

    :param onnx_model_file_path:
    :param trt_engine_output_file:
    :return:
    """
    if ops.exists(trt_engine_output_file):
        print('Trt engine file: {:s} has been generated'.format(trt_engine_output_file))
        return
    try:
        with trt.Builder(TRT_LOGGER) as builder:
            explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with builder.create_network(explicit_batch) as network:
                with trt.OnnxParser(network, TRT_LOGGER) as parser:
                    # Parse the model to create a network.
                    with open(onnx_model_file_path, 'rb') as model:
                        parser.parse(model.read())
                        for error_index in range(parser.num_errors):
                            print(parser.get_error(error_index).desc())
                            print(parser.get_error(error_index).code())
                            print(parser.get_error(error_index).file())

                    # Configure the builder here.
                    builder.max_batch_size = 8
                    builder.max_workspace_size = 1 << 32

                    # Build and return the engine. Note that the builder,
                    # network and parser are destroyed when this function returns.
                    engine = builder.build_cuda_engine(network)
                    if engine is not None:
                        with open(trt_engine_output_file, "wb") as f:
                            f.write(engine.serialize())
                        print('Successfully construct trt engine')
                        return engine
                    else:
                        print('Failed construct trt engine')
                        return engine
    except Exception as err:
        print(err)
        print('Failed to construct trt engine')
        return None


def time_profile_trt_engine(image_file_path, trt_engine_file_path):
    """

    """
    with open(trt_engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # read input image file
    h_input = np.empty(shape=[512, 1024, 3], dtype=np.float32)
    h_output = np.empty(shape=[512, 1024], dtype=np.int32)

    # Alocate device memory
    d_input = cuda.mem_alloc(1 * h_input.nbytes)
    d_output = cuda.mem_alloc(1 * h_output.nbytes)
    bindings = [int(d_input), int(d_output)]

    # read images
    src_image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    src_image = src_image[:, :, (2, 1, 0)]
    src_image = cv2.resize(src_image, dsize=(1024, 512), interpolation=cv2.INTER_LINEAR)
    src_image = src_image.astype('float32') / 255.0
    img_mean = np.array(CFG.DATASET.MEAN_VALUE).reshape((1, 1, len(CFG.DATASET.MEAN_VALUE)))
    img_std = np.array(CFG.DATASET.STD_VALUE).reshape((1, 1, len(CFG.DATASET.STD_VALUE)))
    src_image -= img_mean
    src_image /= img_std

    loop_times = 5000

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    with engine.create_execution_context() as context:
        t_start = time.time()
        for i in range(loop_times):
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(d_input, src_image, stream)
            # Run inference.
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
        cost_time = (time.time() - t_start) / loop_times
        # Synchronize the stream
        stream.synchronize()
        print('Inference cost: {:.5f}'.format(cost_time))
        print('Inference fps: {:.5f}'.format(1 / cost_time))

    decode_color_mask = _decode_prediction_mask(h_output)
    plt.figure('decode_color_mask')
    plt.imshow(decode_color_mask[0:, :, (2, 1, 0)])
    plt.show()

    return


def estimate_model_gflops(pb_model_file_path):
    """

    :param pb_model_file_path:
    :return:
    """
    graph = _load_graph_from_frozen_pb_file(pb_model_file_path)
    with graph.as_default():
        # placeholder input would result in incomplete shape. So replace it with constant during model frozen.
        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('Model {} needs {} FLOPS after freezing'.format(pb_model_file_path, flops.total_float_ops))
    print('Model {} needs {} GFLOPS after freezing'.format(pb_model_file_path, flops.total_float_ops / 1e9))


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()
    if args.input_image_path is None:
        raise ValueError('Failed init args input image path field')
    if not ops.exists(args.input_image_path):
        raise ValueError('Input image path: {:s} not exist'.format(args.input_image_path))

    # first convert onnx model into trt engine
    trt_engine = convert_onnx_into_tensorrt_engine(
        onnx_model_file_path=args.onnx_file_path,
        trt_engine_output_file=args.output_trt_file_path
    )
    # second timeprofile origin tensorflow frozen model
    time_profile_tensorflow_graph(
        image_file_path=args.input_image_path,
        pb_file_path=args.pb_file_path
    )
    # third timeprofile trt engine
    time_profile_trt_engine(
        image_file_path=args.input_image_path,
        trt_engine_file_path=args.output_trt_file_path
    )
    # fourth estimate model gflops
    estimate_model_gflops(
        pb_model_file_path=args.pb_file_path
    )
