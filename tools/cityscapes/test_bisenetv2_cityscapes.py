#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/16 上午11:15
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : test_bisenetv2_cityscapes.py
# @IDE: PyCharm
"""
Test bisenetv2 on cityspaces dataset
"""
import argparse

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from bisenet_model import bisenet_v2
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.cityscapes_cfg_v2
LABEL_CONTOURS = [(0, 0, 0),  # 0=road
                  # 1=sidewalk, 2=building, 3=wall, 4=fence, 5=pole
                  (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  # 6=traffic light, 7=traffic sign, 8=vegetation, 9=terrain, 10=sky
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                  # 11=person, 12=rider, 13=car, 14=truck, 15=bus
                  (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                  # 16=train, 17=motorcycle, 18=bicycle
                  (0, 64, 0), (128, 64, 0), (0, 192, 0)]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--src_image_path', type=str, help='The input source image')
    parser.add_argument('-w', '--weights_path', type=str, help='The model weights file path')

    return parser.parse_args()


def decode_prediction_mask(mask):
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


def preprocess_image(src_image, input_tensor_size):
    """

    :param src_image:
    :param input_tensor_size:
    :return:
    """
    output_image = src_image[:, :, (2, 1, 0)]
    output_image = cv2.resize(
        output_image,
        dsize=(input_tensor_size[0], input_tensor_size[1]),
        interpolation=cv2.INTER_LINEAR
    )
    output_image = output_image.astype('float32') / 255.0
    img_mean = np.array(CFG.DATASET.MEAN_VALUE).reshape((1, 1, len(CFG.DATASET.MEAN_VALUE)))
    img_std = np.array(CFG.DATASET.STD_VALUE).reshape((1, 1, len(CFG.DATASET.STD_VALUE)))
    output_image -= img_mean
    output_image /= img_std
    return output_image


def compute_iou(y_pred, y_true, num_classes):
    """

    :param y_pred:
    :param y_true:
    :param num_classes:
    :return:
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    idx = np.where(y_true <= num_classes - 1)
    y_pred = y_pred[idx]
    y_true = y_true[idx]
    current = confusion_matrix(y_true, y_pred)
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / union.astype(np.float32)

    return np.mean(iou)


def test_bisenet_cityspaces(image_path, weights_path):
    """

    :param image_path:
    :param weights_path:
    :return:
    """
    # define bisenet
    input_tensor_size = CFG.AUG.EVAL_CROP_SIZE
    input_tensor_size = [int(tmp / 2) for tmp in input_tensor_size]
    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[1, input_tensor_size[1], input_tensor_size[0], 3],
        name='input_tensor'
    )
    bisenet_model = bisenet_v2.BiseNetV2(phase='test', cfg=CFG)
    prediction = bisenet_model.inference(
        input_tensor=input_tensor,
        name='BiseNetV2',
        reuse=False
    )

    # define session and gpu config
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    # prepare input image
    src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(src_image, input_tensor_size)

    # run net and decode output prediction
    with sess.as_default():
        saver.restore(sess, weights_path)

        t_start = time.time()
        loop_times = 2000
        for i in range(loop_times):
            prediction_value = sess.run(
                fetches=prediction,
                feed_dict={
                    input_tensor: [preprocessed_image]
                }
            )
        t_cost = (time.time() - t_start) / loop_times
        print('Mean cost time: {:.5f}s'.format(t_cost))
        print('Mean fps: {:.5f}fps'.format(1.0 / t_cost))
        prediction_value = np.squeeze(prediction_value, axis=0)
        prediction_value = cv2.resize(
            prediction_value,
            dsize=(input_tensor_size[0] * 2, input_tensor_size[1] * 2),
            interpolation=cv2.INTER_NEAREST
        )

        print('Prediction mask unique label ids: {}'.format(np.unique(prediction_value)))

        prediction_mask_color = decode_prediction_mask(prediction_value)
        plt.figure('src_image')
        plt.imshow(src_image[:, :, (2, 1, 0)])
        plt.figure('prediction_mask_color')
        plt.imshow(prediction_mask_color[:, :, (2, 1, 0)])
        plt.show()


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    test_bisenet_cityspaces(
        image_path=args.src_image_path,
        weights_path=args.weights_path
    )
