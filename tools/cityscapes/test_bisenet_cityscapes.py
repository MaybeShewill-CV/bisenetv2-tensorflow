#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/16 上午11:15
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : test_bisenet_pascal_voc.py
# @IDE: PyCharm
"""
Test bisenet on cityspaces dataset
"""
import os.path as ops
import argparse

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from bisenet_model import bisenet
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.cityscapes_cfg
LABEL_CONTOURS = [(0, 0, 0),  # 0=road
                  # 1=sidewalk, 2=building, 3=wall, 4=fence, 5=pole
                  (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  # 6=traffic light, 7=traffic sign, 8=vegetation, 9=terrain, 10=sky
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                  # 11=person, 12=rider, 13=car, 14=truck, 15=bus
                  (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                  # 16=train, 17=motorcycle, 18=bicycle
                  (0, 64, 0), (128, 64, 0), (0, 192, 0)]
MASK_LABEL_IMG_DIR = 'PATH/IMAGE_SCENE_SEGMENTATION/CITYSPACES/gt_annotation/gtFine/train'


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--src_image_path', type=str, help='The input source image')
    parser.add_argument('-w', '--weights_path', type=str, help='The model weights file path')
    parser.add_argument('-s', '--save_path', type=str, help='The output mask image save path')

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


def split_ipm_image(src_ipm_image, block_width, block_height):
    """

    :param src_ipm_image:
    :param block_width:
    :param block_height:
    :return:
    """
    [src_image_height, src_image_width, _] = src_ipm_image.shape
    split_image_data_table = dict()

    for h in range(0, src_image_height, block_height):
        if h + block_height >= src_image_height:
            h = src_image_height - block_height - 1
        for w in range(0, src_image_width, block_width):
            if w + block_width >= src_image_width:
                w = src_image_width - block_width - 1

            block_image_data = src_ipm_image[h:h + block_height, w:w + block_width, :]
            split_image_data_table[(w, h)] = block_image_data
    return split_image_data_table


def _preprocess(image):
    """
    实现图像数据预处理接口
    :param image: np.ndarry图像数据, 强制数据类型
    :raise ValueError: 如果输入数据的类型不是ndarray类型会跑出ValueError
    :return: 预处理结果, ndarray
    """
    if not isinstance(image, np.ndarray):
        raise ValueError('Input image data should be np.ndarray rather than {}'.format(type(image)))
    # 图像大小归一化
    src_image_height, src_image_width = image.shape[:2]
    if src_image_height != CFG.AUG.EVAL_CROP_SIZE[0] or src_image_width != CFG.AUG.EVAL_CROP_SIZE[1]:
        image = cv2.resize(image, (CFG.AUG.EVAL_CROP_SIZE[0], CFG.AUG.EVAL_CROP_SIZE[1]),
                           interpolation=cv2.INTER_LINEAR)
    # 图像数据归一化
    image = image.astype(np.float32) - CFG.DATASET.MEAN_VALUE
    # 扩展图像dims
    image = np.expand_dims(image, axis=0)

    return image


def do_inference_on_single_image_block(sess, prediction, input_tensor, image_block):
    """

    :param image_block:
    :param prediction:
    :param input_tensor:
    :param sess
    :return:
    """
    if not isinstance(image_block, np.ndarray):
        raise ValueError('Input image data should be np.ndarray rather than {}'.format(type(image_block)))

    image = _preprocess(image_block)

    prediction_value = sess.run(
        fetches=[prediction],
        feed_dict={
            input_tensor: image
        }
    )

    prediction_value = np.squeeze(prediction_value, axis=[0, -1])
    prediction_value = np.array(prediction_value, dtype=np.uint8)
    return prediction_value


def do_inference_on_image_blocks(split_image_data, src_image_height, src_image_width, sess, preds, input_tensor):
    """

    :param split_image_data:
    :param src_image_width:
    :param src_image_height:
    :param sess:
    :param preds:
    :param input_tensor:
    :return:
    """
    segmentation_result = np.zeros(shape=[src_image_height, src_image_width], dtype=np.uint8)
    for block_coord, block_image_data in split_image_data.items():
        block_detection_result = do_inference_on_single_image_block(sess, preds, input_tensor, block_image_data)
        output_h, output_w = block_detection_result.shape[:2]
        if output_h != CFG.AUG.EVAL_CROP_SIZE[0] or output_w != CFG.AUG.EVAL_CROP_SIZE[1]:
            block_detection_result = cv2.resize(
                block_detection_result,
                dsize=(CFG.AUG.EVAL_CROP_SIZE[0], CFG.AUG.EVAL_CROP_SIZE[1]),
                interpolation=cv2.INTER_NEAREST
            )
        block_x, block_y = block_coord
        segmentation_result[block_y:block_y + CFG.AUG.EVAL_CROP_SIZE[0],
        block_x:block_x + CFG.AUG.EVAL_CROP_SIZE[1]] = block_detection_result

    return segmentation_result


def test_bisenet_cityspaces(image_path, weights_path):
    """

    :param image_path:
    :param weights_path:
    :return:
    """
    # define bisenet
    input_tensor_size = CFG.AUG.EVAL_CROP_SIZE
    input_tensor = tf.placeholder(
        dtype=tf.float32, shape=[1, input_tensor_size[1], input_tensor_size[0], 3], name='input_tensor'
    )
    bisenet_model = bisenet.BiseNet(phase='test')
    prediction = bisenet_model.inference(
        input_tensor=input_tensor,
        name='BiseNet',
        reuse=False
    )

    # define session and gpu config
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=sess_config)

    # define saver
    saver = tf.train.Saver(tf.global_variables())

    # prepare input image
    image_name = ops.split(image_path)[1]
    src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    src_image_vis = src_image
    # src_image = cv2.resize(src_image, (720, 720))
    split_image_data = split_ipm_image(src_image, input_tensor_size[1], input_tensor_size[0])

    # run net and decode output prediction
    with sess.as_default():
        saver.restore(sess, weights_path)

        t_start = time.time()
        for i in range(100):
            prediction_value = do_inference_on_image_blocks(
                split_image_data=split_image_data,
                src_image_height=src_image.shape[0],
                src_image_width=src_image.shape[1],
                sess=sess,
                preds=prediction,
                input_tensor=input_tensor
            )
        cost_time = time.time() - t_start
        print('Mean cost time: {:.5f}s'.format(cost_time / 100))
        print('Mean fps: {:.5f}fps'.format(100 / cost_time))

        image_name = ops.split(image_path)[1]
        city_name = image_name.split('_')[0]
        label_mask_image_name = image_name.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
        label_mask_image_path = ops.join(MASK_LABEL_IMG_DIR, city_name, label_mask_image_name)
        assert ops.exists(label_mask_image_path), '{:s} not exist'.format(label_mask_image_path)
        label_mask_image = cv2.imread(label_mask_image_path, cv2.IMREAD_GRAYSCALE)

        print('Prediction mask unique label ids: {}'.format(np.unique(prediction_value)))
        print('Label mask unique label ids: {}'.format(np.unique(label_mask_image)))
        miou = compute_iou(
            y_pred=np.reshape(prediction_value, newshape=[-1, ]),
            y_true=np.reshape(label_mask_image, newshape=[-1, ]),
            num_classes=CFG.DATASET.NUM_CLASSES,
        )
        print('Miou: {:.5f}'.format(miou))

        prediction_mask_color = decode_prediction_mask(prediction_value)
        label_mask_color = decode_prediction_mask(label_mask_image)
        plt.figure('src_image')
        plt.imshow(src_image_vis[:, :, (2, 1, 0)])
        plt.figure('prediction_mask_color')
        plt.imshow(prediction_mask_color[:, :, (2, 1, 0)])
        plt.figure('label_mask_color')
        plt.imshow(label_mask_color[:, :, (2, 1, 0)])
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
