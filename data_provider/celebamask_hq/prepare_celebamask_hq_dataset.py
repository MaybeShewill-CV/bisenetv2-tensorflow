#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 下午2:52
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : prepare_celebamask_hq_dataset.py
# @IDE: PyCharm
"""
Prepare celebamask hq dataset
"""
import os
import os.path as ops
import argparse
import random

import cv2
import numpy as np
import loguru
import tqdm

LOG = loguru.logger
LABELS = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
          'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='The root dataset dir of celebamask hq dataset')

    return parser.parse_args()


def prepare_dataset(dataset_root_dir):
    """

    :param dataset_root_dir:
    :return:
    """
    source_image_dir = ops.join(dataset_root_dir, 'CelebA-HQ-img')
    assert ops.exists(source_image_dir), '{:s} not exist'.format(source_image_dir)
    annotation_image_dir = ops.join(dataset_root_dir, 'CelebAMask-HQ-mask-anno')
    assert ops.exists(annotation_image_dir), '{:s} not exist'.format(annotation_image_dir)
    output_label_image_dir = ops.join(dataset_root_dir, 'CelebAMaskHQ-mask')
    os.makedirs(output_label_image_dir, exist_ok=True)

    img_num = 30000
    label_info = []

    for k in tqdm.tqdm(range(img_num)):

        output_label_image_save_path = os.path.join(output_label_image_dir, '{:d}.png'.format(k))
        source_image_path = ops.join(source_image_dir, '{:d}.jpg'.format(k))
        assert ops.exists(source_image_path), '{:s} not exist'.format(source_image_path)
        if ops.exists(output_label_image_save_path):
            label_info.append([source_image_path, output_label_image_save_path])
            continue

        folder_num = k // 2000
        im_base = np.zeros((512, 512))
        for idx, label in enumerate(LABELS):
            anno_file_path = ops.join(
                annotation_image_dir,
                str(folder_num),
                str(k).rjust(5, '0') + '_' + label + '.png'
            )
            if ops.exists(anno_file_path):
                im = cv2.imread(anno_file_path, cv2.IMREAD_UNCHANGED)
                im = im[:, :, 0]
                im_base[im != 0] = (idx + 1)

        cv2.imwrite(output_label_image_save_path, im_base)
        label_info.append([source_image_path, output_label_image_save_path])

    return label_info


def generate_training_image_index_file(dataset_dir, label_info):
    """

    :param dataset_dir:
    :param label_info:
    :return:
    """
    assert len(label_info) != 0, 'Empty label info'

    label_nums = len(label_info)
    train_label_nums = int(label_nums * 0.85)
    val_label_nums = int(label_nums * 0.1)
    test_label_nums = int(label_nums * 0.05)

    random.shuffle(label_info)

    train_label_info = label_info[:train_label_nums]
    val_label_info = label_info[train_label_nums: train_label_nums + val_label_nums]
    test_label_info = label_info[train_label_nums + val_label_nums: train_label_nums + val_label_nums + test_label_nums]

    image_file_index_dir = ops.join(dataset_dir, 'image_file_index')
    os.makedirs(image_file_index_dir, exist_ok=True)

    train_image_index_file = ops.join(image_file_index_dir, 'train.txt')
    with open(train_image_index_file, 'w') as file:
        for info in train_label_info:
            file.write(' '.join(info) + os.linesep)

    val_image_index_file = ops.join(image_file_index_dir, 'val.txt')
    with open(val_image_index_file, 'w') as file:
        for info in val_label_info:
            file.write(' '.join(info) + os.linesep)

    test_image_index_file = ops.join(image_file_index_dir, 'test.txt')
    with open(test_image_index_file, 'w') as file:
        for info in test_label_info:
            file.write(' '.join(info) + os.linesep)

    print('Complete')
    return


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    tmp_label_info = prepare_dataset(args.dataset_dir)

    generate_training_image_index_file(
        dataset_dir=args.dataset_dir,
        label_info=tmp_label_info
    )
