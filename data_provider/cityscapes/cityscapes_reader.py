#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 下午3:09
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : cityscapes_reader.py
# @IDE: PyCharm
"""
CityScapes dataset reader
"""
import os.path as ops

import cv2
import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image

from local_utils.augment_utils.cityscapes import augmentation_utils as aug
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.cityscapes_cfg_v2


class _CitySpacesDataset(object):
    """
    cityscapes dataset
    """
    def __init__(self, image_file_paths):
        """

        :param image_file_paths:
        """
        self._image_file_paths = image_file_paths
        self._epoch_nums = CFG.TRAIN.EPOCH_NUMS
        self._batch_size = CFG.TRAIN.BATCH_SIZE
        self._batch_count = 0
        self._sample_nums = len(image_file_paths)
        self._num_batchs = int(np.ceil(self._sample_nums / self._batch_size))

    @staticmethod
    def _pil_imread(file_path):
        """

        :param file_path:
        :return:
        """
        im = Image.open(file_path)
        return np.asarray(im)

    def _load_batch_images(self, image_paths):
        """

        :param image_paths:
        :return:
        """
        src_images = []
        label_images = []

        for paths in image_paths:
            src_images.append(cv2.imread(paths[0], cv2.IMREAD_COLOR))
            label_images.append(self._pil_imread(paths[1]))

        return src_images, label_images

    @staticmethod
    def _multiprocess_preprocess_images(src_images, label_images):
        """

        :param src_images:
        :param label_images:
        :return:
        """
        assert len(src_images) == len(label_images)

        output_src_images = []
        output_label_images = []
        resize_image_size = (int(CFG.AUG.TRAIN_CROP_SIZE[0] / 2), int(CFG.AUG.TRAIN_CROP_SIZE[1] / 2))

        for index, src_image in enumerate(src_images):
            output_src_image, output_label_image = aug.preprocess_image(
                src_image, label_images[index]
            )
            output_src_image = cv2.resize(output_src_image, resize_image_size, interpolation=cv2.INTER_LINEAR)
            output_label_image = cv2.resize(output_label_image, resize_image_size, interpolation=cv2.INTER_NEAREST)
            output_src_images.append(output_src_image)
            output_label_images.append(output_label_image)

        return output_src_images, output_label_images

    def __len__(self):
        """

        :return:
        """
        return self._num_batchs

    def __iter__(self):
        """

        :return:
        """
        return self

    def __next__(self):
        """

        :return:
        """
        with tf.device('/cpu:0'):
            if self._batch_count < self._num_batchs:
                batch_image_paths = self._image_file_paths[self._batch_count:self._batch_count + self._batch_size]
                batch_src_images, batch_label_images = self._load_batch_images(batch_image_paths)
                batch_src_images, batch_label_images = self._multiprocess_preprocess_images(
                    batch_src_images, batch_label_images
                )
                self._batch_count += 1

                return batch_src_images, batch_label_images
            else:
                self._batch_count = 0
                np.random.shuffle(self._image_file_paths)

                raise StopIteration


class CitySpacesReader(object):
    """
    City spaces dataset reader
    """
    def __init__(self):
        """

        """
        self._dataset_dir = CFG.DATASET.DATA_DIR
        self._batch_size = CFG.TRAIN.BATCH_SIZE
        self._train_image_index_file_path = CFG.DATASET.TRAIN_FILE_LIST
        self._val_image_index_file_path = CFG.DATASET.VAL_FILE_LIST
        self._test_image_index_file_path = CFG.DATASET.TEST_FILE_LIST

        self._train_image_paths = []
        self._val_image_paths = []
        self._test_image_paths = []
        self._load_train_val_image_index()
        np.random.shuffle(self._train_image_paths)
        np.random.shuffle(self._val_image_paths)
        np.random.shuffle(self._test_image_paths)

        self._train_dataset = _CitySpacesDataset(
            image_file_paths=self._train_image_paths
        )
        self._val_dataset = _CitySpacesDataset(
            image_file_paths=self._val_image_paths
        )
        self._test_dataset = _CitySpacesDataset(
            image_file_paths=self._test_image_paths
        )

    def _load_train_val_image_index(self):
        """

        :return:
        """
        try:
            with open(self._train_image_index_file_path, 'r') as file:
                for line in file:
                    line_info = line.rstrip('\r').rstrip('\n').strip(' ').split()
                    train_src_image_path = line_info[0]
                    train_label_image_path = line_info[1]
                    assert ops.exists(train_src_image_path), '{:s} not exist'.format(train_src_image_path)
                    assert ops.exists(train_label_image_path), '{:s} not exist'.format(train_label_image_path)

                    self._train_image_paths.append([train_src_image_path, train_label_image_path])
        except OSError as err:
            print(err)
            raise err
        try:
            with open(self._val_image_index_file_path, 'r') as file:
                for line in file:
                    line_info = line.rstrip('\r').rstrip('\n').strip(' ').split()
                    val_src_image_path = line_info[0]
                    val_label_image_path = line_info[1]
                    assert ops.exists(val_src_image_path), '{:s} not exist'.format(val_src_image_path)
                    assert ops.exists(val_label_image_path), '{:s} not exist'.format(val_label_image_path)

                    self._val_image_paths.append([val_src_image_path, val_label_image_path])
        except OSError as err:
            print(err)
            raise err
        try:
            with open(self._test_image_index_file_path, 'r') as file:
                for line in file:
                    line_info = line.rstrip('\r').rstrip('\n').strip(' ').split()
                    test_src_image_path = line_info[0]
                    test_label_image_path = line_info[1]
                    assert ops.exists(test_src_image_path), '{:s} not exist'.format(test_src_image_path)
                    assert ops.exists(test_label_image_path), '{:s} not exist'.format(test_label_image_path)

                    self._test_image_paths.append([test_src_image_path, test_label_image_path])
        except OSError as err:
            print(err)
            raise err

        return

    @property
    def train_dataset(self):
        """

        :param
        :return:
        """
        return self._train_dataset

    @property
    def val_dataset(self):
        """

        :return:
        """
        return self._val_dataset

    @property
    def test_dataset(self):
        """

        :return:
        """
        return self._test_dataset


if __name__ == '__main__':
    """
    test code
    """

    reader = CitySpacesReader()
    train_dataset = reader.train_dataset
    val_dataset = reader.val_dataset

    LABEL_CONTOURS = [(0, 0, 0),  # 0=road
                      # 1=sidewalk, 2=building, 3=wall, 4=fence, 5=pole
                      (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                      # 6=traffic light, 7=traffic sign, 8=vegetation, 9=terrain, 10=sky
                      (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                      # 11=person, 12=rider, 13=car, 14=truck, 15=bus
                      (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                      # 16=train, 17=motorcycle, 18=bicycle
                      (0, 64, 0), (128, 64, 0), (0, 192, 0)]


    def decode_inference_prediction(mask):
        """
        Decode batch of segmentation masks.
        :param mask: result of inference after taking argmax.
        :return:  A batch with num_images RGB images of the same size as the input.
        """
        unique_value = np.unique(mask)
        color_mask = np.zeros(shape=[mask.shape[0], mask.shape[1], 3], dtype=np.uint8)
        for index, value in enumerate(unique_value):
            if value == 0:
                continue
            if value == 255:
                continue
            idx = np.where(mask == value)
            color_mask[idx] = LABEL_CONTOURS[value]

        return color_mask

    import matplotlib.pyplot as plt

    for train_samples in tqdm.tqdm(train_dataset):
        src_imgs = train_samples[0]
        label_imgs = train_samples[1]

        print(src_imgs[2].shape)

        plt.figure('src')
        plt.imshow(src_imgs[5][:, :, (2, 1, 0)])
        plt.figure('mask')
        plt.imshow(label_imgs[5], cmap='gray')
        plt.figure('color_mask')
        plt.imshow(decode_inference_prediction(label_imgs[5]))

        plt.show()
        raise ValueError
