#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 下午7:47
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : cityscapes_tf_io.py
# @IDE: PyCharm
"""
Cityscapes tensorflow dataset io module
"""
import os
import os.path as ops
import collections
import six

import tensorflow as tf
import numpy as np
import loguru

from local_utils.config_utils import parse_config_utils
from local_utils.augment_utils.cityscapes import augmentation_tf_utils as aug

CFG = parse_config_utils.cityscapes_cfg_v2
LOG = loguru.logger


def _int64_list_feature(values):
    """

    :param values:
    :return:
    """
    if not isinstance(values, collections.Iterable):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
    """

    :param values:
    :return:
    """
    def _norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[_norm2bytes(values)]))


class _CityScapesTfWriter(object):
    """

    """
    def __init__(self):
        """

        """
        self._dataset_dir = CFG.DATASET.DATA_DIR
        self._tfrecords_dir = ops.join(self._dataset_dir, 'tfrecords')
        os.makedirs(self._tfrecords_dir, exist_ok=True)
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

    @classmethod
    def _write_example_tfrecords(cls, sample_image_paths, tfrecords_path):
        """
        write tfrecords
        :param sample_image_paths:
        :param tfrecords_path:
        :return:
        """
        tfrecords_dir = ops.split(tfrecords_path)[0]
        os.makedirs(tfrecords_dir, exist_ok=True)

        LOG.info('Writing {:s}....'.format(tfrecords_path))

        with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
            for sample_path in sample_image_paths:
                gt_src_image_path = sample_path[0]
                gt_label_image_path = sample_path[1]

                # prepare gt image
                gt_image_raw = tf.gfile.FastGFile(gt_src_image_path, 'rb').read()

                # prepare gt binary image
                gt_binary_image_raw = tf.gfile.FastGFile(gt_label_image_path, 'rb').read()

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'gt_src_image_raw': _bytes_list_feature(gt_image_raw),
                            'gt_label_image_raw': _bytes_list_feature(gt_binary_image_raw),
                        }))
                writer.write(example.SerializeToString())

        LOG.info('Writing {:s} complete'.format(tfrecords_path))

        return

    def write_tfrecords(self):
        """

        :return:
        """
        # generate training tfrecords
        train_tfrecords_file_name = 'cityscapes_train.tfrecords'
        train_tfrecords_file_path = ops.join(self._tfrecords_dir, train_tfrecords_file_name)
        self._write_example_tfrecords(
            sample_image_paths=self._train_image_paths,
            tfrecords_path=train_tfrecords_file_path
        )

        # generate validation tfrecords
        val_tfrecords_file_name = 'cityscapes_val.tfrecords'
        val_tfrecords_file_path = ops.join(self._tfrecords_dir, val_tfrecords_file_name)
        self._write_example_tfrecords(
            sample_image_paths=self._val_image_paths,
            tfrecords_path=val_tfrecords_file_path
        )

        LOG.info('Generating tfrecords complete')

        return


class _CityScapesTfReader(object):
    """

    """
    def __init__(self, dataset_flag):
        """

        :return:
        """
        self._dataset_dir = CFG.DATASET.DATA_DIR
        self._tfrecords_dir = ops.join(self._dataset_dir, 'tfrecords')
        self._epoch_nums = CFG.TRAIN.EPOCH_NUMS
        self._train_batch_size = CFG.TRAIN.BATCH_SIZE
        self._val_batch_size = CFG.TRAIN.VAL_BATCH_SIZE
        assert ops.exists(self._tfrecords_dir)

        self._dataset_flags = dataset_flag.lower()
        if self._dataset_flags not in ['train', 'val']:
            raise ValueError('flags of the data feeder should be \'train\', \'val\'')

    def __len__(self):
        """

        :return:
        """
        tfrecords_file_paths = ops.join(self._tfrecords_dir, 'cityscapes_{:s}.tfrecords'.format(self._dataset_flags))
        assert ops.exists(tfrecords_file_paths), '{:s} not exist'.format(tfrecords_file_paths)

        sample_counts = 0
        sample_counts += sum(1 for _ in tf.python_io.tf_record_iterator(tfrecords_file_paths))
        if self._dataset_flags == 'train':
            num_batchs = int(np.ceil(sample_counts / self._train_batch_size))
        elif self._dataset_flags == 'val':
            num_batchs = int(np.ceil(sample_counts / self._val_batch_size))
        else:
            raise ValueError('Wrong dataset flags')
        return num_batchs

    def next_batch(self, batch_size):
        """
        dataset feed pipline input
        :return: A tuple (images, labels), where:
                    * images is a float tensor with shape [batch_size, H, W, C]
                      in the range [-0.5, 0.5].
                    * labels is an int32 tensor with shape [batch_size, H, W, 1] with the label train id,
                      a number in the range [0, CLASS_NUMS).
        """
        tfrecords_file_paths = ops.join(self._tfrecords_dir, 'cityscapes_{:s}.tfrecords'.format(self._dataset_flags))
        assert ops.exists(tfrecords_file_paths), '{:s} not exist'.format(tfrecords_file_paths)

        with tf.device('/cpu:0'):
            with tf.name_scope('input_tensor'):

                # TFRecordDataset opens a binary file and reads one record at a time.
                # `tfrecords_file_paths` could also be a list of filenames, which will be read in order.
                dataset = tf.data.TFRecordDataset(tfrecords_file_paths)

                # The map transformation takes a function and applies it to every element
                # of the dataset.
                dataset = dataset.map(
                    map_func=aug.decode,
                    num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
                )
                if self._dataset_flags == 'train':
                    dataset = dataset.map(
                        map_func=aug.preprocess_image_for_train,
                        num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
                    )
                elif self._dataset_flags == 'val':
                    dataset = dataset.map(
                        map_func=aug.preprocess_image_for_val,
                        num_parallel_calls=CFG.DATASET.CPU_MULTI_PROCESS_NUMS
                    )

                # The shuffle transformation uses a finite-sized buffer to shuffle elements
                # in memory. The parameter is the number of elements in the buffer. For
                # completely uniform shuffling, set the parameter to be the same as the
                # number of elements in the dataset.
                dataset = dataset.shuffle(buffer_size=512)
                # repeat num epochs
                dataset = dataset.repeat(self._epoch_nums)

                dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
                dataset = dataset.prefetch(buffer_size=batch_size * 16)

                iterator = dataset.make_one_shot_iterator()

        return iterator.get_next(name='{:s}_IteratorGetNext'.format(self._dataset_flags))


class CityScapesTfIO(object):
    """

    """
    def __init__(self):
        """

        """
        self._writer = _CityScapesTfWriter()
        self._train_dataset_reader = _CityScapesTfReader(dataset_flag='train')
        self._val_dataset_reader = _CityScapesTfReader(dataset_flag='val')

    @property
    def writer(self):
        """

        :return:
        """
        return self._writer

    @property
    def train_dataset_reader(self):
        """

        :return:
        """
        return self._train_dataset_reader

    @property
    def val_dataset_reader(self):
        """

        :return:
        """
        return self._val_dataset_reader


if __name__ == '__main__':
    """
    test code
    """
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
        if len(mask.shape) == 3:
            mask = np.squeeze(mask, axis=-1)
        unique_value = np.unique(mask)
        print(unique_value)
        color_mask = np.zeros(shape=[mask.shape[0], mask.shape[1], 3], dtype=np.uint8)
        for index, value in enumerate(unique_value):
            if value == 0:
                continue
            if value == 255:
                continue
            idx = np.where(mask == value)
            try:
                color_mask[idx] = LABEL_CONTOURS[value]
            except IndexError as err:
                print(err)
                print(value)

        return color_mask

    import matplotlib.pyplot as plt
    import time

    io = CityScapesTfIO()
    src_images, label_images = io.val_dataset_reader.next_batch(batch_size=4)
    relu_ret = tf.nn.relu(src_images)

    count = 1
    with tf.Session() as sess:
        while True:
            try:
                t_start = time.time()
                images, labels = sess.run([src_images, label_images])
                print('Iter: {:d}, cost time: {:.5f}s'.format(count, time.time() - t_start))
                count += 1
                src_image = np.array((images[0] + 1.0) * 127.5, dtype=np.uint8)
                print(labels[0].shape)
                color_mask_image = decode_inference_prediction(mask=labels[0])

                plt.figure('src')
                plt.imshow(src_image)
                plt.figure('label')
                plt.imshow(color_mask_image)
                plt.show()
            except tf.errors.OutOfRangeError as err:
                print(err)
