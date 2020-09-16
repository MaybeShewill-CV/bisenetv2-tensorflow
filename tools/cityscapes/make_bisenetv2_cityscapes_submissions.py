#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/29 上午11:09
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : make_bisenetv2_cityscapes_submissions.py
# @IDE: PyCharm
"""
Make cityscapes submission result
"""
import os
import os.path as ops
import argparse
import time

import tqdm
import cv2
import numpy as np
import tensorflow as tf

from local_utils.config_utils import parse_config_utils
from local_utils.cityspaces_dataset_utils import label_utils

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
    parser.add_argument('-w', '--pb_file_path', type=str, help='The model weights file path')
    parser.add_argument('-d', '--dataset_dir', type=str, help='The dataset dir')

    return parser.parse_args()


class CityscapesSubmittor(object):
    """

    """
    def __init__(self, pb_file_path, dataset_dir, class_nums=19):
        """

        :param pb_file_path:
        :param dataset_dir:
        :param class_nums:
        """
        # load computation graph
        self._sess_graph = self._load_graph_from_frozen_pb_file(
            frozen_pb_file_path=pb_file_path
        )

        # fetch node
        self._crop_size = CFG.AUG.EVAL_CROP_SIZE
        self._input_tensor_size = [int(tmp / 2) for tmp in CFG.AUG.EVAL_CROP_SIZE]
        self._input_tensor = self._sess_graph.get_tensor_by_name('prefix/input_tensor:0')
        self._probs = self._sess_graph.get_tensor_by_name('prefix/BiseNetV2/prob:0')
        self._probs = tf.squeeze(self._probs, axis=0, name='final_prob')

        # define session and gpu config
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        self._sess = tf.Session(config=sess_config, graph=self._sess_graph)

        # define dataset image paths
        self._test_image_index_file_path = ops.join(dataset_dir, 'image_file_index', 'test.txt')
        assert ops.exists(self._test_image_index_file_path), '{:s} not exist'.format(self._test_image_index_file_path)
        self._test_image_sample_paths = []
        with open(self._test_image_index_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                info = line.rstrip('\r').rstrip('\n').split(' ')
                self._test_image_sample_paths.append([info[0], info[1]])

        # define eval paramm
        self._class_nums = class_nums
        self._label_index = np.arange(0, self._class_nums, dtype=np.int8)
        self._dataset_dir = dataset_dir

    @classmethod
    def _load_graph_from_frozen_pb_file(cls, frozen_pb_file_path):
        """
        通过frozen模型权重文件加载模型计算图
        :param frozen_pb_file_path:
        :return:
        """
        # 解析pb文件
        with tf.gfile.GFile(frozen_pb_file_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
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

    @classmethod
    def _map_trainid_to_labelid(cls, train_id):
        """

        :param train_id:
        :return:
        """
        label_name = label_utils.trainId2label[train_id].name
        label_id = label_utils.name2label[label_name].id

        return label_id

    @classmethod
    def _decode_prediction_mask(cls, mask):
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

    @classmethod
    def _prepare_image(cls, src_image, input_tensor_size, is_label=False):
        """

        :param src_image:
        :param input_tensor_size:
        :param is_label:
        :return:
        """
        # prepare input image
        if is_label:
            src_image = cv2.resize(
                src_image,
                dsize=(input_tensor_size[0], input_tensor_size[1]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            src_image = src_image[:, :, (2, 1, 0)]
            src_image = cv2.resize(
                src_image,
                dsize=(input_tensor_size[0], input_tensor_size[1]),
                interpolation=cv2.INTER_LINEAR
            )

        if not is_label:
            src_image = src_image.astype('float32') / 255.0
            img_mean = np.array(CFG.DATASET.MEAN_VALUE).reshape((1, 1, len(CFG.DATASET.MEAN_VALUE)))
            img_std = np.array(CFG.DATASET.STD_VALUE).reshape((1, 1, len(CFG.DATASET.STD_VALUE)))
            src_image -= img_mean
            src_image /= img_std
        src_image = np.expand_dims(src_image, axis=0)
        return src_image

    def _run_session(self, input_image):
        """

        :return:
        """
        # prepare image
        input_image = self._prepare_image(
            src_image=input_image,
            input_tensor_size=self._input_tensor_size,
            is_label=False
        )
        # run session
        t_start = time.time()
        probs_value = self._sess.run(
            self._probs,
            feed_dict={self._input_tensor: input_image}
        )
        inference_cost_time = time.time() - t_start
        # restore image scale
        probs_value_split = cv2.split(probs_value)
        probs_value_merge_sources = []
        for prob in probs_value_split:
            prob_resize = cv2.resize(
                prob,
                dsize=(self._crop_size[0], self._crop_size[1]),
                interpolation=cv2.INTER_LINEAR
            )
            probs_value_merge_sources.append(prob_resize)
        probs_value = cv2.merge(probs_value_merge_sources)
        return probs_value, inference_cost_time

    def _convert_prediction_from_trainid_to_labelid(self, prediction):
        """

        :param prediction:
        :return:
        """
        trainids = np.unique(prediction)
        prediction_copy = np.zeros_like(prediction)
        for train_id in trainids:
            label_id = self._map_trainid_to_labelid(train_id)
            mask = np.where(prediction == train_id)
            prediction_copy[mask] = label_id

        return prediction_copy

    def _test_image(self, src_input_image):
        """

        :param src_input_image:
        :return:
        """
        final_probs = np.zeros(shape=(self._crop_size[1], self._crop_size[0], self._class_nums), dtype=np.float32)
        # split image
        split_image_data_table = dict()
        img_height, img_width = src_input_image.shape[:2]
        for h in range(0, img_height, self._crop_size[1]):
            if h + self._crop_size[1] >= img_height:
                h = img_height - self._crop_size[1]
            for w in range(0, img_width, self._crop_size[0]):
                if w + self._crop_size[0] >= img_width:
                    w = img_width - self._crop_size[0]
                block_image_data = src_input_image[h:h + self._crop_size[1], w:w + self._crop_size[0], :]
                split_image_data_table[(w, h)] = block_image_data

        # run session on each block
        tmp_final_probs_height = max(img_height, self._crop_size[1])
        tmp_final_probs_width = max(img_width, self._crop_size[0])
        tmp_final_probs = np.zeros(
            shape=(tmp_final_probs_height, tmp_final_probs_width, self._class_nums),
            dtype=np.float32
        )
        inference_cost_time = 0.0
        for block_coord, block_image_data in split_image_data_table.items():
            block_probs, inference_cost_time = self._run_session(input_image=block_image_data)
            output_h, output_w = block_probs.shape[:2]
            if output_h != self._crop_size[1] or output_w != self._crop_size[0]:
                raise ValueError('Size error')
            block_x, block_y = block_coord
            tmp_final_probs[block_y:block_y + self._crop_size[1], block_x:block_x + self._crop_size[0], :] += \
                block_probs
        if tmp_final_probs.shape[0] != self._crop_size[1] or tmp_final_probs.shape[1] != self._crop_size[0]:
            probs_value_split = cv2.split(tmp_final_probs)
            probs_value_merge_sources = []
            for prob in probs_value_split:
                prob_resize = cv2.resize(
                    prob,
                    dsize=(self._crop_size[0], self._crop_size[1]),
                    interpolation=cv2.INTER_LINEAR
                )
                probs_value_merge_sources.append(prob_resize)
            tmp_final_probs = cv2.merge(probs_value_merge_sources)
        final_probs += tmp_final_probs
        origin_preds = np.argmax(final_probs, axis=-1)
        mapped_pred = self._convert_prediction_from_trainid_to_labelid(origin_preds)

        return origin_preds, mapped_pred, inference_cost_time

    def process(self):
        """

        :return:
        """
        # prepare folder
        output_result_dir = ops.join(self._dataset_dir, 'submission_result')
        os.makedirs(output_result_dir, exist_ok=True)
        decode_mask_output_dir = ops.join(output_result_dir, 'decode_mask_image')
        os.makedirs(decode_mask_output_dir, exist_ok=True)
        submission_result_output_dir = ops.join(output_result_dir, 'submission_image')
        os.makedirs(submission_result_output_dir, exist_ok=True)

        # prepare test images
        test_samples_tqdm = tqdm.tqdm(self._test_image_sample_paths)

        for samples in test_samples_tqdm:
            t_start = time.time()
            src_image_name = ops.split(samples[0])[1]
            input_src_image_path = samples[0]
            input_src_image = cv2.imread(input_src_image_path, cv2.IMREAD_UNCHANGED)

            origin_preds, mapped_preds, infer_cost_time = self._test_image(
                src_input_image=input_src_image
            )

            decode_mask_image = self._decode_prediction_mask(mask=origin_preds)
            output_decode_mask_image_path = ops.join(decode_mask_output_dir, src_image_name)
            cv2.imwrite(output_decode_mask_image_path, decode_mask_image)

            output_mapped_mask_image_path = ops.join(
                submission_result_output_dir,
                '{:s}.png'.format(src_image_name.split('.')[0])
            )
            cv2.imwrite(output_mapped_mask_image_path, mapped_preds, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            t_cost = time.time() - t_start
            test_samples_tqdm.set_description('Testing cost time: {:.5f}s, inference cost time: {:.5f}'.format(
                t_cost, infer_cost_time)
            )

        print('Testing complete')


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    submittor = CityscapesSubmittor(
        pb_file_path=args.pb_file_path,
        dataset_dir=args.dataset_dir,
    )
    print(args.pb_file_path)
    submittor.process()
