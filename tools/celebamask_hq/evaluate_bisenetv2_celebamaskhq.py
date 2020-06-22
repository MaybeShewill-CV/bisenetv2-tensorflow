#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 下午5:45
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : evaluate_bisenetv2_celebamaskhq.py
# @IDE: PyCharm
"""
Evaluate bisenetv2 model in celebamaskhq dataset
"""
import os.path as ops
import argparse

import tqdm
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from local_utils.config_utils import parse_config_utils
from local_utils.augment_utils.celebamask_hq import augmentation_utils as aug

CFG = parse_config_utils.celebamask_hq_cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--pb_file_path', type=str, help='The model weights file path')
    parser.add_argument('-d', '--dataset_dir', type=str, help='The dataset dir',
                        default='./data/example_dataset/cityscapes')
    parser.add_argument('--min_scale', type=float, default=1.0, help='min rescale ratio')
    parser.add_argument('--max_scale', type=float, default=1.0, help='max rescale ratio')
    parser.add_argument('--scale_step_size', type=float, default=0.25, help='rescale ratio step size')
    parser.add_argument('--mirror', type=args_str2bool, default=False, help='Mirror image during evaluation')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class MultiScaleEvaluator(object):
    """

    """
    def __init__(self, pb_file_path, dataset_dir, min_scale=0.75, max_scale=2.0, scale_step=0.25,
                 class_nums=CFG.DATASET.NUM_CLASSES, crop_size=CFG.AUG.EVAL_CROP_SIZE, need_flip=False):
        """

        :param pb_file_path:
        :param dataset_dir:
        :param min_scale:
        :param max_scale:
        :param scale_step:
        :param class_nums:
        :param crop_size:
        :param need_flip:
        """
        # load computation graph
        self._sess_graph = self._load_graph_from_frozen_pb_file(
            frozen_pb_file_path=pb_file_path
        )

        # fetch node
        self._crop_size = crop_size
        self._input_tensor_size = [int(tmp) for tmp in crop_size]
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
        self._val_image_index_file_path = ops.join(dataset_dir, 'image_file_index', 'val.txt')
        assert ops.exists(self._val_image_index_file_path), '{:s} not exist'.format(self._val_image_index_file_path)
        self._val_image_sample_paths = []
        with open(self._val_image_index_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                info = line.rstrip('\r').rstrip('\n').split(' ')
                self._val_image_sample_paths.append([info[0], info[1]])

        # define eval params
        self._eval_scales = np.arange(min_scale, max_scale + scale_step, scale_step)
        self._class_nums = class_nums
        self._label_index = np.arange(0, self._class_nums, dtype=np.int8)
        self._need_flip = need_flip

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

    @classmethod
    def _prepare_image(cls, src_image, input_tensor_size, is_label=False):
        """

        :param src_image:
        :param input_tensor_size:
        :param is_label:
        :return:
        """
        # prepare input image
        if not is_label:
            src_image = src_image.astype('float32') / 255.0
            img_mean = np.array(CFG.DATASET.MEAN_VALUE).reshape((1, 1, len(CFG.DATASET.MEAN_VALUE)))
            img_std = np.array(CFG.DATASET.STD_VALUE).reshape((1, 1, len(CFG.DATASET.STD_VALUE)))
            src_image -= img_mean
            src_image /= img_std

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
        src_image = np.expand_dims(src_image, axis=0)
        return src_image

    @classmethod
    def _compute_miou_v1(cls, y_pred, y_true, labels):
        """

        :param y_pred:
        :param y_true:
        :return:
        """
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        current = confusion_matrix(y_true, y_pred, labels=labels)
        return current

    @classmethod
    def _compute_miou_v2(cls, pred, label, ignore_idx, n_classes):
        """

        :param pred:
        :param label:
        :param ignore_idx:
        :param n_classes:
        :return:
        """
        keep = np.logical_not(label == ignore_idx)
        merge = pred[keep] * n_classes + label[keep]
        hist = np.bincount(merge, minlength=n_classes ** 2)
        hist = hist.reshape((n_classes, n_classes))

        return hist

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
        probs_value = self._sess.run(
            self._probs,
            feed_dict={self._input_tensor: input_image}
        )
        if self._need_flip:
            probs_value_flip = self._sess.run(
                self._probs,
                feed_dict={self._input_tensor: input_image[:, ::-1, :]}
            )
            probs_value += probs_value_flip[:, ::-1, :]
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
        return probs_value

    def _scale_crop_evaluate(self, src_input_image, src_label_image):
        """

        :param src_input_image:
        :param src_label_image:
        :return:
        """
        final_probs = np.zeros(shape=(self._crop_size[1], self._crop_size[0], self._class_nums), dtype=np.float32)
        for scale in self._eval_scales:
            # scale image
            scaled_image, _ = aug.randomly_scale_image_and_label(
                image=src_input_image,
                label=src_label_image,
                scale=scale
            )
            # pad or crop image
            img_height, img_width = scaled_image.shape[:2]
            split_image_data_table = dict()
            if scale < 1.0:
                pad_height = max(self._crop_size[1] - img_height, 0)
                pad_width = max(self._crop_size[0] - img_width, 0)
                crop_img = cv2.copyMakeBorder(
                    scaled_image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=CFG.DATASET.PADDING_VALUE
                )
                split_image_data_table[(0, 0)] = crop_img
            else:
                for h in range(0, img_height, self._crop_size[1]):
                    if h + self._crop_size[1] >= img_height:
                        h = img_height - self._crop_size[1]
                    for w in range(0, img_width, self._crop_size[0]):
                        if w + self._crop_size[0] >= img_width:
                            w = img_width - self._crop_size[0]
                        block_image_data = scaled_image[h:h + self._crop_size[1], w:w + self._crop_size[0], :]
                        split_image_data_table[(w, h)] = block_image_data
            # run session on each block
            tmp_final_probs_height = max(img_height, self._crop_size[1])
            tmp_final_probs_width = max(img_width, self._crop_size[0])
            tmp_final_probs = np.zeros(
                shape=(tmp_final_probs_height, tmp_final_probs_width, self._class_nums),
                dtype=np.float32
            )
            for block_coord, block_image_data in split_image_data_table.items():
                block_probs = self._run_session(input_image=block_image_data)
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
        preds = np.argmax(final_probs, axis=-1)
        preds = cv2.resize(preds, (src_label_image.shape[1], src_label_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        confuse_matrix = self._compute_miou_v2(
            pred=preds,
            label=src_label_image,
            ignore_idx=255,
            n_classes=self._class_nums
        )
        return confuse_matrix

    def evaluate(self):
        """

        :return:
        """
        eval_samples_tqdm = tqdm.tqdm(self._val_image_sample_paths)

        total_confuse_matrix = np.zeros(shape=(self._class_nums, self._class_nums), dtype=np.float32)
        for samples in eval_samples_tqdm:
            input_src_image_path = samples[0]
            input_label_image_path = samples[1]

            input_src_image = cv2.imread(input_src_image_path, cv2.IMREAD_UNCHANGED)
            input_label_image = cv2.imread(input_label_image_path, cv2.IMREAD_UNCHANGED)

            confuse_matrix = self._scale_crop_evaluate(
                src_input_image=input_src_image,
                src_label_image=input_label_image
            )
            total_confuse_matrix += confuse_matrix

        intersection = np.diag(total_confuse_matrix)
        ground_truth_set = total_confuse_matrix.sum(axis=1)
        predicted_set = total_confuse_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        ious = intersection / union.astype(np.float32)
        miou_total = np.mean(ious)

        print('MIOUS: {:.5f}'.format(miou_total))


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    evaluator = MultiScaleEvaluator(
        pb_file_path=args.pb_file_path,
        dataset_dir=args.dataset_dir,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        scale_step=args.scale_step_size,
        need_flip=args.mirror
    )
    print(args.pb_file_path)
    evaluator.evaluate()
