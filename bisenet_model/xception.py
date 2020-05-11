#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/10 下午2:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : xception.py.py
# @IDE: PyCharm
"""
Implement xception model
"""
import collections

import tensorflow as tf

from bisenet_model import cnn_basenet


class Xception(cnn_basenet.CNNBaseModel):
    """
    Xception model
    """
    def __init__(self, phase):
        """

        :param phase: whether testing or training
        """
        super(Xception, self).__init__()

        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._feature_maps = collections.OrderedDict()

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)

        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _xception_conv_block(self, input_tensor, k_size, output_channels, name,
                             stride=1, padding='VALID', need_bn=True, use_bias=False):
        """

        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param name:
        :param stride:
        :param padding:
        :param need_bn:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_bn:
                result = self.layerbn(
                    inputdata=result, is_training=self._is_training, name='bn', scale=False
                )
            result = self.relu(inputdata=result, name='xception_conv_block_output')

            return result

    def _xception_residual_conv_block(
            self, input_tensor, k_size, output_channels, name,
            stride=1, padding='VALID', need_bn=True, use_bias=False):
        """

        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param name:
        :param stride:
        :param padding:
        :param need_bn:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            result = self.relu(inputdata=result, name='relu')
            if need_bn:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn')
        return result

    def _xception_separate_conv_block(
            self, input_tensor, k_size, output_channels, name,
            stride=1, padding='SAME', need_bn=True, bn_scale=True):
        """

        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param name:
        :param stride:
        :param padding:
        :param need_bn:
        :param bn_scale:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.separate_conv(
                input_tensor=input_tensor,
                output_channels=output_channels,
                kernel_size=k_size,
                name='xception_separate_conv',
                depth_multiplier=1,
                padding=padding,
                stride=stride
            )
            if need_bn:
                result = self.layerbn(
                    inputdata=result,
                    is_training=self._is_training,
                    name='xception_separate_conv_bn',
                    scale=bn_scale
                )
        return result

    def _entry_flow(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            with tf.variable_scope(name_or_scope='stage_1'):
                result = self._xception_conv_block(
                    input_tensor=input_tensor,
                    k_size=3,
                    output_channels=32,
                    name='conv_block_1',
                    stride=2,
                    use_bias=False
                )
                result = self._xception_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=64,
                    name='conv_block_2',
                    stride=1,
                    use_bias=False
                )
                residual = self._xception_residual_conv_block(
                    input_tensor=result,
                    k_size=1,
                    output_channels=128,
                    name='residual_block',
                    stride=2,
                    padding='SAME',
                    use_bias=False
                )
            with tf.variable_scope(name_or_scope='stage_2'):
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=128,
                    name='separate_conv_block_1',
                    bn_scale=False
                )
                result = self.relu(inputdata=result, name='relu_1')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=128,
                    name='separate_conv_block_2'
                )
                result = self.maxpooling(
                    inputdata=result,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    name='maxpool'
                )
                residual = tf.add(residual, result, name='residual_block_add')
                self._feature_maps['downsample_4'] = self.relu(residual, name='downsample_4_features')
                residual = self._xception_residual_conv_block(
                    input_tensor=residual,
                    k_size=1,
                    output_channels=256,
                    name='residual_block',
                    stride=2,
                    use_bias=False
                )
            with tf.variable_scope(name_or_scope='stage_3'):
                result = self.relu(result, name='relu_1')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=256,
                    name='separate_conv_block_1',
                    bn_scale=False
                )
                result = self.relu(result, name='relu_2')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=256,
                    name='separate_conv_block_2'
                )
                result = self.maxpooling(
                    inputdata=result,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    name='maxpool'
                )
                residual = tf.add(residual, result, name='residual_block_add')
                self._feature_maps['downsample_8'] = self.relu(residual, name='downsample_8_features')
                residual = self._xception_residual_conv_block(
                    input_tensor=residual,
                    k_size=1,
                    output_channels=512,
                    name='residual_block',
                    stride=2,
                    use_bias=False
                )
            with tf.variable_scope(name_or_scope='stage_4'):
                result = self.relu(result, name='relu_1')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=512,
                    name='separate_conv_block_1',
                    bn_scale=False
                )
                result = self.relu(result, name='relu_2')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=512,
                    name='separate_conv_block_2'
                )
                result = self.maxpooling(
                    inputdata=result,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    name='maxpool'
                )
                residual = tf.add(residual, result, name='residual_block_add')
        return residual

    def _middle_flow(self, input_tensor, name, repeat_times=8):
        """

        :param input_tensor:
        :param name:
        :param repeat_times
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            residual = None
            input_tensor_proj = None
            for i in range(repeat_times):
                with tf.variable_scope(name_or_scope='repeat_stack_{:d}'.format(i + 1)):
                    if residual is None:
                        residual = input_tensor
                    if input_tensor_proj is None:
                        input_tensor_proj = input_tensor
                    result = self.relu(inputdata=input_tensor_proj, name='relu_1')
                    result = self._xception_separate_conv_block(
                        input_tensor=result,
                        k_size=3,
                        output_channels=512,
                        name='separate_conv_block_1',
                        bn_scale=False
                    )
                    result = self.relu(inputdata=result, name='relu_2')
                    result = self._xception_separate_conv_block(
                        input_tensor=result,
                        k_size=3,
                        output_channels=512,
                        name='separate_conv_block_2',
                        bn_scale=False
                    )
                    result = self.relu(inputdata=result, name='relu_3')
                    result = self._xception_separate_conv_block(
                        input_tensor=result,
                        k_size=3,
                        output_channels=512,
                        name='separate_conv_block_3'
                    )
                    residual = tf.add(residual, result, name='residual_block')
                    input_tensor_proj = residual
            self._feature_maps['downsample_16'] = self.relu(residual, name='downsample_16_features')
        return residual

    def _exit_flow(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            residual = input_tensor
            with tf.variable_scope(name_or_scope='stage_1'):
                result = self.relu(inputdata=input_tensor, name='relu_1')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=512,
                    name='separate_conv_block_1',
                    bn_scale=False
                )
                result = self.relu(inputdata=result, name='relu_2')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=1024,
                    name='separate_conv_block_2'
                )
                result = self.maxpooling(
                    inputdata=result,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    name='maxpooling'
                )
                residual = self._xception_residual_conv_block(
                    input_tensor=residual,
                    k_size=1,
                    output_channels=1024,
                    name='residual_block',
                    stride=2,
                    use_bias=True
                )
                residual = tf.add(residual, result, name='residual_block_add')
            with tf.variable_scope(name_or_scope='stage_2'):
                result = self._xception_separate_conv_block(
                    input_tensor=residual,
                    k_size=3,
                    output_channels=1536,
                    name='separate_conv_block_1',
                    bn_scale=False
                )
                result = self.relu(inputdata=result, name='relu_1')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=2048,
                    name='separate_conv_block_2',
                    bn_scale=False
                )
                result = self.relu(inputdata=result, name='relu_2')
                self._feature_maps['downsample_32'] = tf.identity(result)
            with tf.variable_scope(name_or_scope='stage_3'):
                self._feature_maps['global_avg_pool'] = tf.reduce_mean(result, axis=[1, 2], keepdims=True)
                result = self.globalavgpooling(
                    inputdata=result,
                    name='global_average_pooling'
                )
                result = self.fullyconnect(
                    inputdata=result,
                    out_dim=1001,
                    name='final_logits'
                )
        return result

    @property
    def feature_maps(self):
        """

        :return:
        """
        return self._feature_maps

    def build_net(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # firstly build entry flow
            entry_flow_output = self._entry_flow(
                input_tensor=input_tensor,
                name='entry_flow'
            )
            # secondly build middle flow
            middle_flow_output = self._middle_flow(
                input_tensor=entry_flow_output,
                name='middle_flow'
            )
            # thirdly exit flow
            exit_flow_output = self._exit_flow(
                input_tensor=middle_flow_output,
                name='exit_flow'
            )

        return exit_flow_output


if __name__ == '__main__':
    """
    test code
    """
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 960, 480, 3], name='test_input')
    model = Xception(phase='train')
    test_result = model.build_net(input_tensor=test_input_tensor, name='xception_net')
    print('Complete')
