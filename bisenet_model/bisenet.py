#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/11 下午4:19
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : bisenet.py
# @IDE: PyCharm
"""
BiseNet Model
"""
import tensorflow as tf

from bisenet_model import cnn_basenet
from bisenet_model import xception
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.cityscapes_cfg


class _AttentionRefine(cnn_basenet.CNNBaseModel):
    """
    Implemention of attention refinement module
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_AttentionRefine, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._k_size = 1
        self._stride = 1
        self._padding = 'SAME'

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
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
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=False)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        output_channels = kwargs['output_channels']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        with tf.variable_scope(name_or_scope=name_scope):
            input_proj = self.relu(input_tensor)
            input_proj = self._conv_block(
                input_tensor=input_proj,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='attention_project',
                use_bias=False,
                need_activate=True
            )
            result = tf.reduce_mean(input_proj, [1, 2], keepdims=True, name='global_avg_pool')
            result = self.conv2d(
                inputdata=result,
                out_channel=output_channels,
                kernel_size=self._k_size,
                padding=self._padding,
                stride=self._stride,
                use_bias=False,
                name='conv',
            )
            result = self.layerbn(
                inputdata=result,
                is_training=self._is_training,
                name='bn',
                scale=False
            )
            result = self.sigmoid(
                inputdata=result,
                name='sigmoid'
            )
            result = tf.multiply(input_proj, result, name='attention_refine_output')
        return result


class _FeatureFusion(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_FeatureFusion, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'
        self._stride = 1

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False):
        """
        conv block in ffm
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
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
            result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=False)
            result = self.relu(inputdata=result, name='relu')
            return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_feature_1 = kwargs['input_tensor_1']
        input_feature_2 = kwargs['input_tensor_2']
        name_scope = kwargs['name']
        output_channels = kwargs['output_channels']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']

        with tf.variable_scope(name_or_scope=name_scope):
            result = tf.concat([input_feature_1, input_feature_2], axis=-1, name='concat')
            result = self._conv_block(
                input_tensor=result,
                k_size=3,
                output_channels=output_channels,
                stride=self._stride,
                name='conv_block_1',
                padding=self._padding,
                use_bias=False
            )
            residual = result
            result = tf.reduce_mean(result, [1, 2], keepdims=True, name='global_avg_pool')
            result = self.conv2d(
                inputdata=result,
                out_channel=output_channels,
                kernel_size=1,
                padding=self._padding,
                stride=1,
                use_bias=True,
                name='conv_block_2'
            )
            result = self.relu(inputdata=result, name='relu')
            result = self.conv2d(
                inputdata=result,
                out_channel=output_channels,
                kernel_size=1,
                padding=self._padding,
                stride=1,
                use_bias=True,
                name='conv_block_3'
            )
            result = self.sigmoid(inputdata=result, name='sigmoid')
            result = tf.multiply(result, residual, name='mul_output')
            result = tf.add(result, residual, name='ffm_output')
        return result


class _SpatialCnn(cnn_basenet.CNNBaseModel):
    """
    Spatial cnn module mentioned in "Spatial As Deep: Spatial CNN for Traffic Scene Understanding"
    """
    def __init__(self, phase):
        super(_SpatialCnn, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'
        self._stride = 1
        self._need_separate = False

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

    def _separate_conv_block(self, input_tensor, depthwise_filter, pointwise_filter, stride, name, padding='SAME'):
        """

        :param input_tensor:
        :param depthwise_filter:
        :param pointwise_filter:
        :param stride:
        :param name:
        :param padding:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = tf.nn.separable_conv2d(
                input=input_tensor,
                depthwise_filter=depthwise_filter,
                pointwise_filter=pointwise_filter,
                strides=[1, stride, stride, 1],
                padding=padding,
                name='separate_conv'
            )
            result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=False)
            result = self.relu(inputdata=result, name='relu')
        return result

    def _conv_block(self, input_tensor, filter_kernel, stride,
                    name, padding='SAME'):
        """
        conv block in ffm
        :param input_tensor:
        :param filter_kernel:
        :param stride:
        :param name:
        :param padding:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = tf.nn.conv2d(
                input=input_tensor,
                filter=filter_kernel,
                strides=[1, stride, stride, 1],
                padding=padding,
                name='conv'
            )
            result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=False)
            result = self.relu(inputdata=result, name='relu')
        return result

    @classmethod
    def _fetch_features(cls, input_tensor, axis, loop):
        """

        :param input_tensor:
        :param axis:
        :param loop:
        :return:
        """
        feature_list = []
        for i in range(loop):
            if axis == 1:
                feature_list.append(tf.expand_dims(input_tensor[:, i, :, :], axis=axis))
            elif axis == 2:
                feature_list.append(tf.expand_dims(input_tensor[:, :, i, :], axis=axis))
            else:
                raise ValueError('Not supported axis here')
        return feature_list

    @classmethod
    def _merge_feature_list(cls, feature_list, stack_axis, squeeze_axis, name):
        """

        :param feature_list:
        :param stack_axis:
        :param squeeze_axis:
        :param name:
        :return:
        """
        feature_list.reverse()
        with tf.variable_scope(name_or_scope=name):
            stack_features = tf.stack(feature_list, axis=stack_axis, name='stack')
            squeeze_feature = tf.squeeze(stack_features, axis=squeeze_axis, name='squeeze')
        return squeeze_feature

    def _apply_message_passing(self, input_tensor, name, k_size, loop_range,
                               output_channels, need_reverse=False, need_separate=False):
        """
        apply message pass downward
        :param input_tensor:
        :param name:
        :param k_size:
        :param loop_range:
        :param output_channels:
        :param need_reverse:
        :param need_separate:
        :return:
        """
        features_before = input_tensor
        if need_reverse:
            features_before.reverse()
        features_after = [features_before[0]]

        with tf.variable_scope(name_or_scope=name):
            in_shape = input_tensor[0].get_shape().as_list()
            w_init = tf.contrib.layers.variance_scaling_initializer()
            if not need_separate:
                filter_shape = [k_size[0], k_size[1], in_shape[-1], output_channels]
                filter_kernel = tf.get_variable('conv_filter', filter_shape, initializer=w_init)
                for nh in range(1, loop_range):
                    feature = self._conv_block(
                        input_tensor=features_after[nh - 1],
                        filter_kernel=filter_kernel,
                        stride=self._stride,
                        name='conv_block_{:d}'.format(nh),
                        padding=self._padding
                    )
                    feature = tf.add(feature, features_before[nh])
                    features_after.append(feature)
            else:
                depth_multiplier = 1
                depthwise_filter_shape = [k_size[0], k_size[1], in_shape[-1], depth_multiplier]
                pointwise_filter_shape = [1, 1, in_shape[-1] * depth_multiplier, output_channels]
                depthwise_filter_kernel = tf.get_variable(
                    'depthwise_conv_filter', depthwise_filter_shape, initializer=w_init
                )
                pointwise_filter_kernel = tf.get_variable(
                    'pointwise_conv_filter', pointwise_filter_shape, initializer=w_init
                )
                for nh in range(1, loop_range):
                    feature = self._separate_conv_block(
                        input_tensor=features_after[nh - 1],
                        depthwise_filter=depthwise_filter_kernel,
                        pointwise_filter=pointwise_filter_kernel,
                        stride=self._stride,
                        name='separate_conv_block_{:d}'.format(nh),
                        padding=self._padding
                    )
                    feature = tf.add(feature, features_before[nh])
                    features_after.append(feature)
        return features_after

    def __call__(self, *args, **kwargs):
        """
        apply scnn module to features
        :param args:
        :param kwargs:
        :return:
        """
        input_features = kwargs['input_tensor']
        [_, h, w, c] = input_features.shape
        name_scope = kwargs['name']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        else:
            self._padding = 'SAME'
        if 'stride' in kwargs:
            self._stride = kwargs['stride']
        else:
            self._stride = 1

        with tf.variable_scope(name_or_scope=name_scope):
            # do downward spatial cnn
            feature_list = self._fetch_features(input_tensor=input_features, axis=1, loop=h)
            feature_top_down = self._apply_message_passing(
                input_tensor=feature_list,
                name='top_down_scnn',
                k_size=[1, 9],
                loop_range=h,
                output_channels=c,
                need_reverse=False,
                need_separate=self._need_separate
            )
            # do upward spatial cnn
            feature_down_top = self._apply_message_passing(
                input_tensor=feature_top_down,
                name='down_top_scnn',
                k_size=[1, 9],
                loop_range=h,
                output_channels=c,
                need_reverse=True,
                need_separate=self._need_separate
            )
            # merge features
            merge_features = self._merge_feature_list(
                feature_list=feature_down_top,
                stack_axis=1,
                squeeze_axis=2,
                name='du_merge_features'
            )
            # do leftward spatial cnn
            feature_list = self._fetch_features(input_tensor=merge_features, axis=2, loop=w)
            feature_left_right = self._apply_message_passing(
                input_tensor=feature_list,
                name='left_right_scnn',
                k_size=[9, 1],
                loop_range=w,
                output_channels=c,
                need_reverse=False,
                need_separate=self._need_separate
            )
            feature_right_left = self._apply_message_passing(
                input_tensor=feature_left_right,
                name='right_left_scnn',
                k_size=[9, 1],
                loop_range=w,
                output_channels=c,
                need_reverse=True,
                need_separate=self._need_separate
            )
            # merge features
            merge_features = self._merge_feature_list(
                feature_list=feature_right_left,
                stack_axis=2,
                squeeze_axis=3,
                name='lr_merge_features'
            )
        return merge_features


class _ContextPath(cnn_basenet.CNNBaseModel):
    """
    Implementation of context path in bisenet
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_ContextPath, self).__init__()
        self._backbone = xception.Xception(phase=phase)
        self._attention_refine_module = _AttentionRefine(phase=phase)
        self._ffm_module = _FeatureFusion(phase=phase)

        self._phase = phase
        self._is_training = self._is_net_for_training()

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in context path
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
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
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=False)
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=True)
        return result

    def build_context(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # firstly build backbone
            self._backbone.build_net(input_tensor=input_tensor, name='xception')
            # build attention refinement
            feature_maps = self._backbone.feature_maps
            downsample_16_attention_refine = self._attention_refine_module(
                input_tensor=feature_maps['downsample_16'],
                output_channels=128,
                name='downsamle_16_attention_refine_module'
            )
            downsample_32_attention_refine = self._attention_refine_module(
                input_tensor=feature_maps['downsample_32'],
                output_channels=128,
                name='downsamle_32_attention_refine_module'
            )
            # combine path
            global_avg_pool_feats = feature_maps['global_avg_pool']
            global_avg_pool_feats = self._conv_block(
                input_tensor=global_avg_pool_feats,
                k_size=1,
                output_channels=128,
                stride=1,
                name='global_avg_pool_conv_block',
                use_bias=False,
                need_activate=True
            )
            downsample_32_feature_map_size = feature_maps['downsample_32'].get_shape().as_list()
            upsample_32_feature_map = tf.image.resize_bilinear(
                global_avg_pool_feats, size=downsample_32_feature_map_size[1:3], name='upsample_32_feature_maps'
            )
            fused_32_feature_map = tf.add(
                downsample_32_attention_refine, upsample_32_feature_map, name='32_fused_feats_map'
            )
            downsample_16_feature_map_size = feature_maps['downsample_16'].get_shape().as_list()
            upsample_16_feature_map = tf.image.resize_bilinear(
                fused_32_feature_map, size=downsample_16_feature_map_size[1:3], name='upsample_16_feature_maps'
            )
            upsample_16_feature_map = self._conv_block(
                input_tensor=upsample_16_feature_map,
                k_size=3,
                output_channels=128,
                stride=1,
                name='upsample_16_conv_block',
                use_bias=False,
                need_activate=True
            )
            fused_16_feature_map = tf.add(
                downsample_16_attention_refine, upsample_16_feature_map, name='16_fused_feats_map'
            )
            downsample_8_feature_map_size = feature_maps['downsample_8'].get_shape().as_list()
            upsample_8_feature_map = tf.image.resize_bilinear(
                fused_16_feature_map, size=downsample_8_feature_map_size[1:3], name='upsample_8_feature_maps'
            )
            upsample_8_feature_map = self._conv_block(
                input_tensor=upsample_8_feature_map,
                k_size=3,
                output_channels=128,
                stride=1,
                name='upsample_8_conv_block',
                use_bias=False,
                need_activate=True
            )
        return upsample_8_feature_map, upsample_16_feature_map


class _SpatialPath(cnn_basenet.CNNBaseModel):
    """
    Implementation of spatial path in bisenet
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_SpatialPath, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False):
        """
        conv block in spatial path
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
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
            result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=False)
            result = self.relu(inputdata=result, name='relu')

        return result

    def build_spatial(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=64,
                stride=2,
                name='spatial_conv_block_1',
                use_bias=False
            )
            result = self._conv_block(
                input_tensor=result,
                k_size=3,
                output_channels=64,
                stride=2,
                name='spatial_conv_block_2',
                use_bias=False
            )
            result = self._conv_block(
                input_tensor=result,
                k_size=3,
                output_channels=64,
                stride=2,
                name='spatial_conv_block_3',
                use_bias=False
            )
            result = self._conv_block(
                input_tensor=result,
                k_size=1,
                output_channels=128,
                stride=1,
                name='spatial_conv_block_4',
                use_bias=False
            )
        return result


class BiseNet(cnn_basenet.CNNBaseModel):
    """
    Implementation of bisenet model
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(BiseNet, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._class_nums = CFG.DATASET.NUM_CLASSES
        self._weights_decay = CFG.SOLVER.WEIGHT_DECAY
        self._loss_type = CFG.SOLVER.LOSS_TYPE
        self._use_spatial_cnn = CFG.MODEL.BISENET.USE_SPATIAL_CNN

        self._context_module = _ContextPath(phase=phase)
        self._spatial_module = _SpatialPath(phase=phase)
        self._ffm_module = _FeatureFusion(phase=phase)
        self._scnn_module = _SpatialCnn(phase=phase)

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

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False):
        """
        conv block in bisenet
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
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
            result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=False)
            result = self.relu(inputdata=result, name='relu')
        return result

    def _build_net(self, input_tensor, phase):
        """

        :param input_tensor:
        :param phase:
        :return:
        """
        # build spatial and context
        spatial_output = self._spatial_module.build_spatial(input_tensor=input_tensor, name='spatial_path')
        context_output_8, context_output_16 = self._context_module.build_context(
            input_tensor=input_tensor, name='context_path'
        )
        # fusion spatial and context feature map
        fused_output = self._ffm_module(
            input_tensor_1=spatial_output,
            input_tensor_2=context_output_8,
            output_channels=256,
            name='ffm_module'
        )
        if self._use_spatial_cnn:
            fused_output = self._scnn_module(
                input_tensor=fused_output,
                name='scnn_fused_feature'
            )
        logits = self.conv2d(
            inputdata=fused_output,
            out_channel=self._class_nums,
            kernel_size=1,
            padding='SAME',
            stride=1,
            use_bias=False,
            name='logits'
        )
        input_size = input_tensor.get_shape().as_list()[1:3]
        logits = tf.image.resize_bilinear(
            logits, size=input_size, name='final_logits'
        )
        if phase.lower() not in ['train', 'validation']:
            return logits
        context_output_8 = self._conv_block(
            input_tensor=context_output_8,
            k_size=3,
            output_channels=128,
            stride=1,
            name='context_8_conv_block',
            use_bias=False
        )
        if self._use_spatial_cnn:
            context_output_8 = self._scnn_module(
                input_tensor=context_output_8,
                name='scnn_context_8_feature'
            )
        context_8_logits = self.conv2d(
            inputdata=context_output_8,
            kernel_size=1,
            out_channel=self._class_nums,
            stride=1,
            name='context_8_logits',
            use_bias=False
        )
        context_8_logits = tf.image.resize_bilinear(
            context_8_logits, size=input_size, name='context_8_final_logits'
        )
        context_output_16 = self._conv_block(
            input_tensor=context_output_16,
            k_size=3,
            output_channels=128,
            stride=1,
            name='context_16_conv_block',
            use_bias=False
        )
        if self._use_spatial_cnn:
            context_output_16 = self._scnn_module(
                input_tensor=context_output_16,
                name='scnn_context_16_feature'
            )
        context_16_logits = self.conv2d(
            inputdata=context_output_16,
            kernel_size=1,
            out_channel=self._class_nums,
            stride=1,
            name='context_16_logits',
            use_bias=False
        )
        context_16_logits = tf.image.resize_bilinear(
            context_16_logits, size=input_size, name='context_16_final_logits'
        )

        return logits, context_8_logits, context_16_logits

    def _compute_cross_entropy_loss(self, logits, context_8_logits, context_16_logits, label_tensor):
        """

        :param logits:
        :param context_8_logits:
        :param context_16_logits:
        :param label_tensor:
        :return:
        """
        # compute loss
        principal_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=logits)
        )
        principal_loss = tf.identity(principal_loss, name='principal_loss')
        joint_loss_8 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=context_8_logits)
        )
        joint_loss_8 = tf.identity(joint_loss_8, name='joint_loss_8')
        joint_loss_16 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=context_16_logits)
        )
        joint_loss_16 = tf.identity(joint_loss_16, name='joint_loss_16')
        l2_reg_loss = tf.constant(0.0, tf.float32)
        for vv in tf.trainable_variables():
            if 'beta' in vv.name or 'gamma' in vv.name:
                continue
            else:
                l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
        l2_reg_loss *= self._weights_decay
        l2_reg_loss = tf.identity(l2_reg_loss, 'l2_loss')
        total_loss = principal_loss + joint_loss_8 + joint_loss_16 + l2_reg_loss
        total_loss = tf.identity(total_loss, name='total_loss')

        ret = {
            'total_loss': total_loss,
            'principal_loss': principal_loss,
            'joint_loss_8': joint_loss_8,
            'joint_loss_16': joint_loss_16,
            'l2_loss': l2_reg_loss,
        }

        return ret

    def _compute_dice_loss(self, logits, context_8_logits, context_16_logits, label_tensor):
        """
        dice loss is combined with bce loss here
        :param logits:
        :param context_8_logits:
        :param context_16_logits:
        :param label_tensor:
        :return:
        """
        def __dice_loss(_y_pred, _y_true):
            """

            :param _y_pred:
            :param _y_true:
            :return:
            """
            _intersection = tf.reduce_sum(_y_true * _y_pred, axis=-1)
            _l = tf.reduce_sum(_y_pred * _y_pred, axis=-1)
            _r = tf.reduce_sum(_y_true * _y_true, axis=-1)
            _dice = (2.0 * _intersection + 1e-5) / (_l + _r + 1e-5)
            _dice = tf.reduce_mean(_dice)
            return 1.0 - _dice

        # compute dice loss
        local_label_tensor = tf.one_hot(label_tensor, depth=self._class_nums, dtype=tf.float32)
        principal_loss_dice = __dice_loss(tf.nn.softmax(logits), local_label_tensor)
        principal_loss_dice = tf.identity(principal_loss_dice, name='principal_loss_dice')
        joint_loss_8_dice = __dice_loss(tf.nn.softmax(context_8_logits), local_label_tensor)
        joint_loss_8_dice = tf.identity(joint_loss_8_dice, name='joint_loss_8_dice')
        joint_loss_16_dice = __dice_loss(tf.nn.softmax(context_16_logits), local_label_tensor)
        joint_loss_16_dice = tf.identity(joint_loss_16_dice, name='joint_loss_16_dice')

        # compute bce loss
        principal_loss_bce = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=logits)
        )
        principal_loss_bce = tf.identity(principal_loss_bce, name='principal_loss_bce')
        joint_loss_8_bce = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=context_8_logits)
        )
        joint_loss_8_bce = tf.identity(joint_loss_8_bce, name='joint_loss_8_bce')
        joint_loss_16_bce = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=context_16_logits)
        )
        joint_loss_16_bce = tf.identity(joint_loss_16_bce, name='joint_loss_16_bce')

        # compute l2 loss
        l2_reg_loss = tf.constant(0.0, tf.float32)
        for vv in tf.trainable_variables():
            if 'beta' in vv.name or 'gamma' in vv.name:
                continue
            else:
                l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
        l2_reg_loss *= self._weights_decay
        l2_reg_loss = tf.identity(l2_reg_loss, 'l2_loss')
        total_loss = \
            principal_loss_dice + joint_loss_8_dice + joint_loss_16_dice + \
            principal_loss_bce + joint_loss_8_bce + joint_loss_16_bce + l2_reg_loss
        total_loss = tf.identity(total_loss, name='total_loss')

        ret = {
            'total_loss': total_loss,
            'principal_loss': principal_loss_bce + principal_loss_dice,
            'joint_loss_8': joint_loss_8_bce + joint_loss_8_dice,
            'joint_loss_16': joint_loss_16_bce + joint_loss_16_dice,
            'l2_loss': l2_reg_loss,
        }

        return ret

    def compute_loss(self, input_tensor, label_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param label_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            logits, context_8_logits, context_16_logits = self._build_net(
                input_tensor=input_tensor, phase=self._phase
            )
            # ignore label index
            logits = tf.reshape(logits, [-1, self._class_nums])
            context_8_logits = tf.reshape(context_8_logits, [-1, self._class_nums])
            context_16_logits = tf.reshape(context_16_logits, [-1, self._class_nums])
            label_tensor = tf.reshape(label_tensor, [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(label_tensor, self._class_nums - 1)), 1)
            logits = tf.gather(logits, indices)
            context_8_logits = tf.gather(context_8_logits, indices)
            context_16_logits = tf.gather(context_16_logits, indices)
            label_tensor = tf.cast(tf.gather(label_tensor, indices), tf.int32)
            # compute loss according to the loss type
            if self._loss_type.lower() == 'cross_entropy':
                ret = self._compute_cross_entropy_loss(
                    logits=logits,
                    context_8_logits=context_8_logits,
                    context_16_logits=context_16_logits,
                    label_tensor=label_tensor
                )
            elif self._loss_type.lower() == 'dice':
                ret = self._compute_dice_loss(
                    logits=logits,
                    context_8_logits=context_8_logits,
                    context_16_logits=context_16_logits,
                    label_tensor=label_tensor
                )
            else:
                raise ValueError('Not supported loss type: {:s}'.format(self._loss_type))

        return ret

    def inference(self, input_tensor, name, reuse):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            logits = self._build_net(
                input_tensor=input_tensor, phase='test'
            )
            seg_score = tf.nn.softmax(logits=logits)
            seg_prediction = tf.argmax(seg_score, axis=-1, name='prediction')
        return seg_prediction


if __name__ == '__main__':
    """
    test code
    """
    import time

    input_tensor_width, input_tensor_height = CFG.AUG.TRAIN_CROP_SIZE
    test_input_tensor = tf.random.normal(shape=[1, input_tensor_height, input_tensor_width, 3], dtype=tf.float32)
    test_label_tensor = tf.random.normal(shape=[1, input_tensor_height, input_tensor_width], dtype=tf.float32)
    test_label_tensor = tf.cast(test_label_tensor, tf.int32)

    bisenet = BiseNet(phase='train')
    loss = bisenet.compute_loss(
        input_tensor=test_input_tensor,
        label_tensor=test_label_tensor,
        name='BiseNet',
        reuse=False
    )
    prediction = bisenet.inference(
        input_tensor=test_input_tensor,
        name='BiseNet',
        reuse=True
    )

    time_comsuming_loops = 5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # bisenet inference time consuming
        t_start = time.time()
        for loop_index in range(time_comsuming_loops):
            print(sess.run(prediction).shape)
        print('Bisenet inference cost time: {:.5f}s'.format((time.time() - t_start) / time_comsuming_loops))
