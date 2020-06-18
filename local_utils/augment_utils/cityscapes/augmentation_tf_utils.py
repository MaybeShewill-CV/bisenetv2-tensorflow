#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 下午4:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/bisenetv2-tensorflow
# @File    : augmentation_tf_utils.py
# @IDE: PyCharm
"""
Tensorflow version image data augmentation tools
"""
import tensorflow as tf
import numpy as np
import loguru

from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.cityscapes_cfg_v2
LOG = loguru.logger


def decode(serialized_example):
    """
    Parses an image and label from the given `serialized_example`
    :param serialized_example:
    :return:
    """
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'gt_src_image_raw': tf.FixedLenFeature([], tf.string),
            'gt_label_image_raw': tf.FixedLenFeature([], tf.string),
        })

    # decode gt image
    gt_image = tf.image.decode_png(features['gt_src_image_raw'], channels=3)
    gt_image = tf.reshape(gt_image, shape=[CFG.AUG.TRAIN_CROP_SIZE[1], CFG.AUG.TRAIN_CROP_SIZE[0], 3])

    # decode gt binary image
    gt_binary_image = tf.image.decode_png(features['gt_label_image_raw'], channels=1)
    gt_binary_image = tf.reshape(gt_binary_image, shape=[CFG.AUG.TRAIN_CROP_SIZE[1], CFG.AUG.TRAIN_CROP_SIZE[0], 1])

    return gt_image, gt_binary_image


def resize(img, grt=None, mode='train', align_corners=True):
    """
    改变图像及标签图像尺寸
    AUG.AUG_METHOD为unpadding，所有模式均直接resize到AUG.FIX_RESIZE_SIZE的尺寸
    AUG.AUG_METHOD为stepscaling, 按比例resize，训练时比例范围AUG.MIN_SCALE_FACTOR到AUG.MAX_SCALE_FACTOR,
    间隔为AUG.SCALE_STEP_SIZE，其他模式返回原图
    AUG.AUG_METHOD为rangescaling，长边对齐，短边按比例变化，训练时长边对齐范围AUG.MIN_RESIZE_VALUE
    到AUG.MAX_RESIZE_VALUE，其他模式长边对齐AUG.INF_RESIZE_VALUE
    Args：
        img(numpy.ndarray): 输入图像
        grt(numpy.ndarray): 标签图像，默认为None
        mode(string): 模式, 默认训练模式
        align_corners: 是否对齐角点
    Returns：
        resize后的图像和标签图
    """
    mode = mode.lower()
    img = tf.expand_dims(img, axis=0)
    grt = tf.expand_dims(grt, axis=0)
    if CFG.AUG.RESIZE_METHOD == 'unpadding':
        target_size = (CFG.AUG.FIX_RESIZE_SIZE[0], CFG.AUG.FIX_RESIZE_SIZE[1])
        img = tf.image.resize_bilinear(images=img, size=target_size, align_corners=align_corners)
        if grt is not None:
            grt = tf.image.resize_nearest_neighbor(images=grt, size=target_size, align_corners=align_corners)
    elif CFG.AUG.RESIZE_METHOD == 'stepscaling':
        if mode == 'train':
            min_scale_factor = CFG.AUG.MIN_SCALE_FACTOR
            max_scale_factor = CFG.AUG.MAX_SCALE_FACTOR
            step_size = CFG.AUG.SCALE_STEP_SIZE
            scale_factor = get_random_scale(
                min_scale_factor, max_scale_factor, step_size
            )
            img, grt = randomly_scale_image_and_label(
                img, grt, scale=scale_factor, align_corners=align_corners
            )
    elif CFG.AUG.RESIZE_METHOD == 'rangescaling':
        min_resize_value = CFG.AUG.MIN_RESIZE_VALUE
        max_resize_value = CFG.AUG.MAX_RESIZE_VALUE
        if mode == 'train':
            if min_resize_value == max_resize_value:
                random_size = min_resize_value
            else:
                random_size = tf.random.uniform(
                    shape=[1], minval=min_resize_value, maxval=max_resize_value, dtype=tf.float32) + 0.5
                random_size = tf.cast(random_size, dtype=tf.int32)
        else:
            random_size = CFG.AUG.INF_RESIZE_VALUE

        value = tf.maximum(img.shape[0], img.shape[1])
        scale = float(random_size) / float(value)
        target_size = (int(img.shape[0] * scale), int(img.shape[1] * scale))
        img = tf.image.resize_bilinear(images=img, size=target_size, align_corners=align_corners)
        if grt is not None:
            grt = tf.image.resize_nearest_neighbor(
                images=grt,
                size=target_size,
                align_corners=align_corners
            )
    else:
        raise Exception("Unexpect data augmention method: {}".format(CFG.AUG.AUG_METHOD))

    return tf.squeeze(img, axis=0), tf.squeeze(grt, axis=0)


def _image_dimensions(image, rank):
    """
    Returns the dimensions of an image tensor.
    :param image:
    :param rank:
    :return:
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), rank)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]


def pad_to_bounding_box(
        image, offset_height, offset_width, target_height,
        target_width, pad_value):
    """

    :param image:
    :param offset_height:
    :param offset_width:
    :param target_height:
    :param target_width:
    :param pad_value:
    :return:
    """
    with tf.name_scope(None, 'pad_to_bounding_box', [image]):
        image = tf.convert_to_tensor(image, name='image')
        original_dtype = image.dtype
        if original_dtype != tf.float32 and original_dtype != tf.float64:
            image = tf.cast(image, tf.int32)
        image_rank_assert = tf.Assert(
            tf.logical_or(
                tf.equal(tf.rank(image), 3),
                tf.equal(tf.rank(image), 4)),
            ['Wrong image tensor rank.'])
        with tf.control_dependencies([image_rank_assert]):
            image -= pad_value
        image_shape = image.get_shape()
        is_batch = True
        if image_shape.ndims == 3:
            is_batch = False
            image = tf.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = tf.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image.get_shape().ndims != 4:
            raise ValueError('Input image must have either 3 or 4 dimensions.')
        _, height, width, _ = _image_dimensions(image, rank=4)
        target_width_assert = tf.Assert(
            tf.greater_equal(
                target_width, width),
            ['target_width must be >= width'])
        target_height_assert = tf.Assert(
            tf.greater_equal(target_height, height),
            ['target_height must be >= height'])
        with tf.control_dependencies([target_width_assert]):
            after_padding_width = target_width - offset_width - width
        with tf.control_dependencies([target_height_assert]):
            after_padding_height = target_height - offset_height - height
        offset_assert = tf.Assert(
            tf.logical_and(
                tf.greater_equal(after_padding_width, 0),
                tf.greater_equal(after_padding_height, 0)),
            ['target size not possible with the given target offsets'])
        batch_params = tf.stack([0, 0])
        height_params = tf.stack([offset_height, after_padding_height])
        width_params = tf.stack([offset_width, after_padding_width])
        channel_params = tf.stack([0, 0])
        with tf.control_dependencies([offset_assert]):
            paddings = tf.stack([batch_params, height_params, width_params,
                               channel_params])
        padded = tf.pad(image, paddings)
        if not is_batch:
            padded = tf.squeeze(padded, axis=[0])
        outputs = padded + pad_value
        if outputs.dtype != original_dtype:
            outputs = tf.cast(outputs, original_dtype)
        return outputs


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """
    在一定范围内得到随机值，范围为min_scale_factor到max_scale_factor，间隔为step_size
    Args：
        min_scale_factor(float): 随机尺度下限，大于0
        max_scale_factor(float): 随机尺度上限，不小于下限值
        step_size(float): 尺度间隔，非负, 等于为0时直接返回min_scale_factor到max_scale_factor范围内任一值
    Returns：
        随机尺度值
    """

    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return tf.cast(min_scale_factor, tf.float32)

        # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random_uniform([1],
                                 minval=min_scale_factor,
                                 maxval=max_scale_factor)

        # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random_shuffle(scale_factors)
    return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image, label=None, scale=1.0, align_corners=True):
    """
    按比例resize图像和标签图, 如果scale为1，返回原图
    Args：
        image(numpy.ndarray): 输入图像
        label(numpy.ndarray): 标签图，默认None
        sclae(float): 图片resize的比例，非负，默认1.0
        align_corners:
    Returns：
        resize后的图像和标签图
    """
    if scale == 1.0:
        return image, label
    image_shape = tf.shape(image)
    new_dim = tf.cast(
        tf.cast([image_shape[1], image_shape[2]], tf.float32) * scale,
        tf.int32)

    image = tf.image.resize_bilinear(image, new_dim, align_corners=True)
    if label is not None:
        label = tf.image.resize_nearest_neighbor(label, new_dim, align_corners=True)
    return image, label


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """

    :param image:
    :param offset_height:
    :param offset_width:
    :param crop_height:
    :param crop_width:
    :return:
    """
    original_shape = tf.shape(image)

    if len(image.get_shape().as_list()) != 3:
        raise ValueError('input must have rank of 3')
    original_channels = image.get_shape().as_list()[2]

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), tf.int32)

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    image = tf.reshape(image, cropped_shape)
    image.set_shape([crop_height, crop_width, original_channels])
    return image


def rand_crop(image_list, crop_height, crop_width):
    """

    :param image_list:
    :param crop_height:
    :param crop_width:
    :return:
    """
    if not image_list:
        raise ValueError('Empty image_list.')

        # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]


def resolve_shape(tensor, rank=None, scope=None):
    """
    resolves the shape of a Tensor.
    :param tensor:
    :param rank:
    :param scope:
    :return:
    """
    with tf.name_scope(scope, 'resolve_shape', [tensor]):
        if rank is not None:
            shape = tensor.get_shape().with_rank(rank).as_list()
        else:
            shape = tensor.get_shape().as_list()

        if None in shape:
            shape_dynamic = tf.shape(tensor)
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = shape_dynamic[i]

    return shape


def random_flip_image(img, grt):
    """

    :param img:
    :param grt:
    :return:
    """
    if CFG.AUG.FLIP:
        if CFG.AUG.FLIP_RATIO <= 0:
            n = 0
        elif CFG.AUG.FLIP_RATIO >= 1:
            n = 1
        else:
            n = int(1.0 / CFG.AUG.FLIP_RATIO)
        if n > 0:
            random_value = tf.random_uniform([])
            is_flipped = tf.less_equal(random_value, 0.5)
            img = tf.cond(is_flipped, true_fn=lambda: img[::-1, :, :], false_fn=lambda: img)
            grt = tf.cond(is_flipped, true_fn=lambda: grt[::-1, :, :], false_fn=lambda: grt)

    return img, grt


def random_mirror_image(img, grt):
    """

    :param img:
    :param grt:
    :return:
    """
    if CFG.AUG.MIRROR:
        random_value = tf.random_uniform([])
        is_mirrored = tf.less_equal(random_value, 0.5)
        img = tf.cond(is_mirrored, true_fn=lambda: img[:, ::-1, :], false_fn=lambda: img)
        grt = tf.cond(is_mirrored, true_fn=lambda: grt[:, ::-1, :], false_fn=lambda: grt)

    return img, grt


def normalize_image(img, grt):
    """

    :param img:
    :param grt:
    :return:
    """
    img = tf.divide(img, tf.constant(255.0))
    img_mean = tf.convert_to_tensor(
        np.array(CFG.DATASET.MEAN_VALUE).reshape((1, 1, len(CFG.DATASET.MEAN_VALUE))),
        dtype=tf.float32
    )
    img_std = tf.convert_to_tensor(
        np.array(CFG.DATASET.STD_VALUE).reshape((1, 1, len(CFG.DATASET.STD_VALUE))),
        dtype=tf.float32
    )
    img -= img_mean
    img /= img_std

    return img, grt


def preprocess_image_for_train(src_image, label_image):
    """

    :param src_image:
    :param label_image:
    :return:
    """
    # resize image
    src_image, label_image = resize(src_image, label_image)
    # random flip
    src_image, label_image = random_flip_image(src_image, label_image)
    # random mirror
    src_image, label_image = random_mirror_image(src_image, label_image)
    # padding images
    image_shape = tf.shape(src_image)
    image_height = image_shape[0]
    image_width = image_shape[1]
    target_height = image_height + tf.maximum(CFG.AUG.TRAIN_CROP_SIZE[1] - image_height, 0)
    target_width = image_width + tf.maximum(CFG.AUG.TRAIN_CROP_SIZE[0] - image_width, 0)

    pad_pixel = tf.reshape(CFG.DATASET.PADDING_VALUE, [1, 1, 3])
    src_image = pad_to_bounding_box(src_image, 0, 0, target_height, target_width, pad_pixel)
    label_image = pad_to_bounding_box(label_image, 0, 0, target_height, target_width, CFG.DATASET.IGNORE_INDEX)
    # random crop
    src_image, label_image = rand_crop(
        image_list=[src_image, label_image],
        crop_height=CFG.AUG.TRAIN_CROP_SIZE[1],
        crop_width=CFG.AUG.TRAIN_CROP_SIZE[0]
    )
    # normalize image
    src_image, label_image = normalize_image(src_image, label_image)
    # downsample input image
    resize_image_size = (int(CFG.AUG.TRAIN_CROP_SIZE[1] / 2), int(CFG.AUG.TRAIN_CROP_SIZE[0] / 2))
    src_image = tf.image.resize_bilinear(
        images=tf.expand_dims(src_image, axis=0),
        size=resize_image_size,
        align_corners=True
    )
    label_image = tf.image.resize_nearest_neighbor(
        images=tf.expand_dims(label_image, axis=0),
        size=resize_image_size,
        align_corners=True
    )

    src_image = tf.squeeze(src_image, axis=0)
    label_image = tf.squeeze(label_image, axis=0)

    return src_image, label_image


def preprocess_image_for_val(src_image, label_image):
    """

    :param src_image:
    :param label_image:
    :return:
    """
    src_image = tf.cast(src_image, tf.float32)
    # normalize image
    src_image, label_image = normalize_image(src_image, label_image)
    # downsample input image
    resize_image_size = (int(CFG.AUG.TRAIN_CROP_SIZE[1] / 2), int(CFG.AUG.TRAIN_CROP_SIZE[0] / 2))
    src_image = tf.image.resize_bilinear(
        images=tf.expand_dims(src_image, axis=0),
        size=resize_image_size,
        align_corners=True
    )
    label_image = tf.image.resize_nearest_neighbor(
        images=tf.expand_dims(label_image, axis=0),
        size=resize_image_size,
        align_corners=True
    )

    src_image = tf.squeeze(src_image, axis=0)
    label_image = tf.squeeze(label_image, axis=0)

    return src_image, label_image


if __name__ == '__main__':
    """
    test code
    """
    source_image_path = '/media/baidu/DataRepo/IMAGE_SCENE_SEGMENTATION/CITYSPACES/' \
                        'gt_images/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'
    source_label_path = '/media/baidu/DataRepo/IMAGE_SCENE_SEGMENTATION/CITYSPACES/' \
                        'gt_annotation/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png'

    source_image = tf.io.read_file(source_image_path)
    source_image = tf.image.decode_png(source_image)
    source_image.set_shape(shape=[1024, 2048, 3])
    source_label_image = tf.io.read_file(source_label_path)
    source_label_image = tf.image.decode_png(source_label_image, channels=0)
    source_label_image.set_shape(shape=[1024, 2048, 1])

    preprocess_src_img, preprocess_label_img = preprocess_image_for_train(
        src_image=source_image,
        label_image=source_label_image
    )

    with tf.Session() as sess:
        while True:
            ret = sess.run([preprocess_src_img])
            print(ret[0].shape)
