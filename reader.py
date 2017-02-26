"""
Code for reading TFRecords data for crowd flow.
"""
import tensorflow as tf

import settings


def read_single_image_and_head_positions_example(file_name_queue):
    """
    Reads a single image and head positions example.

    :param file_name_queue: The queue of the TFRecords file names.
    :type file_name_queue: tf.train.QueueBase
    :return: The image and head positions tensors
    :rtype: (tf.Tensor, tf.Tensor)
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name_queue)
    feature_types = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'head_positions_raw': tf.FixedLenFeature([], tf.string)
    }
    features = tf.parse_single_example(serialized_example, features=feature_types)

    flat_image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(flat_image, settings.image_shape)

    flat_head_positions = tf.decode_raw(features['head_positions_raw'], tf.int32)
    head_positions = tf.reshape(flat_head_positions, [-1, settings.head_positions_width])

    return image, head_positions
