"""
A simple script to reformat the LCrowdV data TFRecords format.
"""
import json
import os
import tensorflow as tf
from PIL import Image
import numpy as np

import settings


def _bytes_feature(value):
    """
    Converts a value to a bytes feature.

    :param value: The value to convert.
    :type value: str
    :return: The bytes feature.
    :rtype: tf.train.Feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def data_directory_to_tfrecords(data_directory, output_directory):
    """
    Converts a LCrowdV directory to a TFRecords file.

    :param data_directory: The location of the LCrowdV directory.
    :type data_directory: str
    :param output_directory: The directory to output the TFRecords file to.
    :type output_directory: str
    """
    tfrecords_name = os.path.basename(data_directory)
    tfrecords_path = os.path.join(output_directory, '{}.tfrecords'.format(tfrecords_name))
    images_directory = os.path.join(data_directory, 'png')
    head_positions_directory = os.path.join(data_directory, 'csv')
    meta_file_written = False
    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for index, file_name in enumerate([name for name in os.listdir(images_directory) if name.endswith('.png')]):
            image_file_path = os.path.join(images_directory, file_name)
            pil_image = Image.open(image_file_path)
            pil_image.load()
            image = np.asarray(pil_image, dtype=np.uint8)[:, :, :3]
            csv_file_path = os.path.join(head_positions_directory, '{}.csv'.format(file_name[:-4]))
            head_positions = np.full(shape=(settings.max_head_count, settings.head_positions_width), fill_value=-1,
                                     dtype=np.int32)
            if os.path.getsize(csv_file_path) > 0:
                csv_data = np.genfromtxt(csv_file_path, delimiter=',', dtype=np.int32)
                if csv_data.ndim < 2:
                    csv_data = np.expand_dims(csv_data, axis=0)
                head_positions[0:csv_data.shape[0]] = csv_data[:, :3]
            image_raw = image.tostring()
            head_positions_raw = head_positions.tostring()
            features = {'image_raw': _bytes_feature(image_raw),
                        'head_positions_raw': _bytes_feature(str(head_positions_raw))}
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            if not meta_file_written:
                meta_dictionary = {'image_shape': image.shape, 'max_head_count': settings.max_head_count}
                with open(os.path.join(output_directory, '{}_meta.json'.format(tfrecords_name)), 'w') as meta_json_file:
                    json.dump(meta_dictionary, meta_json_file, indent=2)
                meta_file_written = True
            print('\r{} frames written.'.format(index + 1), end='')


if __name__ == '__main__':
    data_directory_to_tfrecords('data/lcrowdv_micro', 'data/lcrowdv_micro')
