"""
Code for the recurrent neural network to estimate crowd flow.
"""
import tensorflow as tf

import reader
import settings


class SequenceExample():
    def __init__(self):
        image_sequence_shape = [settings.sequence_length, *settings.image_shape]
        head_sequence_shape = [settings.sequence_length, settings.max_head_count, settings.head_positions_width]
        self.image_sequence = tf.Variable(initial_value=tf.zeros(shape=image_sequence_shape, dtype=tf.uint8))
        self.head_positions_sequence = tf.Variable(
            initial_value=tf.constant(-1, shape=head_sequence_shape, dtype=tf.int32)
        )
        self.test = None

    def enqueue_cropped_sequence_example(self, file_name_queue):
        image_tensor, head_positions = reader.read_single_image_and_head_positions_example(file_name_queue)
        self.test = image_tensor
        expanded_image_tensor = tf.expand_dims(image_tensor, axis=0)
        continue_image_sequence = tf.assign(self.image_sequence,
                                            tf.concat([self.image_sequence[1:], expanded_image_tensor], axis=0))
        expanded_head_positions = tf.expand_dims(head_positions, axis=0)
        continue_head_sequence = tf.assign(self.head_positions_sequence,
                                           tf.concat([self.head_positions_sequence[1:], expanded_head_positions],
                                                     axis=0))
        with tf.control_dependencies([continue_image_sequence, continue_head_sequence]):
            return tf.identity(self.image_sequence), tf.identity(self.head_positions_sequence)
