"""
Code for the recurrent neural network to estimate crowd flow.
"""
import tensorflow as tf

import reader
import settings


class SequenceGenerator:
    """
    A class for generating example sequences and adding them to the queue.
    """

    def __init__(self, file_name_queue):
        """
        :param file_name_queue: The file name queue of the data containing the images and head positions.
        :type file_name_queue: tf.Queue
        """
        self.file_name_queue = file_name_queue
        image_sequence_shape = [settings.sequence_length, *settings.image_shape]
        head_positions_shape = [settings.max_head_count, settings.head_positions_width]
        self.image_sequence = tf.Variable(initial_value=tf.zeros(shape=image_sequence_shape, dtype=tf.uint8),
                                          trainable=False)
        self.head_positions = tf.Variable(initial_value=tf.constant(-1, shape=head_positions_shape, dtype=tf.int32),
                                          trainable=False)

    def step_to_next_frame(self):
        """
        Returns an op that shifts the sequence a frame in the video. This moves all frames over 1 in the sequence and
        removes the oldest frame.

        :return: The op to continue the sequence.
        :rtype: tf.Op
        """
        image_tensor, head_positions = reader.read_single_image_and_head_positions_example(self.file_name_queue)
        expanded_image_tensor = tf.expand_dims(image_tensor, axis=0)
        continue_image_sequence = tf.assign(self.image_sequence,
                                            tf.concat([self.image_sequence[1:], expanded_image_tensor], axis=0))
        continue_head_sequence = tf.assign(self.head_positions, head_positions)
        with tf.control_dependencies([continue_image_sequence, continue_head_sequence]):
            return tf.no_op(name='step_to_next_frame')

    def generate_examples_from_current_sequence(self):
        """
        Generate a set of examples from the current sequence data. Cuts out a random spatial patch of the sequence and
        the corresponding head count in that patch for the final frame.

        :return: A list of examples of tuples with image patch sequence and head count for final frame.
        :rtype: list[(tf.Tensor, tf.Tensor)]
        """
        counter = tf.constant(0, dtype=tf.int32)
        image_sequence_patch_list = tf.zeros(
            [0, settings.sequence_length, *settings.patch_shape,
             settings.image_shape[2]], dtype=tf.uint8
        )
        head_count_list = tf.zeros([0], dtype=tf.int32)

        # noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
        def condition(counter_, image_sequence_patch_list_, head_count_list_):
            return tf.less(counter_, settings.examples_to_generate_per_sequence)

        # noinspection PyMissingOrEmptyDocstring
        def body(counter_, image_sequence_patch_list_, head_count_list_):
            y = tf.random_uniform(shape=[], minval=0, maxval=settings.image_shape[0] - settings.patch_shape[0],
                                  dtype=tf.int32)
            x = tf.random_uniform(shape=[], minval=0, maxval=settings.image_shape[1] - settings.patch_shape[1],
                                  dtype=tf.int32)
            image_sequence_patch = self.image_sequence[:, y:y + settings.patch_shape[0], x:x + settings.patch_shape[1],
                                                       :]
            match = tf.logical_and(tf.greater_equal(self.head_positions[:, 1:], [x, y]),
                                   tf.less(self.head_positions[:, 1:], tf.add([x, y], settings.patch_shape)))
            head_count_in_patch = tf.reduce_sum(tf.cast(tf.logical_and(match[:, 0], match[:, 1]), dtype=tf.int32))
            image_sequence_patch_list_ = tf.concat(
                [image_sequence_patch_list_, tf.expand_dims(image_sequence_patch, axis=0)], axis=0)
            head_count_list_ = tf.concat([head_count_list_, tf.expand_dims(head_count_in_patch, axis=0)], axis=0)
            return tf.add(counter_, 1), image_sequence_patch_list_, head_count_list_

        shape_invariants = [counter.get_shape(), tf.TensorShape([None, *image_sequence_patch_list.get_shape()[1:]]),
                            tf.TensorShape([None])]
        test = tf.while_loop(condition, body, [counter, image_sequence_patch_list, head_count_list], back_prop=False,
                             shape_invariants=shape_invariants)
        return test[1:]
