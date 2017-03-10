"""
Code for the recurrent neural network to estimate crowd density.
"""
import datetime
import os

import tensorflow as tf

import settings
from example_generator import ExampleGenerator
from settings import min_after_dequeue


def train():
    file_name_queue = tf.train.string_input_producer(['data/lcrowdv_micro.tfrecords'])
    examples_join = [ExampleGenerator(file_name_queue).outputs() for _ in range(2)]
    capacity = min_after_dequeue + (3 * settings.batch_size)
    image_sequences_tensor, head_counts_tensor = tf.train.shuffle_batch_join(examples_join, settings.batch_size,
                                                                             capacity=capacity,
                                                                             min_after_dequeue=min_after_dequeue,
                                                                             enqueue_many=True)

    net = tf.layers.conv2d(image_sequences_tensor[:, 9], 8, [5, 5])
    net = tf.layers.conv2d(net, 16, [5, 5])
    net = tf.layers.conv2d(net, 32, [5, 5])
    net = tf.layers.conv2d(net, 64, [18, 18])
    net = tf.layers.conv2d(net, 1, [1, 1])
    net = tf.squeeze(net)
    loss = tf.reduce_mean(tf.square(net - head_counts_tensor))
    tf.summary.scalar('myloss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    training_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name='training_op')

    now_string = datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S")
    supervisor = tf.train.Supervisor(logdir=os.path.join(settings.log_directory, now_string))
    with supervisor.managed_session() as session:
        while True:
            if supervisor.should_stop():
                break
            session.run([training_op])


def test():
    most_recent = os.listdir(settings.log_directory)[-1]
    supervisor = tf.train.Supervisor(logdir=os.path.join(settings.log_directory, most_recent))
    with supervisor.managed_session() as session:
        while True:
            if supervisor.should_stop():
                break
            # session.run([training_op])


def run():
    if settings.run_mode == 'train':
        train()
    elif settings.run_mode == 'test':
        test()
    else:
        raise ValueError('`{}` is not a valid run mode.'.format(settings.run_mode))


if __name__ == '__main__':
    run()
