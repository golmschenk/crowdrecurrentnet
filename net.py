"""
Code for the recurrent neural network to estimate crowd density.
"""
import datetime
import os
import shutil

import tensorflow as tf

import settings
from example_generator import ExampleGenerator
from settings import min_after_dequeue


def train():
    """
    Run the network training.
    """
    file_name_queue = tf.train.string_input_producer(['data/lcrowdv_micro.tfrecords'])
    with tf.variable_scope('Example_Generators'):
        examples_join = [ExampleGenerator(file_name_queue).outputs() for _ in range(2)]
    capacity = min_after_dequeue + (3 * settings.batch_size)
    image_sequences_tensor, head_counts_tensor = tf.train.shuffle_batch_join(examples_join, settings.batch_size,
                                                                             capacity=capacity,
                                                                             min_after_dequeue=min_after_dequeue,
                                                                             enqueue_many=True)

    predicted = basic_convolution_inference(image_sequences_tensor)
    loss = tf.reduce_mean(tf.square(predicted - head_counts_tensor))
    tf.summary.scalar('loss', loss)
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
    """
    Runs a network for testing.
    """
    if os.path.exists(os.path.join(settings.log_directory, 'test')):
        shutil.rmtree(os.path.join(settings.log_directory, 'test'))
    most_recent = os.listdir(settings.log_directory)[-1]
    checkpoint_path = tf.train.latest_checkpoint(os.path.join(settings.log_directory, most_recent))
    images = tf.constant(0, shape=[10, 30, 30, 3])
    labels = tf.constant(1, shape=[10])
    saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_path), input_map={'images:0': images,
                                                                                     'labels:0': labels}, name='import')
    def load_model(session_):
        saver.restore(session_, checkpoint_path)
    supervisor = tf.train.Supervisor(logdir=os.path.join(settings.log_directory, 'test'), init_fn=load_model)
    with supervisor.managed_session() as session:
        training_op = session.graph.get_operation_by_name('input_producer')
        loss_tensor = session.graph.get_tensor_by_name('Mean:0')
        training_op = session.graph.get_operation_by_name('training_op')
        for index in range(100):
            _, loss = session.run([training_op, loss_tensor])
            print(loss)


    # most_recent = os.listdir(settings.log_directory)[-1]
    # supervisor = tf.train.Supervisor(logdir=os.path.join(settings.log_directory, most_recent))
    # with supervisor.managed_session() as session:
    #     while True:
    #         if supervisor.should_stop():
    #             break
    #         training_op = session.graph.get_operation_by_name('input_producer')
    #         session.run([training_op])


def basic_convolution_inference(image_sequences_tensor):
    """
    A basic convolutional network. Only uses the last image in the sequence.
    :param image_sequences_tensor: The image sequence tensor.
    :type image_sequences_tensor: tf.Tensor
    :return: The predicted labels.
    :rtype: tf.Tensor
    """
    net = tf.layers.conv2d(image_sequences_tensor[:, 9], 8, [5, 5])
    net = tf.layers.conv2d(net, 16, [5, 5])
    net = tf.layers.conv2d(net, 32, [5, 5])
    net = tf.layers.conv2d(net, 64, [18, 18])
    net = tf.layers.conv2d(net, 1, [1, 1])
    net = tf.squeeze(net)
    return net


def run():
    if settings.run_mode == 'train':
        train()
    elif settings.run_mode == 'test':
        test()
    else:
        raise ValueError('`{}` is not a valid run mode.'.format(settings.run_mode))


if __name__ == '__main__':
    run()
