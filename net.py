"""
Code for the recurrent neural network to estimate crowd density.
"""
import datetime
import tensorflow as tf

import settings
from example_generator import ExampleGenerator

file_name_queue = tf.train.string_input_producer(['data/lcrowdv_micro.tfrecords'])  # Move to CPU
examples_join = [ExampleGenerator(file_name_queue).outputs() for _ in range(2)]
min_after_dequeue = 300
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
training_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

supervisor = tf.train.Supervisor(logdir='logs/{}'.format(datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S")))
with supervisor.managed_session() as session:
    for step in range(100000):
        if supervisor.should_stop():
            break
        session.run([training_op])
