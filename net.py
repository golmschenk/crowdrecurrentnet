"""
Code for the recurrent neural network to estimate crowd density.
"""
import datetime
import tensorflow as tf

from example_generator import ExampleGenerator

file_name_queue = tf.train.string_input_producer(['data/lcrowdv_micro.tfrecords'])
inputs = [ExampleGenerator(file_name_queue).outputs() for _ in range(2)]
batch_size = 10
min_after_dequeue = 300
capacity = min_after_dequeue + (3 * batch_size)
image_sequences_tensor, head_counts_tensor = tf.train.shuffle_batch_join(inputs, 10, capacity=capacity,
                                                                         min_after_dequeue=min_after_dequeue,
                                                                         enqueue_many=True)

supervisor = tf.train.Supervisor(logdir='logs/{}'.format(datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S")))
with supervisor.managed_session() as session:
    for step in range(10):
        if supervisor.should_stop():
            break
        image_sequences, head_counts = session.run([image_sequences_tensor, head_counts_tensor])
        print()
