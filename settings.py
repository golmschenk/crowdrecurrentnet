"""
Settings for the network.
"""

sequence_length = 10
image_shape = [576, 768, 3]
head_positions_width = 3
max_head_count = 10000  # Large enough for largest count of people. Should only be used on CPU.
examples_to_generate_per_sequence = 20
patch_shape = [30, 30]
batch_size = 10
