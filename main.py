import tensorflow as tf
import numpy as np
from random import shuffle


def toArray(binString):
	return np.array([int(i) for i in binString])


# https://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# Naming conventions:
# (name)_d or t for raw data or tensors
def main():
	train_batch_size = 2 ** 20 // 4
	test_batch_size = 2 ** 20 // 4
	sequence_length = 20
	num_hidden = 16
	data_type = tf.float32

	all_possible = ['{0:020b}'.format(i) for i in range(2 ** 20)] # Generate all 20-char long sequences of 0s and 1s
	shuffle(all_possible)
	print("dataset generated")

	# Input data dimensions : [batch size, sequence_length]
	train_input_d = [toArray(i) for i in all_possible[:train_batch_size]] # Take test_batch_size of that randomly as training data
	train_expected_d = [np.sum(i) for i in train_input_d]
	print("train set generated")
	# No validation set.
	test_input_d = [toArray(i) for i in all_possible[train_batch_size:train_batch_size + test_batch_size]] # Take test_batch_size of that again for the test set
	test_expected_d = [np.sum(i) for i in test_input_d]
	print("test set generated")

	# Build the model
	train_input_t = tf.placeholder(tf.float32, (None, sequence_length))
	train_expected_t = tf.one_hot(train_expected_d, sequence_length + 1)
	test_input_t = tf.placeholder(tf.float32, (None, sequence_length))
	test_expected_t = tf.one_hot(test_expected_d, sequence_length + 1)
	cell_t = tf.nn.rnn_cell.GRUCell(num_hidden)
	output_t, state_t = tf.nn.dynamic_rnn(cell_t, train_input_t, dtype=data_type)


if __name__ == "__main__":
	main()
