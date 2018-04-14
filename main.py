import tensorflow as tf
import numpy as np
from random import shuffle
import sys


def toArray(binString):
	# Wrap the resulting ints in an array with length of 1 to make the inputs 3-dimensional.
	return np.array([[int(i)] for i in binString])


# https://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# Naming conventions:
# (name)_d or t for raw data or tensors
def main():
	string_size = 20
	temp_dividend = 20
	train_batch_size = 2 ** string_size // temp_dividend
	test_batch_size = 2 ** string_size // temp_dividend
	sequence_length = 20
	num_hidden = 16
	input_dim = 1
	data_type = tf.float32

	all_possible = ['{0:020b}'.format(i) for i in range(2 ** string_size)] # Generate all 20-char long sequences of 0s and 1s
	shuffle(all_possible)
	print("dataset generated")

	# Input data dimensions : [batch size, sequence_length, input_dim]
	train_input_d = [toArray(i) for i in all_possible[:train_batch_size]] # Take test_batch_size of that randomly as training data
	train_expected_d = [np.sum(i) for i in train_input_d]
	print("train set generated")
	# No validation set.
	test_input_d = [toArray(i) for i in all_possible[train_batch_size:train_batch_size + test_batch_size]] # Take test_batch_size of that again for the test set
	test_expected_d = [np.sum(i) for i in test_input_d]
	print("test set generated")

	# Build the model
	# Train set
	train_input_t = tf.placeholder(tf.float32, (None, sequence_length, 1))
	train_expected_t = tf.one_hot(train_expected_d, sequence_length + 1) # TODO: separate data insertion from model building
	cell_t = tf.nn.rnn_cell.LSTMCell(num_hidden)
	# Unroll the cells. The dimension of the output will be (batch_size, string_size, num_hidden)
	train_output_t, train_state_t = tf.nn.dynamic_rnn(cell_t, train_input_t, dtype=data_type)
	# Extract the last timesteps' outputs. The dimension will be (batch_size, num_hidden)
	transposed_temp = tf.transpose(train_output_t, [1, 0, 2])
	train_last_timesteps_t = tf.gather(transposed_temp, int(transposed_temp.get_shape()[0]) - 1)

	# Test set
	test_input_t = tf.placeholder(tf.float32, (None, sequence_length, 1))
	test_expected_t = tf.one_hot(test_expected_d, sequence_length + 1)

	# Train the model
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# When predicting, the result is an ndarray of dimension (batch_size, string_size, num_hidden). You can pass it to a perceptron for final processing.
		# The axes represent output from each dataset, each timestep(after each character in this case) and each neuron.
		# Verify it by result = sess.run(output_t, feed_dict=data)ing and print(f"0 : {len(result)}, 1 : {len(result[0])}, 2 : {len(result[0][0])}")ing.
		data = {train_input_t: train_input_d, test_input_t: test_input_d}
		result = sess.run(train_last_timesteps_t, data)
		print(f"0 : {len(result)}, 1 : {len(result[0])}")
		# Export to Tensorboard
		tf.summary.FileWriter("/board").add_graph(sess.graph)


if __name__ == "__main__":
	main()
