import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle
import tensorflow as tf


def toArray(binString):
	# Wrap the resulting ints in an array with length of 1 to make the inputs 3-dimensional.
	return np.array([[int(i)] for i in binString])


def onehotToIndices(arr):
	firstI = secondI = -1
	firstVal = secondVal = -99999
	for i, val in enumerate(arr):
		if val > firstVal:
			secondVal = firstVal
			secondI = firstI
			firstVal= val
			firstI = i
		elif val > secondVal:
			secondVal = val
			secondI = i
	return (firstI, firstVal), (secondI, secondVal)


def buildRnn(sequence_length, string_size, num_hidden=16, data_type=tf.float32):
	input_t = tf.placeholder(data_type, (None, sequence_length, 1))
	expected_t = tf.placeholder(tf.int32, None)
	expected_onehot_t = tf.one_hot(expected_t, sequence_length + 1)
	cell_t = tf.nn.rnn_cell.GRUCell(num_hidden)
	# Unroll the cells. The dimension of the output will be (batch_size, string_size, num_hidden)
	rnn_output_t, train_state_t = tf.nn.dynamic_rnn(cell_t, input_t, dtype=data_type)
	# Extract the last timesteps' outputs. The dimension will be (batch_size, num_hidden)
	transposed_temp = tf.transpose(rnn_output_t, [1, 0, 2])
	last_timestep_t = tf.gather(transposed_temp, int(transposed_temp.get_shape()[0]) - 1)
	# Set up a dense layer to process the final output
	dense_weight_t = tf.Variable(tf.truncated_normal((num_hidden, string_size + 1))) # int(expected_t.get_shape()[1] can be also used instead of string_size + 1
	dense_bias_t = tf.Variable(0.1)
	prediction_t = tf.nn.softmax(tf.matmul(last_timestep_t, dense_bias_t + dense_weight_t))
	# Set up the error function and optimizer
	# clip_by_value is needed to avoid getting inf from negative log.
	error_t = tf.reduce_sum(-expected_onehot_t * tf.log(tf.clip_by_value(prediction_t, 1e-10, 1.0)) - ((1 - expected_onehot_t) * tf.log(1 - (tf.clip_by_value(prediction_t, 1e-10, 1.0)))))
	minimizer_t = tf.train.AdamOptimizer().minimize(error_t)
	return input_t, expected_t, prediction_t, error_t, minimizer_t


# https://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
# Naming conventions:
# (name)_d or t for raw data or tensors
def main():
	string_size = 20
	temp_dividend = 100
	train_batch_size = 2 ** string_size // temp_dividend
	test_batch_size = 2 ** string_size // temp_dividend
	sequence_length = 20
	num_hidden = 16
	input_dim = 1
	data_type = tf.float32
	save_dir = "/tfData"

	all_possible = ['{0:020b}'.format(i) for i in range(2 ** string_size)] # Generate all 20-char long sequences of 0s and 1s
	shuffle(all_possible)
	print("all possibilities generated")

	# Input data dimensions : [batch size, sequence_length, input_dim]
	train_input_d = [toArray(i) for i in all_possible[:train_batch_size]] # Take test_batch_size of that randomly as training data
	train_expected_d = [np.sum(i) for i in train_input_d]
	# No validation set.
	test_input_d = [toArray(i) for i in all_possible[train_batch_size:train_batch_size + test_batch_size]] # Take test_batch_size of that again for the test set
	test_expected_d = [np.sum(i) for i in test_input_d]
	print("datasets organized")

	# Build the model
	train_input_t, train_expected_t, train_prediction_t, train_error_t, train_minimizer_t = buildRnn(sequence_length, string_size)

	# Train the model
	train_feed = {train_input_t: train_input_d, train_expected_t: train_expected_d}
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# TODO: make this stochastic
		errors = []
		# https://stackoverflow.com/a/33050617
		plt.ion()
		plt.show()
		for epoch in range(51):
			errors.append(sess.run(train_error_t, train_feed))
			sess.run(train_minimizer_t, train_feed)
			# Export to Tensorboard
			if epoch % 50 == 0:
				print(errors)
				plt.plot(errors)
				plt.draw()
				plt.pause(0.001)
		print("training complete")
		print("prediction: ")
		predicted = onehotToIndices(sess.run(train_prediction_t, train_feed)[0])
		print(f"{predicted[0][0]} ones, {predicted[0][1] * 100}% sure")
		print(f"{predicted[1][0]} ones, {predicted[1][1] * 100}% sure")
		print("expected: ")
		print(train_expected_d[0])
		tf.summary.FileWriter(save_dir).add_graph(sess.graph)
		saver.save(sess, save_dir + '/' + "rnnTestModel.ckpt")
	plt.ioff()
	input("Complete examining the figure?") # Let the user interact with the figure
	plt.close()

	# Test set
	tf.reset_default_graph()
	test_input_t, test_expected_t, test_prediction_t, test_error_t, _ = buildRnn(sequence_length, string_size)
	print("tensors generated")

	test_feed = {test_input_t: test_input_d, test_expected_t: test_expected_d}

# When predicting, the result is an ndarray of dimension (batch_size, string_size, num_hidden). You can pass it to a dense network for final processing.
# The axes represent output from each dataset, each timestep(after each character in this case) and each neuron.
# Verify it by result = sess.run(output_t, feed_dict=data)ing and print(f"0 : {len(result)}, 1 : {len(result[0])}, 2 : {len(result[0][0])}")ing.


if __name__ == "__main__":
	# print(onehotToIndices([4, 12, 76, 4, 2, 7, 43]))
	main()
