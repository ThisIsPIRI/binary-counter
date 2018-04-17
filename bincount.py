import matplotlib.pyplot as plt

import tensorflow as tf

from binaryUtil import toArray, onehotToIndices


class BinaryTensors:
	def __init__(self, input_t, expected_t, prediction_t, error_t, minimizer_t):
		self.input_t = input_t
		self.expected_t = expected_t
		self.prediction_t = prediction_t
		self.error_t = error_t
		self.minimizer_t = minimizer_t

	def get(self):
		return self.input_t, self.expected_t, self.prediction_t, self.error_t, self.minimizer_t
	input_t = expected_t = prediction_t = error_t = minimizer_t = None

class BinaryDatasets:
	def __init__(self, input_d, expected_d):
		self.input_d = input_d
		self.expected_d = expected_d

	def get(self):
		return self.input_d, self.expected_d


class BinaryCounter:
	SAVE_DIR = "/tfData"

	def buildRnn(self, sequence_length, string_size, num_hidden=16, data_type=tf.float32):
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

	def train(self, train_ds, test_ds, tensors, epochs=101, batch_size=-1, load_model=True):
		input_t, expected_t, prediction_t, error_t, minimizer_t = tensors.get()
		saver = tf.train.Saver()
		with tf.Session() as sess:
			if load_model:
				saver.restore(sess, self.SAVE_DIR + "/rnnTestModel.ckpt")
			else:
				sess.run(tf.global_variables_initializer())
			errors = []
			# https://stackoverflow.com/a/33050617
			plt.ion()
			plt.show()
			feed = {input_t: train_ds.input_d, expected_t: train_ds.expected_d}
			for epoch in range(epochs):
				errors.append(sess.run(error_t, feed))
				sess.run(minimizer_t, feed)
				# Verify
				if epoch % 50 == 0:
					print(errors)
					plt.plot(errors)
					plt.draw()
					plt.pause(0.001)
			print("training complete")
			print("prediction: ")
			predicted = onehotToIndices(sess.run(prediction_t, feed)[0])
			print(f"{predicted[0][0]} ones, {predicted[0][1] * 100}% sure")
			print(f"{predicted[1][0]} ones, {predicted[1][1] * 100}% sure")
			print("expected: ")
			print(train_ds.expected_d[0])
			tf.summary.FileWriter(self.SAVE_DIR).add_graph(sess.graph)
			saver.save(sess, self.SAVE_DIR + '/' + "rnnTestModel.ckpt")
		plt.ioff()
		input("Complete examining the figure?")  # Let the user interact with the figure
		plt.close()
