import tensorflow as tf


def binary_model(features, labels, mode, params):
	data_type = params["data_type"]
	sequence_length = params["sequence_length"]
	num_hidden = params["num_hidden"]
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
	prediction_t = tf.layers.dense(last_timestep_t, (None, sequence_length, num_hidden), activation=tf.nn.relu)
	# Set up the error function and optimizer
	# clip_by_value is needed to avoid getting inf from negative log.
	error_t = tf.reduce_sum(-expected_onehot_t * tf.log(tf.clip_by_value(prediction_t, 1e-10, 1.0)) - ((1 - expected_onehot_t) * tf.log(1 - (tf.clip_by_value(prediction_t, 1e-10, 1.0)))))
	minimizer_t = tf.train.AdamOptimizer().minimize(error_t)
	return input_t, expected_t, prediction_t, error_t, minimizer_t
