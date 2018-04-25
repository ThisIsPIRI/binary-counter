import tensorflow as tf


def binary_model(features, labels, mode, params):
	data_type = params["data_type"]
	sequence_length = params["sequence_length"]
	num_hidden = params["num_hidden"]
	expected_onehot_t = tf.one_hot(labels, sequence_length + 1)
	cell_t = tf.nn.rnn_cell.GRUCell(num_hidden, bias_initializer=tf.initializers.random_normal)
	# Unroll the cells. The dimension of the output will be (batch_size, string_size, num_hidden)
	rnn_output_t, train_state_t = tf.nn.dynamic_rnn(cell_t, features, dtype=data_type)
	# Extract the last timesteps' outputs. The dimension will be (batch_size, num_hidden)
	transposed_temp = tf.transpose(rnn_output_t, [1, 0, 2])
	last_timestep_t = tf.gather(transposed_temp, int(transposed_temp.get_shape()[0]) - 1)
	# Set up a dense layer to process the final output
	prediction_t = tf.layers.dense(last_timestep_t, sequence_length + 1, activation=tf.nn.relu)
	predicted_classes = tf.argmax(prediction_t, 1)

	# Predict the results
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions={
			"class_ids": predicted_classes[:, tf.newaxis],
			"probabilities": tf.nn.softmax(prediction_t),
			"logits": prediction_t
		})

	# Set up the error function
	error_t = tf.losses.log_loss(expected_onehot_t, prediction_t)

	# Evaluate the errors
	if mode == tf.estimator.ModeKeys.EVAL:
		accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
		metrics = {"accuracy": accuracy}
		# Register the accuracy to Tensorboard
		tf.summary.scalar("accuracy", accuracy[1])
		return tf.estimator.EstimatorSpec(mode, loss=error_t, eval_metric_ops=metrics)

	# Set up the optimizer
	minimizer_t = tf.train.AdamOptimizer().minimize(error_t, global_step=tf.train.get_global_step())

	#Train the model
	if mode == tf.estimator.ModeKeys.TRAIN:
		return tf.estimator.EstimatorSpec(mode, loss=error_t, train_op=minimizer_t)
