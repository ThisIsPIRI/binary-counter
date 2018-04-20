import tensorflow as tf

from binestimator import binary_model


def main():
	string_size = 20
	num_hidden = 16
	counter = tf.estimator.Estimator(model_fn=binary_model, params={
		"data_type": tf.float32,
		"sequence_length": string_size,
		"num_hidden": num_hidden})


if __name__ == "__main__":
	main()