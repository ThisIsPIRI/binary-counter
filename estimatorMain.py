import numpy as np
from random import shuffle
import tensorflow as tf

from binestimator import binary_model
from binaryUtil import toArray


def input_fn(features, labels, batch_size):
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))
	dataset = dataset.shuffle(1000).repeat().batch(batch_size)
	return dataset.make_one_shot_iterator().get_next()


def main():
	DEBUG_MODE = True
	temp_dividend = 120 if DEBUG_MODE else 20
	sequence_length = 20
	train_batch_size = 2 ** sequence_length // temp_dividend
	test_batch_size = 2 ** sequence_length // temp_dividend
	num_hidden = 16
	save_dir = "/tfData"

	all_possible = ['{0:020b}'.format(i) for i in range(2 ** sequence_length)]  # Generate all 20-char long sequences of 0s and 1s
	shuffle(all_possible)
	print("all possibilities generated")

	# Input data dimensions : [batch size, sequence_length, input_dim]
	train_input_d = [toArray(i) for i in all_possible[:train_batch_size]]  # Take test_batch_size of that randomly as training data
	train_expected_d = [np.sum(i) for i in train_input_d]
	# No validation set.
	test_input_d = [toArray(i) for i in all_possible[train_batch_size:train_batch_size + test_batch_size]]  # Take test_batch_size of that again for the test set
	test_expected_d = [np.sum(i) for i in test_input_d]
	print("datasets organized")

	counter = tf.estimator.Estimator(model_fn=binary_model, model_dir=save_dir, params={
		"data_type": tf.float32,
		"sequence_length": sequence_length,
		"num_hidden": num_hidden})
	counter.train(input_fn=lambda: input_fn(train_input_d, train_expected_d, 32), steps=100)


if __name__ == "__main__":
	main()