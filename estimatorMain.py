from random import shuffle
import tensorflow as tf

from binestimator import binary_model
from binaryUtil import printEstimatorPred, toList


def main():
	DEBUG_MODE = True
	temp_dividend = 10000 if DEBUG_MODE else 20
	sequence_length = 20
	train_batch_size = 2 ** sequence_length // temp_dividend
	test_batch_size = 2 ** sequence_length // temp_dividend
	num_hidden = 16
	save_dir = "/tfData/"

	all_possible = ['{0:020b}'.format(i) for i in range(2 ** sequence_length)]  # Generate all 20-char long sequences of 0s and 1s
	shuffle(all_possible)
	print("all possibilities generated")

	# Input data dimensions : [batch size, sequence_length, input_dim]
	train_input_d = [toList(i) for i in all_possible[:train_batch_size]]  # Take train_batch_size of that randomly as training data
	train_expected_d = [int(sum([sum(x) for x in y])) for y in train_input_d]
	# No validation set.
	test_input_d = [toList(i) for i in all_possible[train_batch_size:train_batch_size + test_batch_size]]  # Take test_batch_size of that again for the test set
	test_expected_d = [int(sum([sum(x) for x in y])) for y in test_input_d]
	print("datasets organized")

	def input_fn(features, labels, mode, batch_size=32):
		dataset = tf.data.Dataset.from_tensor_slices((features, labels))
		if mode == tf.estimator.ModeKeys.TRAIN:
			return dataset.shuffle(1000).repeat().batch(batch_size).make_one_shot_iterator().get_next()
		else:
			return dataset.batch(batch_size).make_one_shot_iterator().get_next()

	counter = tf.estimator.Estimator(model_fn=binary_model, model_dir=save_dir, params={
		"data_type": tf.float32,
		"sequence_length": sequence_length,
		"num_hidden": num_hidden})
	counter.train(input_fn=lambda: input_fn(train_input_d, train_expected_d, tf.estimator.ModeKeys.TRAIN), steps=100)
	results = list(counter.predict(input_fn=lambda: input_fn(test_input_d, test_expected_d, tf.estimator.ModeKeys.PREDICT)))

	for i, r in enumerate(results):
		printEstimatorPred(r, test_input_d[i], test_expected_d[i])

	while True:
		print("p: predict, e: exit")
		what = input("choice: ")
		if what == 'p':
			toP = input("data: ")
			origData = toList(toP)
			origGround = sum([int(x) for x in toP])
			printEstimatorPred(list(counter.predict(input_fn=lambda: input_fn([origData], [origGround], tf.estimator.ModeKeys.PREDICT)))[0], origData, origGround)
		elif what == 'e':
			break


if __name__ == "__main__":
	main()
