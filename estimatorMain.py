import matplotlib.pyplot as plt
from random import shuffle
import tensorflow as tf

from binEstimator import binaryModel
from binaryUtil import printEstimatorPred, show, toList


def main():
	DEBUG_MODE = False
	temp_dividend = 20000 if DEBUG_MODE else 3
	sequence_length = 20
	train_batch_size = 2 ** sequence_length // temp_dividend
	test_batch_size = 2 ** sequence_length // temp_dividend
	num_hidden = 16
	save_dir = "/tfData/"

	all_possible = None
	if input("Generate datasets?(y/n): ") == 'y':
		all_possible = [f"{i:020b}" for i in range(2 ** sequence_length)]  # Generate all 20-char long sequences of 0s and 1s
		shuffle(all_possible)
		print("all possibilities generated")

		# Input data dimensions : [batch size, sequence_length, input_dim]
		train_input_d = [toList(i) for i in all_possible[:train_batch_size]]  # Take train_batch_size of that randomly as training data
		train_expected_d = [int(sum([sum(x) for x in y])) for y in train_input_d]

		test_input_d = [toList(i) for i in all_possible[train_batch_size:train_batch_size + test_batch_size]]  # Take test_batch_size of that again for the test set
		test_expected_d = [int(sum([sum(x) for x in y])) for y in test_input_d]
		print("datasets organized")

	def train_input_fn(features, labels, batch_size=32):
		dataset = tf.data.Dataset.from_tensor_slices((features, labels))
		dataset = dataset.shuffle(len(features)).repeat().batch(batch_size)
		return dataset.make_one_shot_iterator().get_next()

	def test_input_fn(features, labels):
		dataset = tf.data.Dataset.from_tensor_slices((features, labels))
		return dataset.batch(len(features)).make_one_shot_iterator().get_next()

	counter = tf.estimator.Estimator(model_fn=binaryModel, model_dir=save_dir, params={
		"data_type": tf.float32,
		"sequence_length": sequence_length,
		"num_hidden": num_hidden})

	while True:
		print("Train the model(t), predict with it(p) or exit(e).")
		what = input("Choice: ")
		if what == 't':
			if all_possible is None:
				print("Datasets not generated")
				continue
			hundred = int(input("For how many hundreds of steps?: "))
			plt.ion()
			plt.show()
			accuracies = []
			# counter.train(input_fn=lambda: train_input_fn(train_input_d, train_expected_d), steps=hundred * 100)
			"""for i in range(hundred * 2):
				counter.train(input_fn=lambda: train_input_fn(train_input_d, train_expected_d), steps=50)
				accuracies.append(counter.evaluate(input_fn=lambda: test_input_fn(test_input_d, test_expected_d))["accuracy"])
				show(accuracies, 1)
				print(f"{(i + 1) * 50}th step finished")"""
			tf.estimator.train_and_evaluate(counter,
				tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_input_d, train_expected_d), max_steps=hundred * 100),
				tf.estimator.EvalSpec(input_fn=lambda: test_input_fn(test_input_d, test_expected_d)))
		elif what == 'p':
			to_p = input("data: ")
			orig_data = toList(to_p)
			orig_ground = sum([int(x) for x in to_p])
			printEstimatorPred(list(counter.predict(input_fn=lambda: test_input_fn([orig_data], [orig_ground])))[0], orig_data, orig_ground)
		elif what == 'e':
			break


if __name__ == "__main__":
	main()
