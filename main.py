import numpy as np
from random import shuffle
import tensorflow as tf

from bincount import BinaryCounter, BinaryTensors
from binaryUtil import toArray


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
	counter = BinaryCounter()
	tensors = BinaryTensors(*counter.buildRnn(sequence_length, string_size))

	# Train the model
	while True:
		print("Train the model(t), exit(E)")
		yn = input("Choice:")
		if yn == 't':
			load = False if input("Load models?(y/n): ") == "n" else True
			counter.train(train_input_d, train_expected_d, tensors, load_model=load)
		elif yn == 'E':
			break

# When predicting, the result is an ndarray of dimension (batch_size, string_size, num_hidden). You can pass it to a dense network for final processing.
# The axes represent output from each dataset, each timestep(after each character in this case) and each neuron.
# Verify it by result = sess.run(output_t, feed_dict=data)ing and print(f"0 : {len(result)}, 1 : {len(result[0])}, 2 : {len(result[0][0])}")ing.


if __name__ == "__main__":
	# print(onehotToIndices([4, 12, 76, 4, 2, 7, 43]))
	main()
