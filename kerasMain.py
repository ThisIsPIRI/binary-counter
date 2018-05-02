import numpy as np
from random import shuffle
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import load_model, Sequential
from tensorflow.python.keras.layers import Activation, Dense, Dropout, GRU

from binaryUtil import toArray, toList


def main():
	DEBUG_MODE = True
	SAVE_DIR = "D:/tfData/"
	SAVE_FILE = SAVE_DIR + "kerasModel.hdf5"
	feature_num = 3
	sequence_length = 20
	train_batch_size = test_batch_size = 2 ** sequence_length // 10000 if DEBUG_MODE else 2 ** sequence_length // 3

	all_possible = None
	if input("Generate datasets?(y/n): ") == 'y':
		all_possible = [f"{i:020b}" for i in range(2 ** sequence_length)]  # Generate all 20-char long sequences of 0s and 1s
		shuffle(all_possible)
		print("all possibilities generated")

		# Input data dimensions : [batch size, sequence_length, input_dim]
		train_input_d = np.array([toList(i) for i in all_possible[:train_batch_size]])  # Take train_batch_size of that randomly as training data
		train_expected_d = [int(sum([sum(x) for x in y])) for y in train_input_d]
		train_expected_onehot_d = np.eye(sequence_length + 1)[train_expected_d]

		test_input_d = np.array([toList(i) for i in all_possible[train_batch_size:train_batch_size + test_batch_size]])  # Take test_batch_size of that again for the test set
		test_expected_d = [int(sum([sum(x) for x in y])) for y in test_input_d]
		test_expected_onehot_d = np.eye(sequence_length + 1)[test_expected_d]
		print("datasets organized")

	if input("Load the model?(y/n): ") == 'y':
		model = load_model(SAVE_FILE) # TODO: resolve "You are trying to load a weight file containing 2 layers into a model with 0 layers."
	else:
		model = Sequential([
			GRU(16),
			Dropout(0.4),
			Dense(sequence_length + 1, input_shape=(feature_num,)), # (output size, input shape)
			Activation("relu"),
		])
		model.compile(optimizer="adagrad", loss="categorical_crossentropy", metrics=["accuracy"])
	board = TensorBoard(log_dir=SAVE_DIR, write_graph=True)

	while True:
		print("Train the model(t), predict with it(p) or exit(e).")
		what = input("Choice: ")
		if what == 't':
			if all_possible is None:
				print("Datasets not generated")
			else:
				model.fit(train_input_d, train_expected_onehot_d, epochs=20, batch_size=train_batch_size, callbacks=[board])
				print("eval accuracy: ")
				print(model.evaluate(test_input_d, test_expected_onehot_d))
				model.save(SAVE_FILE)
		elif what == 'p':
			to_p = input("data: ")
			orig_data = np.array(toArray(to_p))
			orig_ground = np.eye(sequence_length + 1)[sum([int(x) for x in to_p])]
			print(model.predict(orig_data, batch_size=1))
		elif what == 'e':
			break



if __name__ == "__main__":
	main()
