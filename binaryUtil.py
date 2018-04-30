import numpy as np
import matplotlib.pyplot as plt


def toArray(binString):
	# Wrap the resulting ints in an array with length of 1 to make the inputs 3-dimensional.
	return np.array([[int(i)] for i in binString])


def toList(binString): # Exists because of https://stackoverflow.com/a/49931506
	return [[float(i)] for i in binString]


def onehotToIndices(arr):
	first_i = second_i = -1
	first_val = second_val = -99999
	for i, val in enumerate(arr):
		if val > first_val:
			second_val = first_val
			second_i = first_i
			first_val= val
			first_i = i
		elif val > second_val:
			second_val = val
			second_i = i
	return (first_i, first_val), (second_i, second_val)


def show(data, in_figure):
	plt.figure(in_figure)
	plt.plot(data)
	plt.draw()
	plt.pause(0.001)


def printEstimatorPred(spec, orig_data, orig_ground):
	most = onehotToIndices(spec["probabilities"])
	print(f"Predictions for {''.join([str(int(x[0])) for x in orig_data])}, the ground truth of which is {orig_ground}:")
	print(f"{most[0][0]} ones, {most[0][1] * 100:.4f}% sure")
	print(f"{most[1][0]} ones, {most[1][1] * 100:.4f}% sure")