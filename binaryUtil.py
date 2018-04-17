import numpy as np


def toArray(binString):
	# Wrap the resulting ints in an array with length of 1 to make the inputs 3-dimensional.
	return np.array([[int(i)] for i in binString])


def onehotToIndices(arr):
	firstI = secondI = -1
	firstVal = secondVal = -99999
	for i, val in enumerate(arr):
		if val > firstVal:
			secondVal = firstVal
			secondI = firstI
			firstVal= val
			firstI = i
		elif val > secondVal:
			secondVal = val
			secondI = i
	return (firstI, firstVal), (secondI, secondVal)