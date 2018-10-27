

import numpy as np

class MultiLogisticRegression:

	# Define a constructor with 3 parameters
	def __init__(self):

		self._W = None
		self._classes_num = None
		self._classes_name = None


	# Implement sigmoid function
	def _sigmoid(self, t):
		return 1. / (1. + np.exp(-t))


	def fit(self, X_train, y_train, learning_rate = 0.03, momentum = 0.5 , n_iters=1e4):

		assert X_train.shape[0] == y_train.shape[0], \
			"the size of X_train must be equal to the size of y_train"

		# Define a loss function and return the result
		def cost_function(W, X_b, y):

			y_hat = self._sigmoid(X_b.dot(W))

			try:
				result = (-1)*np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
				return result
			except:
				return float('error')


		# Calculate the gradient of loss function
		def get_gradient(W, X_b, y):
			return X_b.T.dot(self._sigmoid(X_b.dot(W)) - y)


		# Implement gradient descent with learning_rate and momentum parameter
		def gradient_descent(X_b, y, initial_W, learning_rate, momentum , n_iters=1e4, epsilon=1e-8):

			W = initial_W
			cur_iter = 0
			v = 0
			# W_history.append(initial_W)

			while cur_iter < n_iters:

				gradient = get_gradient(W, X_b, y)
				last_W = W 
				v = learning_rate * gradient + 0.1 * v
				W = W - v
				# W_history.append(W)
				if (abs(cost_function(W, X_b, y) - cost_function(last_W, X_b, y)) < epsilon):
					break
				cur_iter += 1

			return W

		X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
		classes = set(np.ravel(y_train))
		classes = list(classes)
		initial_W =[ np.zeros(X_b.shape[1]) for i in range(len(classes))]
		self._W = [ np.zeros(X_b.shape[1]) for i in range(len(classes))]
		self._classes_num = len(classes)
		self._classes_name = classes

		for idx, c in enumerate(classes):
			newY = np.zeros(y_train.shape)
			newY[np.where(y_train == c)] = 1
			g_W = initial_W[idx]
			self._W[idx] = gradient_descent(X_b, newY, g_W, learning_rate, n_iters)


		return self


	# For X vector, add a column and return predict probability by using sigmoid function
	def predict_proba(self, X_predict):

		predict_vector = [ 0 for i in range(self._classes_num)]
		X_b = np.hstack([np.ones((len(X_predict),1)), X_predict])
		y_predict = [ 0 for i in range(X_predict.shape[0])]

		for i in range(self._classes_num):
			predict_vector[i] = self._sigmoid(X_b.dot(self._W[i]))

		for j in range(len(predict_vector[0])):
			max_predict = 0.
			record = -1
			for k in range(self._classes_num):
				if predict_vector[k][j] >= max_predict:
					max_predict = predict_vector[k][j]
					record = k
			y_predict[j] = self._classes_name[record]

		print(predict_vector)
		return y_predict


	# Get the accuracy of the current model based on the test datasets X_test and y_test
	def score(self, X_test, y_test):

		y_predict = self.predict_proba(X_test)
		return np.sum(y_test == y_predict) / len(y_test)


	def __repr__(self):
		return "MultiLogisticRegression()"