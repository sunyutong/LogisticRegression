'''
This is a class for implementing logistic regression algorithm.
In this algorithm, it includes gradient descent function with
momentum parameter to reduce the number of iterations for optimizing
the algorithm.

'''

import numpy as np

class LogisticRegression:

	# Define a constructor with 3 parameters
	def __init__(self):

		self.W_without_w0 = None
		self.w0 = None
		self._W = None

	# Implement sigmoid function
	def _sigmoid(self, t):
		return 1. / (1. + np.exp(-t))


	def fit(self, X_train, y_train, learning_rate = 0.01, momentum = 0.1 , n_iters=1e4):

		assert X_train.shape[0] == y_train.shape[0], \
			"the size of X_train must be equal to the size of y_train"

		# Define a loss function and return the result
		def cost_function(W, X_b, y):

			y_hat = self._sigmoid(X_b.dot(W))
			try:
				return - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
			except:
				return float('error')


		# Calculate the gradient of loss function
		def get_gradient(W, X_b, y):
			return X_b.T.dot(self._sigmoid(X_b.dot(W)) - y)


		# Implement gradient descent with learning_rate and momentum parameter
		def gradient_descent(X_b, y, initial_W, learning_rate, momentum, n_iters=1e4, epsilon=1e-8):

			W = initial_W
			cur_iter = 0
			v = 0
			# W_history.append(initial_W)

			while cur_iter < n_iters:

				gradient = get_gradient(W, X_b, y)
				last_W = W 
				v = (-1) * learning_rate * gradient + momentum * v
				W += v
				# W_history.append(W)
				if (abs(cost_function(W, X_b, y) - cost_function(last_W, X_b, y))) < epsilon:
					break
				cur_iter += 1

			return W

		X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
		initial_W = np.zeros(X_b.shape[1])

		self._W = gradient_descent(X_b, y_train, initial_W, learning_rate, n_iters)
		self.w0 = self._W[0]
		self.W_without_w0 = self._W[1:]

		return self


	# For X vector, add a column and return predict probability by using sigmoid function
	def predict_proba(self, X_predict):

		X_b = np.hstack([np.ones((len(X_predict),1)), X_predict])
		return self._sigmoid(X_b.dot(self._W))


	# Given a dataset X_predict to be predicted, return a result vector
	def predict(self, X_predict):

		assert self.w0 is not None and self.W_without_w0 is not None, \
			"must fit before predict!"
		assert X_predict.shape[1] == len(self.W_without_w0), \
			"the feature number of X_predict must be equal to X_train"

		proba = self.predict_proba(X_predict)
		return np.array(proba >= 0.5, dtype='int')


	# Get the accuracy of the current model based on the test datasets X_test and y_test
	def score(self, X_test, y_test):

		y_predict = self.predict(X_test)
		return np.sum(y_test == y_predict) / len(y_test)


	def __repr__(self):
		return "LogisticRegression()"