'''
This is a class which defines a gradient descent function with a momentum parameter.
We assume that a loss function is a quadratic function which is (x-2.5)**2-1 and calculate
the derivative of this function. A quadratic function is very intuitive for drawing
gradient descent images.

'''

class GradientDescentM:
	# Define a constructor with 3 parameters
	# W_history list records every steps of gradient descent
	# W_number records number of iterations
	# W vector records the best value to minimize the loss function
	def __init__(self):

		self.W_history = []
		self.W_number = None
		self.W = None

	# An assumed loss function 
	def J(self, x):
		return (x-2.5)**2-1

	# Derivative of this loss function 
	def dJ(self,x):
		return 2*(x-2.5)

	# Gradient descent function with learning_rate and momentum parameters.
	def gradient_descent(self, initial_W, learning_rate, momentum, epsilon = 1e-8, n_iter = 1e4, ):
		self.W_history = []
		self.W = initial_W
		self.W_history.append(initial_W)
		cur_iter = 0
		v = 0

		while cur_iter < n_iter:

			gradient = self.dJ(self.W)
			last_W = self.W
			v =  learning_rate * gradient + momentum * v
			self.W -= v
			self.W_history.append(self.W)

			if(abs(self.J(self.W) - self.J(last_W))) < epsilon:
				break
			cur_iter += 1

		return self

