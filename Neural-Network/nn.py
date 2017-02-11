from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		# Seed the random number generator, so it generates the same numbers
		# every time the program runs
		random.seed(1)

		# We model a single neuron, with 3 input connections and 1 output connection.
		# we assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
		# and mean 0
		self.synaptic_weights= 2 * random.random((3,1)) - 1

	#This is our Activation function
	#The sigmoid function, which describes an s shaped curve
	# we pass the weighted sum of the inputs through this function
	# to normalize the output to be between 0 and 1
	def __sigmoid(self, x):
		return 1/(1 + exp(-x))

	#gradient of the sigmoid curve
	def __sigmoid_derivative(self, x):
		return x * (1-x)

	#real meat of our code
	def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
		for inerations in xrange(number_of_iterations):
			#pass the training set through our neural net
			output = self.predict(training_set_inputs)

			#calculate the error between the desired out and predicted output 
			error = training_set_outputs - output

			#we want to minimize this error as we train
			#we do this my iretatively updating our weights
			#multiply the error by the input and again by the graident of the sigmoid curve
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
			#this way let confident weights are adjusted more and inputs that are zero
			#don't cause changes to the weights. This process is calles gradient descent

			#process of adjusting the weights is called back propogation
			self.synaptic_weights += adjustment

	def predict(self, inputs):
		#pass inputs through our neural network (our single neuron)
		return self.__sigmoid(dot(inputs, self.synaptic_weights))



if __name__ == '__main__':

	#initialize a single neuron neural network
	#will define NeuralNetwork class later
	neural_network = NeuralNetwork() 

	print 'Random starting synaptic weights:'
	print neural_network.synaptic_weights

	#The training set. We have 4 examples, each consisting of 3 input values
	# and 1 outout value.
	training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])

	#T function transposes the values from horizontal to vertical matrix
	training_set_outputs = array([[0,1,1,0]]).T

	#train the neural network using a training set.
	#Do it 10,000 times and make small adjustments each time
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print 'New synaptic weights after training: '
	print neural_network.synaptic_weights

	#Test the neural network with a new situation
	print 'predicting new situation [1,0,0] -> ?: '
	print neural_network.predict(array([1,0,0]))
