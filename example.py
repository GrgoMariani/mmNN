from pymmNN import LearningRate, NeuralNetwork, ActivationType, AccumulationType, ErrorType
import random

# We'll be using these two rand functions
def getRand():
	return random.uniform(-1., 1.)
def rand01():
	return random.randint(0, 1)

# define our learning rate
lr = LearningRate(0.0051)

# create our net and set error function for backprop
net = NeuralNetwork(4, 1, ActivationType.LINEAR)
net.setErrorFunction(ErrorType.LOSS_SQUARED)
net.setLearningRate(lr.get())

# get the net's inputs and outputs to connect
input_layer = net.getInputLayer()
second_layer = []
third_layer = []
output_layer = net.getOutputLayer()

# let's create our second and third layer
for _ in range(10):
	temp_neuron = net.createNeuron(ActivationType.ARCTAN, AccumulationType.NORMAL)
	second_layer.append(temp_neuron)
	net.linkToBias(temp_neuron, getRand())
for _ in range(10):
	temp_neuron = net.createNeuron(ActivationType.ARCTAN, AccumulationType.NORMAL)
	third_layer.append(temp_neuron)
	net.linkToBias(temp_neuron, getRand())

# Here is where we fully connect everything
# Input layer to Second layer
for i_neuron in input_layer:
	for s_neuron in second_layer:
		net.linkTwoNeurons(i_neuron, s_neuron, getRand())
# Second layer to Third layer
for s_neuron in second_layer:
	for t_neuron in third_layer:
		net.linkTwoNeurons(s_neuron, t_neuron, getRand())
# Third layer to Output Layer
for t_neuron in third_layer:
	for o_neuron in output_layer:
		net.linkTwoNeurons(t_neuron, o_neuron, getRand())
# Also connect ouput bias
for o_neuron in output_layer:
	net.linkToBias(o_neuron, getRand())

# Do our training
for _ in range(100000):
	data = [rand01(), rand01(), rand01(), rand01()]
	net.forward(data)
	desired = [data[0]*8 + data[1]*4 + data[2]*2 + data[3]]
	net.backprop(desired)

# Print everything
print("INFO: {}".format(net.info))
for a in range(2):
	for b in range(2):
		for c in range(2):
			for d in range(2):
				data = [a, b, c, d]
				print("{}: {}".format(data, net.forward(data)) )

