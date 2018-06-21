# mmNN
##
### micro managed NEURAL NETWORK

![mmNN](Documentation/images/mmNN.png?raw=true "micro managed NEURAL NETWORK")

####  Brief

A simple to use, ___lightweight___ neural network _C++ library/Python module_.

Watch out as this is still in development and written only for ___C++11___ and ___Python___, so I would advise you away from using it in something "big". Not to mention that this also served as a pet project to confirm some of my intuitions about NNs.

The main motivation behind this project was a deeper evaluation of abstractions in neural nets. While other frameworks might use whole layers as their 'atoms' for building the neural net, mmNN's is the neuron and the synapse.

Consider this :
![A different example](Documentation/images/picture1.png?raw=true "A different example")

This neural net has 3 inputs and 3 outputs but how many layers??? One might say 5 as the furthest road from INPUT 1 to OUT 2 is 4, but INPUT 3 directly 'touches' OUT 3 so is it 2 layers??? Or is it something in between.

Consider a problem where you need to find out if a given image is that of a circle? Next consider if it is an image of a car? First one might be easily solved in a couple of layers, but the other one needs a lot more abstractions to work properly, so a lot more layers.
Now consider a problem where you need to find out if there is a car and a circle in an image??? Why would our circle abstraction need to be represented in all the layers after it has been identified???

![Circle Car](Documentation/images/CircleAndCar.png?raw=true "Circle and Car")

(A slightly better analogy could have been a red hat.)

#### Neuron and Synapse

One thing that was quite difficult to do was simulating the pooling layer. However after some thought I've decided something new called the accumulation function.
![Neuron](Documentation/images/NeuronArchitecture.png?raw=true "Neuron Architecture")
Should the concept of activation be clear to you (ReLU, Sigmoid, Sign, ... ) the accumulation should also be.
![Accumulation](Documentation/images/Accumulation.png?raw=true "Accumulation")

It should be clear how pooling could be simulated now.

#### Install
I've made a ___Makefile___ which should take care of most of the Python installation. Be sure to change the Python version to the one you use. Currently it's set to Python2.7 .
```
$ sudo apt-get install libboost-all-dev python-dev python3-dev
$ git clone http://www.github.com/GrgoMariani/mmNN.git
$ cd mmNN
$ make
$ sudo make install
```
You can check out the example with:
```
$ python2.7 example.py
```
![Example 1](Documentation/images/Example1.png?raw=true "Example 1")

The example produces the net with 4 inputs and 1 output, as well as two hidden layers. The network teaches to convert the binary number representation to decimal of 4 inputs.

Check the example2.py also. It does the same thing in a loop, and the outputs are squares now.
Be sure to start the example a couple of times and see how it sometimes learns quickly, sometimes slow, and sometimes not at all.

#### Usage
Most you need to know should be covered in the examples. mmNN is quite new so ... , keep it simple.

After importing the module you should create your net and define number of inputs and outputs.
```
from pymmNN import NeuralNetwork, ActivationType, AccumulationType, ErrorType

net = NeuralNetwork( NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, ActivationType.LINEAR)
net.setLearningRate(0.005)
```

To get the first and the last layer of neurons:
```
input_layer =   net.getInputLayer()
output_layer =  net.getOutputLayer()
```
What we have so far is this:
![Usage 1](Documentation/images/Usage1.png?raw=true "Usage 1")

3 Inputs, 2 Outputs and 1 hidden bias neuron. No neurons are connected to each other.
We can add one neuron and connect it to each input and each output so that we have INPUT->Neuron->OUTPUT direction.
```
newNeuron = net.createNeuron(ACTIVATION_FUNCTION_TYPE, ACCUMULATION_FUNCTION_TYPE) # Define the activations and accumulation
net.linkTwoNeurons(input_layer[0], newNeuron, -1.4 )
net.linkTwoNeurons(input_layer[1], newNeuron, 2.0 )
net.linkTwoNeurons(input_layer[2], newNeuron, 1.2 )
net.linkToBias(newNeuron, 0.6 )
```
We have connected the input layers to new neuron, as well as the bias. Like this:
![Usage 2](Documentation/images/Usage2.png?raw=true "Usage 2")

We only need to connect the output neurons to it and the bias to output neurons.
```
net.linkTwoNeurons(newNeuron, output_layer[0], 4. )
net.linkTwoNeurons(newNeuron, output_layer[1], -2.34 )
net.linkToBias(output_layer[0], 0.05 )
net.linkToBias(output_layer[1], 12.0 )
```
![Usage 3](Documentation/images/Usage3.png?raw=true "Usage 3")

All the weights have been set.

Once all the connections are set feel free to use your network and "teach it" new tricks.

Input _list_ of NOI size and expect _list_ of NOO size as output.
```
# Create some input data
INPUT_DATA = [1. , 2. , 3.]
# Forward the data and get OUTPUT_DATA
OUTPUT_DATA = net.forward(data)
# You can check the outputs you got from the network in OUTPUT_DATA
DESIRED_DATA = [6., 1.]
# You will need to define ERROR_FUNCTION for backpropagation while setting up the net, as well as the learning rate
net.backprop( DESIRED_DATA)
# The network has learned something now
```

#### Other
This short intro explains most of the Python module. The default error function is _loss squared_.

[Check the Python example]( example.py )

[Check the Python example2]( example2.py )

[For C++ code check this link]( Documentation/README.md )

