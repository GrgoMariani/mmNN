# mmNN
##
### micro managed NEURAL NETWORK

![mmNN](images/mmNN.png?raw=true "micro managed NEURAL NETWORK")

####  Brief
A simple to use, lightweight neural network library.

Watch out as this is still in development and written only for C++ (-11), so I would advise you away from using it in something "big". Not to mention that this also served as a pet project to confirm some of my intuitions about NNs.

The main motivation behind this project was a deeper evaluation of abstractions in neural nets. While other frameworks might use whole layers as their 'atoms' for building the neural net, mmNN's is the neuron and the synapse.

Consider this :
![A different example](images/picture1.png?raw=true "A different example")

This neural net has 3 inputs and 3 outputs but how many layers??? One might say 5 as the furthest road from INPUT 1 to OUT 2 is 4, but INPUT 3 directly 'touches' OUT 3 so is it 2 layers??? Or is it something in between.

Consider a problem where you need to find out if a given image is that of a circle? Next consider if it is an image of a car? First one might be easily solved in a couple of layers, but the other one needs a lot more abstractions to work properly, so a lot more layers.
Now consider a problem where you need to find out if there is a car and a circle in an image??? Why would our circle abstraction need to be represented in all the layers after it has been identified???

![Circle Car](images/CircleAndCar.png?raw=true "Circle and Car")

(A slightly better analogy could have been a red hat.)

#### Neuron and Synapse

One thing that was quite difficult to do was simulating the pooling layer. However after some thought I've decided something new called the accumulation function.
![Neuron](images/NeuronArchitecture.png?raw=true "Neuron Architecture")
Should the concept of activation be clear to you (ReLU, Sigmoid, Sign, ... ) the accumulation should also be.
![Accumulation](images/Accumulation.png?raw=true "Accumulation")

It should be clear how pooling could be simulated now.

#### Install
No installation necessary, simply copy the Net directory and mmNN.h header to your project. To use the header library include the mmNN.h to your code.
``` 
#include "mmNN.h"
```
Also, don't forget to compile with C++-11.
You can check out the examples by:
```console
 make -f makefile_optional
```
This builds 3 files `build/example1.exe`, `build/example2.exe` and `build/exampleMNIST.exe` .
![Example 1](images/Example1.png?raw=true "Example 1")

The first example produces the net with 4 inputs and 1 output, as well as two hidden layers. The network teaches to square the binary number representation of 4 inputs.
Example2 is similar though it has 4 outputs [binary representation, (A AND B) OR C, number of ones, A-5B+3C-4D]

#### MNIST example
```console
make -f makefile_optional build/exampleMNIST
```
I recently added a MNIST dataset example. You will need to download the dataset from http://yann.lecun.com/exdb/mnist/ .
I used C:/MNIST/ dir for uncompressed files. You can change the directory in the exampleMNIST.cpp file.
The example tries to learn the dataset by using the following "architecture".

![Example MNIST](images/MNIST.png?raw=true "Example MNIST")

Every 1000 iterations it deletes unnecessary synapses. The images used for training are randomly translated (max 5 px) and rotated (max 35°). Initial learning rate is 0.01 and it decreases every 1000 iterations by a factor of 0.9985. Feel free to play around with these parameters.

#### Usage
Most you need to know should be covered in two examples. mmNN is quite new so ... , keep it simple.
After including the header you should create your net and define number of inputs and outputs.
```
 mmNN::NeuralNetwork net( NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS );   //Let's say NOI is 3 and NOO is 2
```
You can use namespace mmNN should you not want to use mmNN:: every time.
To get the first and last layer of neurons (technically pointers to neurons but who cares):
```
vector<mmNN::Neuron*> input_layer =   net.getInputLayer();
vector<mmNN::Neuron*> output_layer =  net.getOutputLayer();
```
What we have so far is this:
![Usage 1](images/Usage1.png?raw=true "Usage 1")

3 Inputs, 2 Outputs and 1 hidden bias neuron. No neurons are connected to each other.
We can add one neuron and connect it to each input and each output so that we have INPUT->Neuron->OUTPUT direction.
```
mmNN::Neuron* newNeuron = net.newNeuron(ACTIVATION_FUNCTION_TYPE, ACCUMULATION_FUNCTION_TYPE); //What activation to use
net.link2Neurons(input_layer[0],newNeuron, mmNN::random_neg1_to_1() );
net.link2Neurons(input_layer[1],newNeuron, mmNN::random_neg1_to_1() );
net.link2Neurons(input_layer[2],newNeuron, mmNN::random_neg1_to_1() );
net.link2bias(newNeuron, mmNN::random_neg1_to_1() );
```
We have connected the input layers to new neuron, as well as the bias. Like this:
![Usage 2](images/Usage2.png?raw=true "Usage 2")

We only need to connect the output neurons to it and the bias to output neurons.
```
net.link2Neurons(newNeuron, output_layer[0], mmNN::random_neg1_to_1() );
net.link2Neurons(newNeuron, output_layer[1], mmNN::random_neg1_to_1() );
net.link2bias(output_layer[0], mmNN::random_neg1_to_1() );
net.link2bias(output_layer[1], mmNN::random_neg1_to_1() );
```
![Usage 3](images/Usage3.png?raw=true "Usage 3")

All the weights have been set randomly between -1 and 1.
Once all the connections are set feel free to use your network and "teach it new tricks".
Input vector<double> of NOI size and expect vector<double> of NOO size as output.
```
vector<double> INPUT_DATA = { 1. , 2. , 3. };
//Forward the data and get OUTPUT_DATA
vector<double> OUTPUT_DATA = net.forwardNetwork(data);
//You can check the outputs you got from the network in OUTPUT_DATA
vector<double> DESIRED_DATA = { 6., 1. };
//You will need to define ERROR_FUNCTION for backpropagation, check example for usage
net.backPropagateFor( DESIRED_DATA, (double)LEARNING_RATE_FACTOR, ERROR_FUNCTION );
//The network has learned something now
```

#### Other
This short intro explains most of the header library.

One thing I am also working on here is Evolution class which should try (and I really do mean try, as in currently failing)
to replicate neurons replicating and making new connections to model new abstractions when they fail to learn anything new for some time, as well as staying fixed when a quality abstraction has been found. Sort of mimicking the mitosis the neuron cells make.

OpenMP, OpenCL, Cuda also somewhere on the roadmap (don't know where exactly)...
