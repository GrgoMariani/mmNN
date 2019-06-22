#include <iostream>
#include <vector>
#include <math.h>

#include "mmNN.h"
#include <stdlib.h>
#include "MNIST.h"

using namespace std;
using namespace mmNN;

int no_of_train_images=60000;
int no_of_test_images=10000;
int size_of_images=28*28;
uchar **train_images;
uchar *train_labels;
uchar **test_images;
uchar *test_labels;

void CreateNet(NeuralNetwork& net);
void TrainNet(NeuralNetwork& net, LearningRate& lr, ErrorFunction* squaredError, int no_of_times);
void TestNet(NeuralNetwork& net);



int main()
{
    /*! Download and unzip MNIST files to C:/MNIST directory or change your dir path here*/
    train_images = read_mnist_images( "C:/MNIST/train-images.idx3-ubyte", no_of_train_images, size_of_images );
    train_labels = read_mnist_labels( "C:/MNIST/train-labels.idx1-ubyte", no_of_train_images );
    test_images  = read_mnist_images( "C:/MNIST/t10k-images.idx3-ubyte", no_of_test_images, size_of_images );
    test_labels  = read_mnist_labels( "C:/MNIST/t10k-labels.idx1-ubyte", no_of_test_images );

    NeuralNetwork net( size_of_images, 10, AF_SOFTSIGN );
    LearningRate lr(0.01);
    ErrorFunction* squaredError= new ErrorFunction(LOSS_SQUARED);

    /*! CREATE NET              */
    cout<<"CREATING NET"<<endl;
    CreateNet(net);

    /*! TRAIN NET               */
    cout<<"TRAINING NET"<<endl;
    TrainNet(net, lr, squaredError, 3);

    /*! TEST NET                */
    cout<<"TESTING NET"<<endl;
    TestNet(net);

    delete squaredError;
    return 0;
}



void CreateNet(NeuralNetwork& net){
    /*!              60-------40-------40-------40   ArcTan Neurons
     *          /       \             /        /
     *        /            \        /        /         \
     *      /                 \   /        /            \
     *   INPUT                  /\       /              OUTPUT
     *    784                 /     \  /                 10
     *   Neurons            /        / \                Neurons
     *        \           /        /      \             /
     *          \       /        /           \         /
     *               30-------20-------20-------20   TanH Neurons
     */
    vector<Neuron*> input_layer  =  net.getInputLayer();
    vector<Neuron*> output_layer =  net.getOutputLayer();
    vector<Neuron*> deep_layer_11;
    vector<Neuron*> deep_layer_21;
    vector<Neuron*> deep_layer_31;
    vector<Neuron*> deep_layer_41;
    vector<Neuron*> deep_layer_12;
    vector<Neuron*> deep_layer_22;
    vector<Neuron*> deep_layer_32;
    vector<Neuron*> deep_layer_42;
    for( unsigned int i=0; i<60; i++){
        Neuron* newNeuron = net.newNeuron(AF_ARCTAN);
        deep_layer_11.push_back( newNeuron );
        net.link2bias(newNeuron, random_neg1_to_1() );
    }
    for( unsigned int i=0; i<40; i++){
        Neuron* newNeuron = net.newNeuron(AF_ARCTAN);
        deep_layer_21.push_back( newNeuron );
        net.link2bias(newNeuron, random_neg1_to_1() );
    }
    for( unsigned int i=0; i<40; i++){
        Neuron* newNeuron = net.newNeuron(AF_ARCTAN);
        deep_layer_31.push_back( newNeuron );
        net.link2bias(newNeuron, random_neg1_to_1() );
    }
    for( unsigned int i=0; i<40; i++){
        Neuron* newNeuron = net.newNeuron(AF_ARCTAN);
        deep_layer_41.push_back( newNeuron );
        net.link2bias(newNeuron, random_neg1_to_1() );
    }
    for( unsigned int i=0; i<30; i++){
        Neuron* newNeuron = net.newNeuron(AF_TANH);
        deep_layer_12.push_back( newNeuron );
        net.link2bias(newNeuron, random_neg1_to_1() );
    }
    for( unsigned int i=0; i<20; i++){
        Neuron* newNeuron = net.newNeuron(AF_TANH);
        deep_layer_22.push_back( newNeuron );
        net.link2bias(newNeuron, random_neg1_to_1() );
    }
    for( unsigned int i=0; i<20; i++){
        Neuron* newNeuron = net.newNeuron(AF_TANH);
        deep_layer_32.push_back( newNeuron );
        net.link2bias(newNeuron, random_neg1_to_1() );
    }
    for( unsigned int i=0; i<20; i++){
        Neuron* newNeuron = net.newNeuron(AF_TANH);
        deep_layer_42.push_back( newNeuron );
        net.link2bias(newNeuron, random_neg1_to_1() );
    }
    for( unsigned int i=0; i<input_layer.size(); i++)
        for( unsigned int j=0; j<deep_layer_11.size(); j++)
            net.link2Neurons(input_layer[i], deep_layer_11[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_11.size(); i++)
        for( unsigned int j=0; j<deep_layer_21.size(); j++)
            net.link2Neurons(deep_layer_11[i], deep_layer_21[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_21.size(); i++)
        for( unsigned int j=0; j<deep_layer_31.size(); j++)
            net.link2Neurons(deep_layer_21[i], deep_layer_31[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_31.size(); i++)
        for( unsigned int j=0; j<deep_layer_41.size(); j++)
            net.link2Neurons(deep_layer_31[i], deep_layer_41[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_41.size(); i++)
        for( unsigned int j=0; j<output_layer.size(); j++)
            net.link2Neurons(deep_layer_41[i], output_layer[j], random_neg1_to_1() );
    for( unsigned int i=0; i<input_layer.size(); i++)
        for( unsigned int j=0; j<deep_layer_12.size(); j++)
            net.link2Neurons(input_layer[i], deep_layer_12[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_12.size(); i++)
        for( unsigned int j=0; j<deep_layer_22.size(); j++)
            net.link2Neurons(deep_layer_12[i], deep_layer_22[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_22.size(); i++)
        for( unsigned int j=0; j<deep_layer_32.size(); j++)
            net.link2Neurons(deep_layer_22[i], deep_layer_32[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_32.size(); i++)
        for( unsigned int j=0; j<deep_layer_42.size(); j++)
            net.link2Neurons(deep_layer_32[i], deep_layer_42[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_42.size(); i++)
        for( unsigned int j=0; j<output_layer.size(); j++)
            net.link2Neurons(deep_layer_42[i], output_layer[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_11.size(); i++)
        for( unsigned int j=0; j<deep_layer_42.size(); j++)
            net.link2Neurons(deep_layer_11[i], deep_layer_42[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_12.size(); i++)
        for( unsigned int j=0; j<deep_layer_31.size(); j++)
            net.link2Neurons(deep_layer_12[i], deep_layer_31[j], random_neg1_to_1() );
    for( unsigned int i=0; i<deep_layer_22.size(); i++)
        for( unsigned int j=0; j<deep_layer_41.size(); j++)
            net.link2Neurons(deep_layer_22[i], deep_layer_41[j], random_neg1_to_1() );
    for( unsigned int i=0; i<output_layer.size(); i++)
        net.link2bias(output_layer[i], random_neg1_to_1() );
}

void TrainNet(NeuralNetwork& net, LearningRate& lr, ErrorFunction* squaredError, int no_of_times){
    int j=0;
    while(j<no_of_train_images*no_of_times){
        if(j%1000==0){
            lr.multiplyLearningRate(0.99);
            net.removeSynapsesWithAbsWeightLessThan(0.0001);
            cout<<"training : "<<j+1<<" started | LR: "<<lr.getCurrentLearningRate()<<" | NET: "<<net.netInfo()<<endl;
        }
        int randomimage=rand()%no_of_train_images;
        vector<double> input_data;
        vector<double> output_data;
        input_data=ReadImage(train_images, randomimage);
        output_data={0., 0., 0., 0., 0., 0., 0., 0., 0., 0. };
        output_data[(int)train_labels[randomimage] ]=1.;
        net.forwardNetwork(input_data);
        net.backPropagateFor(output_data, lr.getCurrentLearningRate(), squaredError);
        j++;
    }
}

void TestNet(NeuralNetwork& net){
    for( int j=0; j<no_of_test_images; j++){
        vector<double> input_data;
        input_data=ReadRawImage(test_images, j);
        PrintImage(input_data);
        cout<<endl<<"NUMBER ON IMAGE: "<<(int)test_labels[j];
        vector<double> output_data=net.forwardNetwork(input_data);
        cout<<endl<<" 0 : "<<output_data[0];
        cout<<endl<<" 1 : "<<output_data[1];
        cout<<endl<<" 2 : "<<output_data[2];
        cout<<endl<<" 3 : "<<output_data[3];
        cout<<endl<<" 4 : "<<output_data[4];
        cout<<endl<<" 5 : "<<output_data[5];
        cout<<endl<<" 6 : "<<output_data[6];
        cout<<endl<<" 7 : "<<output_data[7];
        cout<<endl<<" 8 : "<<output_data[8];
        cout<<endl<<" 9 : "<<output_data[9];
        cin.ignore();
    }
}
