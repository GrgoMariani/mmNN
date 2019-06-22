#include <iostream>
#include <vector>
#include <math.h>

#include "mmNN.h"
#include <stdlib.h>

using namespace std;
using namespace mmNN;

int main()
{
    //Our net has 4 inputs and 1 output
    NeuralNetwork net( 4, 1 );
    LearningRate lr(0.000051);
    ErrorFunction* squaredError = new ErrorFunction(LOSS_SQUARED);

    vector<Neuron*> input_layer     = net.getInputLayer();
    vector<Neuron*> output_layer    = net.getOutputLayer();
    vector<Neuron*> second_layer;
    vector<Neuron*> third_layer;

    //! CREATE NEURONS and CONNECT THEM TO BIAS
    for( unsigned int i=0; i<10; i++){
        Neuron* newNeuron = net.newNeuron(AF_ARCTAN);
        second_layer.push_back( newNeuron );
        net.link2bias(newNeuron, random_neg1_to_1() );
    }
    for( unsigned int i=0; i<10; i++){
        Neuron* newNeuron = net.newNeuron(AF_ARCTAN);
        third_layer.push_back( newNeuron );
        net.link2bias(newNeuron, random_neg1_to_1() );
    }

    //! CONNECT NEURONS
    //Connect INPUT to second_layer
    for( unsigned int i=0; i<input_layer.size(); i++)
        for( unsigned int j=0; j<second_layer.size(); j++)
            net.link2Neurons(input_layer[i],second_layer[j], random_neg1_to_1() );
    //Connect second_layer to third_layer
    for( unsigned int i=0; i<second_layer.size(); i++)
        for( unsigned int j=0; j<third_layer.size(); j++)
            net.link2Neurons(second_layer[i],third_layer[j], random_neg1_to_1() );
    //Connect third_layer to OUTPUT
    for( unsigned int i=0; i<third_layer.size(); i++)
        for( unsigned int j=0; j<output_layer.size(); j++)
            net.link2Neurons(third_layer[i], output_layer[j], random_neg1_to_1() );

    //! CONNECT BIAS TO OUTPUT LAYER
    for( unsigned int i=0; i<output_layer.size(); i++)
        net.link2bias(output_layer[i], random_neg1_to_1() );

    //! TRAIN NETWORK
    unsigned int i=0, j=0;
    while(1){
                                        //We'll input four random bits
        vector<double> data = { random_0_or_1(), random_0_or_1(), random_0_or_1(), random_0_or_1() };
                                        //We wish input bits represent squares of binary numbers - (1000) becomes 8^2=64 , (1101) becomes 13^2=169
        vector<double> DESIRED_DATA = { pow(data[0]*8+data[1]*4+data[2]*2+data[3], 2) };

        //Forward the data and backpropagate
        net.forwardNetwork(data);
        net.backPropagateFor( DESIRED_DATA, lr.getCurrentLearningRate(), squaredError );

        //Every million iterations check what has been learned and lower the learning rate
        if( (++i)%1000000 == 0){
            lr.multiplyLearningRate(0.99);
            i=0, j++;
            system("cls");
            cout<<"Round "<<j<<" Million. Learning rate: "<<lr.getCurrentLearningRate()<<endl;

            //Print all the possible matches and find out what the net has learned
            data= {0., 0., 0., 0.};
            vector<double> result;
            while(1){
                result=net.forwardNetwork(data);
                cout<<data[0]<<" "<<data[1]<<" "<<data[2]<<" "<<data[3]<<"  ->    "<<result[0]<<endl;
                int k=3;
                while(k>=0){
                    data[k]=((int)data[k])^0x1;
                    if( data[k]==0 && k>=0)
                        k--;
                    else break;
                }
                if(k<0) break;
            }
            //Print no of neurons and synapses
            cout << endl << net.netInfo();
        }
    }
    return 0;
}
