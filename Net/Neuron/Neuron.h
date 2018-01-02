#pragma once
/*!
 * Copyright (c) 2018 Grgo Mariani @ Include Ltd.
 * Gnu GPL license
 * This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <list>
#include "Synapse.h"

#include "Activations/Activation.h"
namespace mmNN{

class NeuralNetwork;
class ListOfSynapses;

class Neuron{
    friend class NeuralNetwork;
    friend class Evolution;
private:
    Neuron(Activation* activationfunction){
        activation=activationfunction;
        neuron_ID = TOTAL_NEURONS++;
        //std::cout<<"Created new neuron "<<neuron_ID<<std::endl;
    }
public:
    ~Neuron(){
        delete activation;
        //std::cout<<"Destroyed neuron "<<neuron_ID<<std::endl;
    }
private:
    void addPrevSynapse(Synapse* whatSynapse){
        prevSynapses.push_back(whatSynapse);
    }
    void addNextSynapse(Synapse* whatSynapse){
        nextSynapses.push_back(whatSynapse);
    }
    void erasePrevSynapse(Synapse* whatSynapse){
        prevSynapses.remove(whatSynapse);
    }
    void eraseNextSynapse(Synapse* whatSynapse){
        nextSynapses.remove(whatSynapse);
    }
    Synapse* decompose(){
        if( !prevSynapses.empty() )
            return prevSynapses.front();
        if( !nextSynapses.empty() )
            return nextSynapses.front();
        return nullptr;
    }
public:
    double getLastActivation(){
        return activation->getLastActivation();
    }
    int getId(){
        return neuron_ID;
    }
private:
    std::list<Synapse*> prevSynapses;
    std::list<Synapse*> nextSynapses;
    unsigned int neuron_ID;
    Activation *activation;
/**< INFERENCE >*/
public:
    void ForwardNeuron(double howMuch){
        activatedPrevious++;
        activation->accumulateActivation(howMuch);
        if(activatedPrevious >= prevSynapses.size()){
            ActivateNeuron();
            activatedPrevious=0;
            activation->resetActivation();
        }
    }
    void ActivateNeuron(){
        //std::cout<<"Neuron "<<neuron_ID<<" activated "<<activation->getActivation( activation->getAccumulated() )<<std::endl;
        double result = activation->getActivation( activation->getAccumulated() );
        for(auto& synapse : nextSynapses)
            synapse->getNextNeuron()->ForwardNeuron( result*synapse->getWeight() );
    }
private:
    unsigned int activatedPrevious=0;
/**< BACKPROPAGATION >*/
public:
    void backpropagateError(double error_from_next, double learning_rate){
        backpropActivated++;
        backpropSum+=error_from_next;
        if( backpropActivated>=nextSynapses.size() ){
            double error=backpropSum*(activation->getDerivative( activation->getLastActivation() ));
            for(auto& synapse : prevSynapses)
                {
                    //double input_der=synapse->getPreviousNeuron()->getLastActivation();  // Added ACCUMULATION FUNCTION derivative so this is redundant
                    double input_der=activation->getAccumulationDerivative(synapse->getPreviousNeuron()->getLastActivation(), synapse->getWeight() );
                    double newWeight=synapse->getWeight() - error*learning_rate*(input_der );
                    synapse->getPreviousNeuron()->backpropagateError(error*synapse->getWeight(), learning_rate);
                    synapse->updateWeight(newWeight);
                }
            lastError=80.*lastError/81.+backpropSum/81.;
            backpropActivated=0;
            backpropSum=0.;
        }
    }
    double getLastError(){
        return lastError;
    }
private:
    double backpropSum=0.;
    unsigned int backpropActivated=0.;
    double lastError=0.;
public:
/**< CHECKS >*/
    bool isNeuronConnectedBefore(Neuron* neuronToCheck){
        for(auto& synapse : prevSynapses){
            if(synapse->getPreviousNeuron()==neuronToCheck)
                return true;
        }
        return false;
    }
    bool isNeuronConnectedAfter(Neuron* neuronToCheck){
        for(auto& synapse : nextSynapses){
            if(synapse->getNextNeuron()==neuronToCheck)
                return true;
        }
        return false;
    }
/**< DEPTH >*/
public:
    /*unsigned short getMinDepth(){
        return _minDepth;
    }*/
    unsigned short getMaxDepth(){
        return _maxDepth;
    }
    void resetDepth(){
        //_minDepth=USHRT_MAX;
        _maxDepth=0;
    }
    void activateDepth(unsigned short depthBefore){
        /*if(depthBefore<_minDepth)
            _minDepth=depthBefore;*/
        if(depthBefore>_maxDepth){
            _maxDepth=depthBefore;
            for(auto& synapse : nextSynapses)
                synapse->getNextNeuron()->activateDepth(depthBefore+1);
        }
    }
private:
    //unsigned short _minDepth=USHRT_MAX;
    unsigned short _maxDepth=0;

public:
    static unsigned int TOTAL_NEURONS;
};

unsigned int Neuron::TOTAL_NEURONS=0;

/** \brief NEURON CLASS
 *         The atom class of the framework
 *         Each atom consists of synapses, accumulation function and activation function
 *
 * \param synapses - links to other neurons, each synapse has it's weight
 * \param activation functions - ReLU, TanH, Linear, SoftSign, Step, ...
 * \param accumulation function - I needed something to simulate pooling layer so I came up with this concept
 *              We can calculate the neuron output (before activation) by Aw+B or we can only take max(a_i*w_i)
                Neurons can also take a geometric mean of inputs from synapses
 * \thoughts To implement RNN I should maybe spend some time drawing graphs
 *
 */

}
