#ifndef __MMNN_NEURON_H__
#define __MMNN_NEURON_H__

#include <list>
#include "Synapse.h"

#include "Activations/Activation.h"

/*!
 * Copyright (c) 2018 Grgo Mariani
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

namespace mmNN {

class NeuralNetwork;
class ListOfSynapses;

class Neuron {
    friend class NeuralNetwork;
    friend class Evolution;
private:
    Neuron(Activation* activationfunction);
public:
    ~Neuron();
private:
    void addPrevSynapse(Synapse* whatSynapse);
    void addNextSynapse(Synapse* whatSynapse);
    void erasePrevSynapse(Synapse* whatSynapse);
    void eraseNextSynapse(Synapse* whatSynapse);
    Synapse* decompose();
public:
    double getLastActivation();
    int getId();
private:
    std::list<Synapse*> prevSynapses;
    std::list<Synapse*> nextSynapses;
    unsigned int neuron_ID;
    Activation *activation;
/**< INFERENCE >*/
public:
    void ForwardNeuron(double howMuch);
    void ActivateNeuron();
private:
    unsigned int activatedPrevious;
/**< BACKPROPAGATION >*/
public:
    void backpropagateError(double error_from_next, double learning_rate);
    double getLastError();
private:
    double          backpropSum;
    unsigned int    backpropActivated;
    double          lastError;
public:
/**< CHECKS >*/
    bool isNeuronConnectedBefore(Neuron* neuronToCheck);
    bool isNeuronConnectedAfter(Neuron* neuronToCheck);
/**< DEPTH >*/
public:
    /*unsigned short getMinDepth();*/
    unsigned short getMaxDepth();
    void resetDepth();
    void activateDepth(unsigned short depthBefore);
private:
    //unsigned short _minDepth=USHRT_MAX;
    unsigned short _maxDepth;

public:
    static unsigned int TOTAL_NEURONS;
};


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

#endif//__MMNN_NEURON_H__
