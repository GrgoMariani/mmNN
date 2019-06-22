#ifndef __MMNN_NEURAL_NETWORK_H__
#define __MMNN_NEURAL_NETWORK_H__

#include <vector>

#include "Neuron/Synapse.h"
#include "Neuron/Neuron.h"
#include "ErrorFunction.h"

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

const double MIN_WEIGHT=0.001;

class NeuralNetwork {
friend class Evolution;
public:
    Neuron* newNeuron(int activationtype, int accumulationtype=ACCUMULATION_NORMAL);
    void insertNewNeuron(Neuron* neuron);
    void link2Neurons(Neuron* neuronBefore, Neuron* neuronAfter, double weight);
    void link2bias(Neuron* neuronToLink, double weight=MIN_WEIGHT);
    void deleteNeuron(Neuron* neuronToDelete);
    void deleteSynapse(Synapse* synapseToDelete);
    void removeSynapsesWithAbsWeightLessThan(double weight);
    void removeUnconnectedNeurons();
    std::string netInfo();
/**< Max Depths >*/
    unsigned short  getNetMaxDepth();
    void            insertNewSynapse(Synapse* synapseToInsert);
    void            deleteSynapsesContainingNeuron(Neuron* neuron);
    void            removeSynapse(Synapse* synapseToRemove);
private:
    std::list<Neuron* > _listofneurons;
    std::list<Synapse*> _listofsynapses;

/** < FIRST AND LAST LAYER >*/
public:
    NeuralNetwork( unsigned int inputLayerSize, unsigned int outputLayerSize, unsigned int outputActivationType=AF_LINEAR);
    ~NeuralNetwork();
    std::vector<double>     forwardNetwork(std::vector<double> input);
    std::vector<Neuron*>    getInputLayer();
    std::vector<Neuron*>    getOutputLayer();
    Neuron* getBias();
    void    activateDepth();
private:
    std::vector<Neuron*> _inputLayer;
    std::vector<Neuron*> _outputLayer;
    Neuron* _bias;
/** < CHECKS >*/
public:
    bool isNeuronInputOrOutput(Neuron* neuronToCheck);
/**< BACKPROPAGATION >*/
public:
    double  getError(std::vector<double> desired, ErrorFunction* error_function);
    void    backPropagateFor(std::vector<double> desired, double learningrate, ErrorFunction* error_function);
private:
    void    set_first_layer();
};

/** \brief Neural Network class
 *
 * \todo Save&Load net to&from file
 * \todo Forward only net (Should be ~50% lighter, at same speed)
 * \todo implement openmp, cuda, opencl support
 * \thoughts The framework is still "fresh"
 * \todo implement yEd graphs
 */
}

#endif//__MMNN_NEURAL_NETWORK_H__
