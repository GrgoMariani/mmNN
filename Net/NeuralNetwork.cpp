#include <math.h>
#include <sstream>
#include <iostream>
#include <omp.h>

#include "Utils/Utils.h"

#include "NeuralNetwork.h"

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

    Neuron* NeuralNetwork::newNeuron(int activationtype, int accumulationtype) {
        Neuron *neuron = new Neuron( new Activation(activationtype, accumulationtype) );
        _listofneurons.push_back(neuron);
        return neuron;
    }
    void NeuralNetwork::insertNewNeuron(Neuron* neuron) {
        _listofneurons.push_back(neuron);
    }
    void NeuralNetwork::link2Neurons(Neuron* neuronBefore, Neuron* neuronAfter, double weight) {
        if(neuronAfter->isNeuronConnectedBefore(neuronBefore)){
            //std::cout<<"ALREADY CONNECTED"<<std::endl;
            return;
        }
        Synapse *newSynapse=new Synapse(neuronBefore->getId(), neuronAfter->getId(), neuronBefore, neuronAfter, weight);
        insertNewSynapse(newSynapse);
        neuronBefore->addNextSynapse(newSynapse);
        neuronAfter->addPrevSynapse(newSynapse);
    }

    void NeuralNetwork::link2bias(Neuron* neuronToLink, double weight) {
        if( !neuronToLink->isNeuronConnectedBefore(_bias) )
            link2Neurons(_bias, neuronToLink, weight);
    }

    void NeuralNetwork::deleteNeuron(Neuron* neuronToDelete) {
        deleteSynapsesContainingNeuron(neuronToDelete);
        _listofneurons.remove(neuronToDelete);
        delete neuronToDelete;
    }

    void NeuralNetwork::deleteSynapse(Synapse* synapseToDelete) {
        synapseToDelete->getNextNeuron()->erasePrevSynapse(synapseToDelete);
        synapseToDelete->getPreviousNeuron()->eraseNextSynapse(synapseToDelete);
        removeSynapse(synapseToDelete);
        delete synapseToDelete;
    }

    void NeuralNetwork::removeSynapsesWithAbsWeightLessThan(double weight) {
        std::list<Synapse* > toRemove;
        for(auto& synapse : _listofsynapses){
            if( synapse->getWeight()<=weight && -synapse->getWeight()<=weight ){
                toRemove.push_back(synapse);
            }
        }
        for(auto& synapse : toRemove){
            std::cout<<"Removed synapse "<<synapse->getSynapseString()<<std::endl;
            deleteSynapse(synapse);
        }
    }

    void NeuralNetwork::removeUnconnectedNeurons() {
        std::list<Neuron*> toDelete;
        for(auto& neuron : _listofneurons)
            if( !isNeuronInputOrOutput(neuron) && (neuron->prevSynapses.size()==0 || neuron->nextSynapses.size()==0))
                toDelete.push_back(neuron);
        for(auto& neuron : toDelete){
            std::cout<<"Deleted neuron : "<<neuron->getId()<<std::endl;
            deleteNeuron(neuron);
        }
    }

    std::string NeuralNetwork::netInfo() {
        std::ostringstream strs;
        strs<<"Neurons : "<<_listofneurons.size()<<"    Synapses: "<<_listofsynapses.size();
        return strs.str();
    }

    unsigned short NeuralNetwork::getNetMaxDepth() {
        unsigned short maxDepth=0;
        for(auto& neuron : _listofneurons){
            if( neuron->getMaxDepth()>maxDepth ){
                maxDepth=neuron->getMaxDepth();
            }
        }
        return maxDepth;
    }
    void NeuralNetwork::insertNewSynapse(Synapse* synapseToInsert) {
        _listofsynapses.push_back(synapseToInsert);
    }

    void NeuralNetwork::deleteSynapsesContainingNeuron(Neuron* neuron) {
        Synapse* l=neuron->decompose();
        while( l!=nullptr ){
            l->getPreviousNeuron()->eraseNextSynapse(l);
            l->getNextNeuron()->erasePrevSynapse(l);
            _listofsynapses.remove(l);
            delete l;
            l=neuron->decompose();
        }
    }

    void NeuralNetwork::removeSynapse(Synapse* synapseToRemove) {
        _listofsynapses.remove(synapseToRemove);
    }

    NeuralNetwork::NeuralNetwork( unsigned int inputLayerSize, unsigned int outputLayerSize, unsigned int outputActivationType) {
        for( unsigned int i=0 ; i<inputLayerSize; i++ )
            _inputLayer.push_back( newNeuron(AF_LINEAR) );
        for( unsigned int i=0 ; i<outputLayerSize; i++ )
            _outputLayer.push_back( newNeuron(outputActivationType%AF_MAXSIZE) );
        _bias=newNeuron(AF_SOFTSIGN);
        //CONNECT OUTPUT TO BIAS
        //MAKE MIDDLE LAYER AND CONNECT TO INPUT AND OUTPUT
    }

    NeuralNetwork::~NeuralNetwork() {
        for(auto& neuron : _listofneurons)
            delete neuron;
        for(auto& synapse : _listofsynapses)
            delete synapse;
        _listofsynapses.clear();
        _listofneurons.clear();
    }

    std::vector<double> NeuralNetwork::forwardNetwork(std::vector<double> input) {
        std::vector<double> result;
        if( input.size()!=_inputLayer.size() ){
            std::cout<<std::endl<<"Input sizes don't match "<<input.size()<<" vs "<<_inputLayer.size()<<std::endl<<std::endl;
            return result;
        }
        //! OPENMP
        //#pragma omp parallel for
        for( unsigned int i=0; i<input.size(); i++){
            _inputLayer[i]->ForwardNeuron(input[i]);
        }
        _bias->ForwardNeuron(1.);
        for(auto& neuron : _outputLayer)
            result.push_back(neuron->getLastActivation());
        return result;
    }

    std::vector<Neuron*> NeuralNetwork::getInputLayer() {
        return _inputLayer;
    }

    std::vector<Neuron*> NeuralNetwork::getOutputLayer() {
        return _outputLayer;
    }

    Neuron* NeuralNetwork::getBias(){
        return _bias;
    }

    void NeuralNetwork::activateDepth() {
        for(auto& neuron : _listofneurons)
            neuron->resetDepth();
        for(auto& neuron : _inputLayer)
            neuron->activateDepth(0);
    }

    bool NeuralNetwork::isNeuronInputOrOutput(Neuron* neuronToCheck) {
        for(auto& neuron : _inputLayer)
            if( neuron==neuronToCheck )
                return true;
        for(auto& neuron : _outputLayer)
            if( neuron==neuronToCheck )
                return true;
        if( _bias==neuronToCheck )
            return true;
        return false;
    }

    double NeuralNetwork::getError(std::vector<double> desired, ErrorFunction* error_function) {
        std::vector<double> current;
        for(auto& neuron : _outputLayer)
            current.push_back( neuron->getLastActivation() );
        return error_function->getError(current, desired);
    }

    void NeuralNetwork::backPropagateFor(std::vector<double> desired, double learningrate, ErrorFunction* error_function){
        if( desired.size()!=_outputLayer.size() ){
            std::cout<<"Wrong size for backprop : "<<desired.size()<<" vs "<<_outputLayer.size()<<std::endl;
            return;
        }
        //! OPENMP
        //#pragma omp parallel for
        for(unsigned int i=0; i<_outputLayer.size(); i++){
            _outputLayer[i]->backpropagateError(error_function->getDerivative(_outputLayer[i]->getLastActivation(), desired[i]), learningrate);
        }
    }

    void NeuralNetwork::set_first_layer(){

    }

}
