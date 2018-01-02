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

#include <vector>
#include <math.h>

#include "Neuron/Synapse.h"
#include "Neuron/Neuron.h"
#include "Utils/Utils.h"
#include "ErrorFunction.h"

namespace mmNN{

const double MIN_WEIGHT=0.001;

class NeuralNetwork{
friend class Evolution;
public:
    Neuron* newNeuron(int activationtype, int accumulationtype=ACCUMULATION_NORMAL){
        Neuron *neuron = new Neuron( new Activation(activationtype, accumulationtype) );
        _listofneurons.push_back(neuron);
        return neuron;
    }
    void insertNewNeuron(Neuron* neuron){
        _listofneurons.push_back(neuron);
    }
    void link2Neurons(Neuron* neuronBefore, Neuron* neuronAfter, double weight){
        if(neuronAfter->isNeuronConnectedBefore(neuronBefore)){
            //std::cout<<"ALREADY CONNECTED"<<std::endl;
            return;
        }
        Synapse *newSynapse=new Synapse(neuronBefore->getId(), neuronAfter->getId(), neuronBefore, neuronAfter, weight);
        insertNewSynapse(newSynapse);
        neuronBefore->addNextSynapse(newSynapse);
        neuronAfter->addPrevSynapse(newSynapse);
    }
    void link2bias(Neuron* neuronToLink, double weight=MIN_WEIGHT){
        if( !neuronToLink->isNeuronConnectedBefore(_bias) )
            link2Neurons(_bias, neuronToLink, weight);
    }
    void deleteNeuron(Neuron* neuronToDelete){
        deleteSynapsesContainingNeuron(neuronToDelete);
        _listofneurons.remove(neuronToDelete);
        delete neuronToDelete;
    }
    void deleteSynapse(Synapse* synapseToDelete){
        synapseToDelete->getNextNeuron()->erasePrevSynapse(synapseToDelete);
        synapseToDelete->getPreviousNeuron()->eraseNextSynapse(synapseToDelete);
        removeSynapse(synapseToDelete);
        delete synapseToDelete;
    }
    void removeSynapsesWithAbsWeightLessThan(double weight){
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
    void removeUnconnectedNeurons(){
        std::list<Neuron*> toDelete;
        for(auto& neuron : _listofneurons)
            if( !isNeuronInputOrOutput(neuron) && (neuron->prevSynapses.size()==0 || neuron->nextSynapses.size()==0))
                toDelete.push_back(neuron);
        for(auto& neuron : toDelete){
            std::cout<<"Deleted neuron : "<<neuron->getId()<<std::endl;
            deleteNeuron(neuron);
        }
    }
    std::string netInfo(){
        std::ostringstream strs;
        strs<<"Neurons : "<<_listofneurons.size()<<"    Synapses: "<<_listofsynapses.size()<<std::endl;
        return strs.str();
    }
/**< Max Depths >*/
    unsigned short getNetMaxDepth(){
        unsigned short maxDepth=0;
        for(auto& neuron : _listofneurons){
            if( neuron->getMaxDepth()>maxDepth ){
                maxDepth=neuron->getMaxDepth();
            }
        }
        return maxDepth;
    }
    void insertNewSynapse(Synapse* synapseToInsert){
        _listofsynapses.push_back(synapseToInsert);
    }
    void deleteSynapsesContainingNeuron(Neuron* neuron){
        Synapse* l=neuron->decompose();
        while( l!=nullptr ){
            l->getPreviousNeuron()->eraseNextSynapse(l);
            l->getNextNeuron()->erasePrevSynapse(l);
            _listofsynapses.remove(l);
            delete l;
            l=neuron->decompose();
        }
    }
    void removeSynapse(Synapse* synapseToRemove){
        _listofsynapses.remove(synapseToRemove);
    }
private:
    std::list<Neuron* > _listofneurons;
    std::list<Synapse*> _listofsynapses;

/** < FIRST AND LAST LAYER >*/
public:
    NeuralNetwork( unsigned int inputLayerSize, unsigned int outputLayerSize){
        for( unsigned int i=0 ; i<inputLayerSize; i++ )
            _inputLayer.push_back( newNeuron(AF_LINEAR) );
        for( unsigned int i=0 ; i<outputLayerSize; i++ )
            _outputLayer.push_back( newNeuron(AF_LINEAR) );
        _bias=newNeuron(AF_SOFTSIGN);
        //CONNECT OUTPUT TO BIAS
        //MAKE MIDDLE LAYER AND CONNECT TO INPUT AND OUTPUT
    }
    ~NeuralNetwork(){
        for(auto& neuron : _listofneurons)
            delete neuron;
        for(auto& synapse : _listofsynapses)
            delete synapse;
        _listofsynapses.clear();
        _listofneurons.clear();
    }
    std::vector<double> forwardNetwork(std::vector<double> input){
        std::vector<double> result;
        if( input.size()!=_inputLayer.size() ){
            std::cout<<std::endl<<"Input sizes don't match "<<input.size()<<" vs "<<_inputLayer.size()<<std::endl<<std::endl;
            return result;
        }
        for( unsigned int i=0; i<input.size(); i++){
            _inputLayer[i]->ForwardNeuron(input[i]);
        }
        _bias->ForwardNeuron(1.);
        for(auto& neuron : _outputLayer)
            result.push_back(neuron->getLastActivation());
        return result;

    }
    std::vector<Neuron*> getInputLayer(){
        return _inputLayer;
    }
    std::vector<Neuron*> getOutputLayer(){
        return _outputLayer;
    }
    Neuron* getBias(){
        return _bias;
    }
    void activateDepth(){
        for(auto& neuron : _listofneurons)
            neuron->resetDepth();
        for(auto& neuron : _inputLayer)
            neuron->activateDepth(0);
    }
private:
    std::vector<Neuron*> _inputLayer;
    std::vector<Neuron*> _outputLayer;
    Neuron* _bias;
/** < CHECKS >*/
public:
    bool isNeuronInputOrOutput(Neuron* neuronToCheck){
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
/**< BACKPROPAGATION >*/
public:
    double getError(std::vector<double> desired, ErrorFunction* error_function){
        std::vector<double> current;
        for(auto& neuron : _outputLayer)
            current.push_back( neuron->getLastActivation() );
        return error_function->getError(current, desired);
    }
    void backPropagateFor(std::vector<double> desired, double learningrate, ErrorFunction* error_function){
        if( desired.size()!=_outputLayer.size() ){
            std::cout<<"Wrong size for backprop : "<<desired.size()<<" vs "<<_outputLayer.size()<<std::endl;
            return;
        }
        for(unsigned int i=0; i<_outputLayer.size(); i++){
            _outputLayer[i]->backpropagateError(error_function->getDerivative(_outputLayer[i]->getLastActivation(), desired[i]), learningrate);
        }
    }
private:

    void set_first_layer(){
    }
};

/** \brief Neural Network class
 *
 * \todo Save&Load net to&from file
 * \todo Forward only net (Should be ~50% lighter, at same speed)
 * \todo implement openmp, cuda, opencl support
 * \thoughts The framework is still "fresh"
 *
 */
}
