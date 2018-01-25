#pragma once

#include <vector>
#include <math.h>

#include "Neuron/Synapse.h"
#include "Neuron/Neuron.h"
#include "Utils/Utils.h"
#include "ErrorFunction.h"

#include <omp.h>
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
        strs<<"Neurons : "<<_listofneurons.size()<<"    Synapses: "<<_listofsynapses.size();
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
    NeuralNetwork( unsigned int inputLayerSize, unsigned int outputLayerSize, unsigned int outputActivationType=AF_LINEAR){
        for( unsigned int i=0 ; i<inputLayerSize; i++ )
            _inputLayer.push_back( newNeuron(AF_LINEAR) );
        for( unsigned int i=0 ; i<outputLayerSize; i++ )
            _outputLayer.push_back( newNeuron(outputActivationType%AF_MAXSIZE) );
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
        //! OPENMP
        //#pragma omp parallel for
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
