#include "Neuron.h"

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

    Neuron::Neuron(Activation* activationfunction) : activatedPrevious(0), backpropSum(0.0), backpropActivated(0.0), lastError(0.0), _maxDepth(0) {
        activation=activationfunction;
        neuron_ID = TOTAL_NEURONS++;
        //std::cout<<"Created new neuron "<<neuron_ID<<std::endl;
    }

    Neuron::~Neuron() {
        delete activation;
        //std::cout<<"Destroyed neuron "<<neuron_ID<<std::endl;
    }
    
    void Neuron::addPrevSynapse(Synapse* whatSynapse) {
        prevSynapses.push_back(whatSynapse);
    }

    void Neuron::addNextSynapse(Synapse* whatSynapse) {
        nextSynapses.push_back(whatSynapse);
    }

    void Neuron::erasePrevSynapse(Synapse* whatSynapse) {
        prevSynapses.remove(whatSynapse);
    }

    void Neuron::eraseNextSynapse(Synapse* whatSynapse) {
        nextSynapses.remove(whatSynapse);
    }

    Synapse* Neuron::decompose() {
        if( !prevSynapses.empty() )
            return prevSynapses.front();
        if( !nextSynapses.empty() )
            return nextSynapses.front();
        return nullptr;
    }

    double Neuron::getLastActivation() {
        return activation->getLastActivation();
    }

    int Neuron::getId() {
        return neuron_ID;
    }

    void Neuron::ForwardNeuron(double howMuch) {
        activatedPrevious++;
        activation->accumulateActivation(howMuch);
        if(activatedPrevious >= prevSynapses.size()){
            ActivateNeuron();
            activatedPrevious=0;
            activation->resetActivation();
        }
    }

    void Neuron::ActivateNeuron() {
        //std::cout<<"Neuron "<<neuron_ID<<" activated "<<activation->getActivation( activation->getAccumulated() )<<std::endl;
        double result = activation->getActivation( activation->getAccumulated() );
        for(auto& synapse : nextSynapses)
            synapse->getNextNeuron()->ForwardNeuron( result*synapse->getWeight() );
    }

    void Neuron::backpropagateError(double error_from_next, double learning_rate) {
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

    double Neuron::getLastError() {
        return lastError;
    }

    bool Neuron::isNeuronConnectedBefore(Neuron* neuronToCheck) {
        for(auto& synapse : prevSynapses){
            if(synapse->getPreviousNeuron()==neuronToCheck)
                return true;
        }
        return false;
    }

    bool Neuron::isNeuronConnectedAfter(Neuron* neuronToCheck) {
        for(auto& synapse : nextSynapses){
            if(synapse->getNextNeuron()==neuronToCheck)
                return true;
        }
        return false;
    }

    /*unsigned short Neuron::getMinDepth() {
        return _minDepth;
    }*/

    unsigned short Neuron::getMaxDepth() {
        return _maxDepth;
    }

    void Neuron::resetDepth() {
        //_minDepth=USHRT_MAX;
        _maxDepth=0;
    }

    void Neuron::activateDepth(unsigned short depthBefore) {
        /*if(depthBefore<_minDepth)
            _minDepth=depthBefore;*/
        if(depthBefore>_maxDepth){
            _maxDepth=depthBefore;
            for(auto& synapse : nextSynapses)
                synapse->getNextNeuron()->activateDepth(depthBefore+1);
        }
    }

unsigned int Neuron::TOTAL_NEURONS = 0;

}
