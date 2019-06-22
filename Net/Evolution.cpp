#include "Evolution.h"
#include "Utils/Utils.h"

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

    Evolution::Evolution(double chance) : _chance(chance) {

    }

    Neuron* Evolution::neuronMitosisVertical(NeuralNetwork* net, Neuron* parent, double nextweight) {
        Neuron* child = net->newNeuron(rand()%AF_MAXSIZE, rand()%ACC_MAXSIZE);
        for(auto& synapse : parent->prevSynapses)
            net->link2Neurons(synapse->getPreviousNeuron(), child, random_neg1_to_1() );
        for(auto& synapse : parent->nextSynapses)
            net->link2Neurons(child, synapse->getNextNeuron(), random_neg1_or_1()*nextweight );
        net->link2bias(child, random_neg1_to_1());
        //MIGHT ERASE THIS or link with some random chance, create evolution variables
        /*net->activateDepth();
        unsigned short depth=parent->getMaxDepth();
        for( auto& neuron : net->_listofneurons )
            if(neuron->getMaxDepth() < depth )
                net->link2Neurons(neuron, child, nextweight);*/
        //TILL HERE
        net->activateDepth();
        evolveNeuron(net, child);
        evolveNeuron(net, parent);
        return child;
    }

    Neuron* Evolution::neuronMitosisHorizontal(NeuralNetwork* net, Neuron* parent, double nextweight) {
        //std::cout<<"MITOSIS HORIZONTAL STARTED"<<std::endl;
        Neuron* child = net->newNeuron(rand()%AF_MAXSIZE, rand()%ACC_MAXSIZE);
        std::vector<Synapse*> toDelete;
        for(auto& synapse : parent->nextSynapses){
            net->link2Neurons( child, synapse->getNextNeuron(), synapse->getWeight() );
            toDelete.push_back(synapse);
        }
        //net->link2Neurons( parent, child, 1. );
        net->link2Neurons( parent, child, child->activation->getInverse(1.) );
        for(auto& synapse : toDelete)
            net->deleteSynapse(synapse);
        /*activateDepth();
        unsigned short depth=parent->getMaxDepth();
        for( auto& neuron : net->_listofneurons )
            if(neuron->getMaxDepth() == depth )
                net->link2Neurons(neuron, child, nextweight);*/
        net->activateDepth();
        evolveNeuron(net, child);
        evolveNeuron(net, parent);
        net->link2bias(child, random_neg1_or_1()*nextweight);
        return child;
    }

    void Evolution::evolveNeuron(NeuralNetwork* net, Neuron* neuronToEvolve, double prevWeight) {
        for( auto& neuron : net->_listofneurons ){
            if( !(neuron->isNeuronConnectedAfter(neuronToEvolve) ) && neuron->getMaxDepth()<neuronToEvolve->getMaxDepth() && (random_neg1_to_1()+1.)/2.<_chance  ) {
                net->link2Neurons(neuron, neuronToEvolve, random_neg1_or_1()*prevWeight );
            }
        }
    }

    Neuron* Evolution::neuronEvolutionRandom(NeuralNetwork* net, unsigned short whatDepth, double nextweights) {
        Neuron* newneuron = net->newNeuron(rand()%AF_MAXSIZE, rand()%ACC_MAXSIZE);
        for( auto& neuron : net->_listofneurons){
            if( neuron->getMaxDepth()<whatDepth && (random_neg1_to_1()+1.)/2.<_chance  )
                net->link2Neurons(neuron, newneuron, random_neg1_to_1() );
            else if( neuron->getMaxDepth()>whatDepth && (random_neg1_to_1()+1.)/2.<_chance )
                net->link2Neurons(newneuron, neuron, random_neg1_or_1()*nextweights );
        }
        net->link2bias(newneuron, random_neg1_to_1());
        return newneuron;
    }

    Neuron* Evolution::neuronEvolutionRandomDepthRange(NeuralNetwork* net, unsigned short whatDepth, unsigned short whatRange, double nextweights) {
        Neuron* newneuron = net->newNeuron(rand()%AF_MAXSIZE, rand()%ACC_MAXSIZE);
        for( auto& neuron : net->_listofneurons ){
            if( neuron->getMaxDepth()<whatDepth+whatRange && (random_neg1_to_1()+1.)/2.<_chance  )
                net->link2Neurons(neuron, newneuron, random_neg1_to_1() );
            else if( neuron->getMaxDepth()>whatDepth-whatRange && (random_neg1_to_1()+1.)/2.<_chance )
                net->link2Neurons(newneuron, neuron, random_neg1_or_1()*nextweights );
        }
        net->link2bias(newneuron, random_neg1_to_1());
        return newneuron;
    }

    void Evolution::netEvolution(NeuralNetwork* net) {
        std::list<Neuron*> toEvolveVertical;
        std::list<Neuron*> toEvolveHorizontal;
        for( auto& neuron : net->_listofneurons ){
            if(neuron->getLastError()<_ErrorForVertical)
                toEvolveVertical.push_back(neuron);
            if(neuron->getLastError()>_ErrorForHorizontal)
                toEvolveHorizontal.push_back(neuron);
        }
        for(auto& neuron : toEvolveVertical)
            neuronMitosisVertical(net, neuron);
        for(auto& neuron : toEvolveHorizontal)
            neuronMitosisHorizontal(net, neuron);
        for(auto& neuron : net->_listofneurons)
            evolveNeuron(net, neuron);
    }

}
