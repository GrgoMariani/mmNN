#include <sstream>
#include "Synapse.h"

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

    Synapse::Synapse(int prevNeuronID, int nextNeuronID, Neuron *prev, Neuron *next, double weight) : _prevID(prevNeuronID), _nextID(nextNeuronID), _prev(prev), _next(next), _weight(weight) {
        //std::cout<<"Synapse created "<<getSynapseString()<<std::endl;
    }

    Synapse::~Synapse() {
        //std::cout<<"Synapse "<<getSynapseString()<<" deleted"<<std::endl;
    }

    void Synapse::updateWeight(double weight) {
        _weight=weight;
    }

    double Synapse::getWeight() {
        return _weight;
    }

    int Synapse::getPreviousNeuronID() {
        return _prevID;
    }

    int Synapse::getNextNeuronID() {
        return _nextID;
    }

    Neuron* Synapse::getPreviousNeuron() {
        return _prev;
    }

    Neuron* Synapse::getNextNeuron() {
        return _next;
    }

    std::string Synapse::getSynapseString() {
        std::ostringstream strs;
        strs << _prevID << "<- " << _weight << " ->" << _nextID;
        return strs.str();
    }

}
