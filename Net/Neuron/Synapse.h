#ifndef __MMNN_SYNAPSE_H__
#define __MMNN_SYNAPSE_H__

#include <string>

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

class Neuron;

class Synapse {
friend class Neuron;
public:
    Synapse(int prevNeuronID, int nextNeuronID, Neuron *prev, Neuron *next, double weight);
    ~Synapse();
    
    void    updateWeight(double weight);
    double  getWeight();
    int     getPreviousNeuronID();
    int     getNextNeuronID();
    Neuron* getPreviousNeuron();
    Neuron* getNextNeuron();
    std::string getSynapseString();
private:
    unsigned int _prevID;
    unsigned int _nextID;
    Neuron *_prev;
    Neuron *_next;
    double _weight;
};

}

#endif//__MMNN_SYNAPSE_H__
