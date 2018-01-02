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
#include <sstream>
namespace mmNN{

class Neuron;

class Synapse{
friend class Neuron;
public:
    Synapse(int prevNeuronID, int nextNeuronID, Neuron *prev, Neuron *next, double weight) : _prevID(prevNeuronID), _nextID(nextNeuronID), _prev(prev), _next(next), _weight(weight){
        //std::cout<<"Synapse created "<<getSynapseString()<<std::endl;
    }
    ~Synapse(){
        //std::cout<<"Synapse "<<getSynapseString()<<" deleted"<<std::endl;
    }
    void updateWeight(double weight){
        _weight=weight;
    }
    double getWeight(){
        return _weight;
    }
    int getPreviousNeuronID(){
        return _prevID;
    }
    int getNextNeuronID(){
        return _nextID;
    }
    Neuron* getPreviousNeuron(){
        return _prev;
    }
    Neuron* getNextNeuron(){
        return _next;
    }
    std::string getSynapseString(){
        std::ostringstream strs;
        strs << _prevID<<"<- "<<_weight<<" ->"<<_nextID;
        return strs.str();
    }
private:
    unsigned int _prevID;
    unsigned int _nextID;
    Neuron *_prev;
    Neuron *_next;
    double _weight=0.;
};

}
