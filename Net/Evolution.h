#ifndef __MMNN_EVOLUTION_H__
#define __MMNN_EVOLUTION_H__

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

//MIN_WEIGHT defined in NeuralNetwork.h

namespace mmNN {

class Evolution {
public:
    Evolution(double chance);
    Neuron* neuronMitosisVertical(NeuralNetwork* net, Neuron* parent, double nextweight=MIN_WEIGHT);
    Neuron* neuronMitosisHorizontal(NeuralNetwork* net, Neuron* parent, double nextweight=MIN_WEIGHT);
    void evolveNeuron(NeuralNetwork* net, Neuron* neuronToEvolve, double prevWeight=MIN_WEIGHT);
    //PROBABLY UNUSED
    Neuron* neuronEvolutionRandom(NeuralNetwork* net, unsigned short whatDepth, double nextweights=MIN_WEIGHT);
    Neuron* neuronEvolutionRandomDepthRange(NeuralNetwork* net, unsigned short whatDepth, unsigned short whatRange, double nextweights=MIN_WEIGHT);

    void netEvolution(NeuralNetwork* net);

private:
    double _chance=0.07;
    double _ErrorForVertical=0.0001;
    double _ErrorForHorizontal=3.;
};

/** \brief Class that evolves the network
 *          Not working
 * \param mitosis vertical copies the neuron synapses to a new neuron and randomizes it's weights. Output synapses are set to MIN_WEIGHT*{-1,1}
 * \param mitosis horizontal adds a layer to a neuron. >x< becomes >x-x<
 * \param evolve neuron adds a random synapse to neuron
 * \param net evolution evolves the ANN where needed
 *
 * \thoughts I guess this sounded like a good idea at some point
 *
 */
}

#endif//__MMNN_EVOLUTION_H__
