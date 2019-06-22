#ifndef __MMNN_ACTIVATION_H__
#define __MMNN_ACTIVATION_H__

#include "Accumulation/Accumulation.h"
#include "ActivationFunction/ActivationFunctionBase.h"

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

enum ActivationFunctionTypes {
    AF_LINEAR,
    AF_RELU,
    AF_TANH,
    AF_SOFTSTEP,
    AF_BINARY,
    AF_ARCTAN,
    AF_SOFTSIGN,
    AF_ISRU,
    AF_LEAKYRELU,
    AF_MAXSIZE
    };

enum AccumulationType {
    ACCUMULATION_NORMAL,
    ACCUMULATION_POOLING,
    ACCUMULATION_ABS_POOLING,
    ACCUMULATION_EUCLIDEAN,
    ACC_MAXSIZE
    };

class Activation {
public:
    Activation(int functiontype, int accumulationtype=ACCUMULATION_NORMAL);
    ~Activation();
    double getActivation(double x);
    double getInverse(double x);
    double getDerivative(double x);
    double getAccumulationDerivative(double input, double weight);
    std::string getName();
    double getLastActivation();
    void accumulateActivation(double howMuch);
    double getAccumulated();
    double getLastAccumulated();
    void resetActivation();
protected:
    double accumulated;
    double lastAccumulated;
    double lastActivation;
    Accumulation* accumulation;
    ActivationFunctionBase* activationfunction;
};

}

#endif//__MMNN_ACTIVATION_H__
