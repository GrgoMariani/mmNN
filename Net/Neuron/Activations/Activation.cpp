#include "Activation.h"

#include "ActivationFunction/SoftStep.h"
#include "ActivationFunction/Linear.h"
#include "ActivationFunction/ReLU.h"
#include "ActivationFunction/TanH.h"
#include "ActivationFunction/Binary.h"
#include "ActivationFunction/ArcTan.h"
#include "ActivationFunction/SoftSign.h"
#include "ActivationFunction/ISRU.h"
#include "ActivationFunction/LeakyReLU.h"

#include "Accumulation/Accumulation.h"
#include "Accumulation/AbsPooling.h"
#include "Accumulation/Pooling.h"
#include "Accumulation/Euclidean.h"

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

    Activation::Activation(int functiontype, int accumulationtype) : accumulated(0.0), lastAccumulated(0.0), lastActivation(0.0) {
        //std::cout<<"Created Activation"<<std::endl;
        switch(functiontype){
            case AF_SOFTSTEP:   activationfunction=&af_softstep;    break;
            case AF_LINEAR:     activationfunction=&af_linear;      break;
            case AF_RELU:       activationfunction=&af_relu;        break;
            case AF_TANH:       activationfunction=&af_tanh;        break;
            case AF_BINARY:     activationfunction=&af_binary;      break;
            case AF_ARCTAN:     activationfunction=&af_arctan;      break;
            case AF_SOFTSIGN:   activationfunction=&af_softsign;    break;
            case AF_ISRU:       activationfunction=&af_isru;        break;
            case AF_LEAKYRELU:  activationfunction=&af_leakyrelu;   break;
            default:            activationfunction=&af_linear;      break;
        }
        switch(accumulationtype) {
            case ACCUMULATION_NORMAL:       accumulation=&acc_normal;       break;
            case ACCUMULATION_ABS_POOLING:  accumulation=&acc_abspooling;   break;
            case ACCUMULATION_POOLING:      accumulation=&acc_pooling;      break;
            case ACCUMULATION_EUCLIDEAN:    accumulation=&acc_euclidean;    break;
            default:                        accumulation=&acc_normal;       break;
        }
    }

    Activation::~Activation() {
        //release
    }

    double Activation::getActivation(double x) {
        lastActivation = activationfunction->getActivation(x);
        return lastActivation;
    }

    double Activation::getInverse(double x) {
        return activationfunction->getInverse(x);
    }

    double Activation::getDerivative(double x) {
        return activationfunction->getDerivative(x);
    }

    double Activation::getAccumulationDerivative(double input, double weight) {   //TRYING SOMETHING NEW
        return accumulation->getDerivative(input, weight, getLastAccumulated() );
    }

    std::string Activation::getName() {
        return activationfunction->getName();
    }

    double Activation::getLastActivation() {
        return lastActivation;
    }

    void Activation::accumulateActivation(double howMuch) {
        accumulation->accumulateActivation(howMuch, accumulated);
    }

    double Activation::getAccumulated() {
        return accumulation->getAccumulated(accumulated);
    }

    double Activation::getLastAccumulated() {
        return lastAccumulated;
    }

    void Activation::resetActivation() {
        lastAccumulated = getAccumulated();
        accumulated=0;
    }

}
