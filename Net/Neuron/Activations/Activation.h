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

namespace mmNN{

enum ActivationFunctionTypes{ AF_LINEAR, AF_RELU, AF_TANH, AF_SOFTSTEP, AF_BINARY, AF_ARCTAN, AF_SOFTSIGN, AF_ISRU, AF_LEAKYRELU, AF_MAXSIZE };
enum AccumulationType       { ACCUMULATION_NORMAL, ACCUMULATION_POOLING, ACCUMULATION_ABS_POOLING, ACCUMULATION_EUCLIDEAN, ACC_MAXSIZE };

class Activation{
public:
    Activation(int functiontype, int accumulationtype=ACCUMULATION_NORMAL){
        //std::cout<<"Created Activation"<<std::endl;
        switch(functiontype){
            case AF_SOFTSTEP:   activationfunction=&af_softstep; break;
            case AF_LINEAR:     activationfunction=&af_linear; break;
            case AF_RELU:       activationfunction=&af_relu; break;
            case AF_TANH:       activationfunction=&af_tanh; break;
            case AF_BINARY:     activationfunction=&af_binary; break;
            case AF_ARCTAN:     activationfunction=&af_arctan; break;
            case AF_SOFTSIGN:   activationfunction=&af_softsign; break;
            case AF_ISRU:       activationfunction=&af_isru; break;
            case AF_LEAKYRELU:  activationfunction=&af_leakyrelu; break;
            default:            activationfunction=&af_linear; break;
        }
        switch(accumulationtype){
            case ACCUMULATION_NORMAL:       accumulation=&acc_normal; break;
            case ACCUMULATION_ABS_POOLING:  accumulation=&acc_abspooling; break;
            case ACCUMULATION_POOLING:      accumulation=&acc_pooling; break;
            case ACCUMULATION_EUCLIDEAN:    accumulation=&acc_euclidean; break;
            default:                        accumulation=&acc_normal; break;
        }
    }
    ~Activation(){
        //release
    }
    double getActivation(double x){
        lastActivation=activationfunction->getActivation(x);
        return lastActivation;
    }
    double getInverse(double x){
        return activationfunction->getInverse(x);
    }
    double getDerivative(double x){
        return activationfunction->getDerivative(x);
    }
    double getAccumulationDerivative(double input, double weight){   //TRYING SOMETHING NEW
        return accumulation->getDerivative(input, weight, getLastAccumulated() );
    }
    std::string getName(){
        return activationfunction->getName();
    }
    double getLastActivation(){
        return lastActivation;
    }
    void accumulateActivation(double howMuch){
        accumulation->accumulateActivation(howMuch, accumulated);
    }
    double getAccumulated(){
        return accumulation->getAccumulated(accumulated);
    }
    double getLastAccumulated(){
        return lastAccumulated;
    }
    void resetActivation(){
        lastAccumulated=getAccumulated();
        accumulated=0;
    }
protected:
    double accumulated=0.;
    double lastAccumulated=0.;
    double lastActivation=0.;
    Accumulation* accumulation;
    ActivationFunctionBase *activationfunction;
};

}
