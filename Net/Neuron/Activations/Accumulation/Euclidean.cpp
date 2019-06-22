#include "Euclidean.h"
#include <cmath>

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

    AccEuclidean::AccEuclidean() {

    }

    AccEuclidean::~AccEuclidean() {

    }

    void AccEuclidean::accumulateActivation(double howMuch, double& accumulated) {
        accumulated+=pow(howMuch,2);
    }

    double AccEuclidean::getAccumulated(double accumulated) {
        return sqrt(accumulated);
    }

    double AccEuclidean::getDerivative(double input, double weight, double accumulated) { //TRYING SOMETHING NEW
        //std::cout<<"DERIVATIVE "<<input<<" "<<weight<<" "<<accumulated<<std::endl;
        if(accumulated==0) return input;  //maybe return 1.
        /** If the neuron activation is Euclidean
         *  the derivative is (i_j^2*w_j) / sqrt( (i_1*w_1)^2 + ... + (i_n*w_n)^2 )
         */
        return pow(input, 2)*weight/accumulated;
    }

AccEuclidean acc_euclidean;

}
