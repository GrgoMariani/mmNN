#define _USE_MATH_DEFINES
#include <math.h>
#include "TanH.h"

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

    TanH::TanH() {

    }

    TanH::~TanH() {

    }

    double TanH::getActivation(double x) {
        return 2./( 1.+pow(M_E, -2*x) )-1.;
    }

    double TanH::getInverse(double x) {
        if(x>=1.)           x=0.999;
        else if(x<=-1.)     x=-0.999;
        return ( log((1.+x)/(1.-x)) )/2.;
    }

    double TanH::getDerivative(double x) {
        return 1-x*x;
    }

    std::string TanH::getName() {
        return "TanH";
    }

TanH af_tanh;

}
