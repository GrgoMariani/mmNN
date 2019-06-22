#define _USE_MATH_DEFINES
#include <math.h>
#include "SoftSign.h"

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

    SoftSign::SoftSign() {

    }

    SoftSign::~SoftSign() {

    }

    double SoftSign::getActivation(double x) {
        return (x)/(1+abs(x));
    }

    double SoftSign::getInverse(double x) {
        return tan(x)*2./M_PI; //An incorrect approximation
    }

    double SoftSign::getDerivative(double x) {
        return 1./(1+pow(abs(getInverse(x)),2.));
    }

    std::string SoftSign::getName() {
        return "SOFTSIGN";
    }

SoftSign af_softsign;

}
