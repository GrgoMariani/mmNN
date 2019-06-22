#include <cmath>
#include "ISRU.h"

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

    ISRU::ISRU() {

    }

    ISRU::~ISRU() {

    }

    double ISRU::getActivation(double x) {
        return x/(sqrt(1.+pow(x,2)));
    }

    double ISRU::getInverse(double x) {
        if( x==1. ) return 100000.;
        return x/(sqrt(1.-pow(x,2)));
    }

    double ISRU::getDerivative(double x) {
        return pow(1./(sqrt(1.+pow(getInverse(x),2))),3);
    }

    std::string ISRU::getName(){
        return "ISRU";
    }

ISRU af_isru;

}
