#include <cmath>
#include "ArcTan.h"

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

    ArcTan::ArcTan() {

    }

    ArcTan::~ArcTan() {

    }

    double ArcTan::getActivation(double x) {
        return atan(x);
    }

    double ArcTan::getInverse(double x) {
        return tan(x);
    }

    double ArcTan::getDerivative(double x) {
        return 1./(pow(tan(x),2.)+1.);
    }

    std::string ArcTan::getName() {
        return "ARCTAN";
    }

ArcTan af_arctan;

}
