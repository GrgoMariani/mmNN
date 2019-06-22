#include "ReLU.h"

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

    ReLU::ReLU() {

    }

    ReLU::~ReLU() {

    }

    double ReLU::getActivation(double x) {
        return (x>0) ? x:0.;
    }

    double ReLU::getInverse(double x) {
        return (x>0) ? x:0.;
    }

    double ReLU::getDerivative(double x) {
        return (x>0) ? 1.:0.;
    }
    
    std::string ReLU::getName() {
        return "ReLU";
    }

ReLU af_relu;

}
