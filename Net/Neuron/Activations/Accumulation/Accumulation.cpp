#include "Accumulation.h"
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

namespace mmNN{

    Accumulation::Accumulation() {

    }

    Accumulation::~Accumulation() {

    }

    void Accumulation::accumulateActivation(double howMuch, double& accumulated) {
        accumulated+=howMuch;
    }

    double Accumulation::getAccumulated(double accumulated) {
        return accumulated;
    }

    double Accumulation::getDerivative(double input, double weight, double accumulated) { //TRYING SOMETHING NEW, MIGHT NOT WORK
        return input;
    }

    Accumulation acc_normal;
}
