#ifndef __MMNN_SOFTSTEP_H__
#define __MMNN_SOFTSTEP_H__

#include "ActivationFunctionBase.h"

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

class SoftStep : public ActivationFunctionBase {
public:
    SoftStep();
    ~SoftStep();
    double getActivation(double x);
    double getInverse(double x);
    double getDerivative(double x);
    std::string getName();
};

extern SoftStep af_softstep;

}

#endif//__MMNN_SOFTSTEP_H__
