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
#include "ActivationFunctionBase.h"
namespace mmNN{

class Linear : public ActivationFunctionBase{
public:
    Linear(){
    }
    ~Linear(){
    }
    double getActivation(double x){
        return x;
    }
    double getInverse(double x){
        return x;
    }
    double getDerivative(double x){
        return 1.;
    }
    std::string getName(){
        return "LINEAR";
    }
} af_linear;

}
