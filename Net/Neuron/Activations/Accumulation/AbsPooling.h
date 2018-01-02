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
#include "Accumulation.h"
namespace mmNN{

class AccAbsPooling : public Accumulation{
public:
    AccAbsPooling(){}
    ~AccAbsPooling(){}
    void accumulateActivation(double howMuch, double& accumulated){
        double d1=(howMuch<0)?(-howMuch):howMuch;
        double d2=(accumulated<0)?(-accumulated):accumulated;
        if( d2<d1 ) accumulated=howMuch;
    }
    /*double getAccumulated(double accumulated){
        return accumulated;
    }*/
} acc_abspooling;

}
