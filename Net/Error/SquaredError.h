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
#include <math.h>
#include "ErrorBase.h"
namespace mmNN{

class SquaredError : public ErrorBase{
public:
    SquaredError(){
    }
    ~SquaredError(){
    }
    double getError(std::vector<double> out, std::vector<double> expected){
        double result=0.;
        if( out.size()!=expected.size() || out.size()==0 ){
            std::cout<<std::endl<<"Sizes not matching : "<<out.size()<<" vs "<<expected.size()<<std::endl;
            return result;
        }
        for( unsigned int i=0; i<out.size(); i++ )
            result+=pow(out[i]-expected[i], 2.)/2.;
        return result;
    }

    double getDerivative(double out, double expected){
        return out-expected;
    }
};

}
