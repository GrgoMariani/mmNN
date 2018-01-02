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

#include "Error/SquaredError.h"

namespace mmNN{

enum ErrorType { LOSS_SQUARED };

class ErrorFunction{
public:
    ErrorFunction(int errortype){
        switch(errortype){
            case LOSS_SQUARED:  error_base=new SquaredError; break;
            default:            error_base=new SquaredError;
        }
    }
    ~ErrorFunction(){
        delete error_base;
    }
    double getError(std::vector<double> out, std::vector<double> expected){
        return error_base->getError(out, expected);
    }
    double getDerivative(double out, double expected){
        return error_base->getDerivative(out, expected);
    }
private:
    ErrorBase* error_base;
};

/** \brief ErrorFunction Class
 *
 * \param
 * \thoughts Squared loss seems like enough, but I left room for more to be defined easily
 */
}
