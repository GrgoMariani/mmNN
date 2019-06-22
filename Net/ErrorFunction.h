#ifndef __ERROR_FUNCTION_H__
#define __ERROR_FUNCTION_H__

#include "Error/SquaredError.h"

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

enum ErrorType {
    LOSS_SQUARED
};

class ErrorFunction {
public:
    ErrorFunction(int errortype);
    ~ErrorFunction();
    double getError(std::vector<double> out, std::vector<double> expected);
    double getDerivative(double out, double expected);
private:
    ErrorBase* error_base;
};

/** \brief ErrorFunction Class
 *
 * \param
 * \thoughts Squared loss seems like enough, but I left room for more to be defined easily
 * just in case
 */

}

#endif//__ERROR_FUNCTION_H__
