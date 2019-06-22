#ifndef __MMNN_ACTIVATIONFUNCTIONBASE_H__
#define __MMNN_ACTIVATIONFUNCTIONBASE_H__

#include <string>

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

class ActivationFunctionBase {
public:
    virtual ~ActivationFunctionBase();
    virtual double getActivation (double x) =0;
    virtual double getInverse    (double x) =0;
    virtual double getDerivative (double x) =0;
    virtual std::string getName()           =0;
};

}

#endif//__MMNN_ACTIVATIONFUNCTIONBASE_H__
