#pragma once
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

//! DEFINING CONSTANTS
const double EULER_NUMBER = 2.7182818284590452353602874713;
const double PI = 3.141592653589793238462643383279502884;

//! Random functions
#include "Net/Utils/Utils.h"

//! Net, learning rate and evolution
#include "Net/NeuralNetwork.h"
#include "Net/LearningRate.h"
#include "Net/Evolution.h"

//! Error functions
#include "Net/ErrorFunction.h"


/** \brief To use the mmNN you only need to include this file
 *
 */
