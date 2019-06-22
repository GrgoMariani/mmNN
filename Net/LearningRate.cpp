#include <math.h>
#include "LearningRate.h"

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

    LearningRate::LearningRate(double setlearningrate) : _lr(setlearningrate) {
        _lr=setlearningrate;
    }

    LearningRate::~LearningRate() {

    }

    double LearningRate::getCurrentLearningRate() {
        return _lr;
    }

    double LearningRate::setCurrentLearningRate(double newlearningrate) {
        _lr=newlearningrate;
        return _lr;
    }

    double LearningRate::multiplyLearningRate(double factor) {
        _lr*=factor;
        return _lr;
    }

    double LearningRate::customLearningFunction(double age) {
        if(age==0. || age==1.)
            return 0.008;
        return pow(log(age),2)/age*0.008 ;
    }

}
