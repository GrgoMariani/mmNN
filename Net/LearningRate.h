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

namespace mmNN{

class LearningRate{
public:
    LearningRate(double setlearningrate=0.05){
        _lr=setlearningrate;
    }
    ~LearningRate(){
    }
    double getCurrentLearningRate(){
        return _lr;
    }
    double setCurrentLearningRate(double newlearningrate){
        _lr=newlearningrate;
        return _lr;
    }
    double multiplyLearningRate(double factor){
        _lr*=factor;
        return _lr;
    }
    double customLearningFunction(double age){
        if(age==0. || age==1.) return 0.008;
        return pow(log(age),2)/age*0.008 ;
    }

private:
    double _lr=0.05;
};

/** \brief Class to set the learning rate factor
 *
 * \param
 * \thoughts If you're having trouble with backpropagation you should play around with learning rate
 *
 */

 }
