#ifndef __MMNN_LEARNING_RATE_H__
#define __MMNN_LEARNING_RATE_H__

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

class LearningRate {
public:
    LearningRate(double setlearningrate=0.05);
    ~LearningRate();

    double getCurrentLearningRate();
    double setCurrentLearningRate(double newlearningrate);
    double multiplyLearningRate(double factor);
    double customLearningFunction(double age);
    
private:
    double _lr;
};

/** \brief Class to set the learning rate factor
 *
 * \param
 * \thoughts If you're having trouble with backpropagation you should play around with learning rate
 *
 */

 }

#endif//__MMNN_LEARNING_RATE_H__
