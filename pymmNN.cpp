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
#include <iostream>
#include <vector>
#include "mmNN.h"
#include <boost/python.hpp>

namespace py = boost::python;

class NN_FACADE{
private:
    mmNN::NeuralNetwork *_nn;
    mmNN::ErrorFunction *_error;
    int input_size;
    int output_size;
public:
    NN_FACADE(int in_size, int out_size, int output_activation){
        this->input_size=in_size;
        this->output_size=out_size;
        this->_nn = new mmNN::NeuralNetwork(in_size, out_size, output_activation);
    }
    
    ~NN_FACADE(){
        delete this->_nn;
        delete this->_error;
    }
    
    void setErrorFunction(int error_type){
        this->_error = new mmNN::ErrorFunction(error_type);
    }
    
    void forward(py::list li){
        std::vector<double> data;
        if(len(li)!=this->input_size){
            std::cout<<"WRONG INPUT SIZE"<<std::endl;
            return;
        }
        for(int i=0; i<len(li); i++){
            data.push_back( py::extract<double>(li[i]) );
        }
        this->_nn->forwardNetwork(data);
    }
    
};

BOOST_PYTHON_MODULE(pymmNN)
{
    
    
    py::class_< NN_FACADE >("NeuralNetwork", py::init<int, int, int>())
    .def("forward", &NN_FACADE::forward);
    
    py::class_< mmNN::LearningRate >("LearningRate", py::init<double>())
      .def("get", &mmNN::LearningRate::getCurrentLearningRate)
      .def("set", &mmNN::LearningRate::setCurrentLearningRate)
      .def("multiply", &mmNN::LearningRate::multiplyLearningRate);
    
    // ENUMS
    py::enum_<mmNN::ActivationFunctionTypes>("ActivationType")
    .value("LINEAR", mmNN::AF_LINEAR).value("RELU", mmNN::AF_RELU)
    .value("TANH", mmNN::AF_TANH).value("SOFTSTEP", mmNN::AF_SOFTSTEP)
    .value("BINARY", mmNN::AF_BINARY).value("ARCTAN", mmNN::AF_ARCTAN)
    .value("SOFTSIGN", mmNN::AF_SOFTSIGN).value("ISRU", mmNN::AF_ISRU)
    .value("LEAKYRELU", mmNN::AF_LEAKYRELU);
    
    py::enum_<mmNN::AccumulationType>("AccumulationType")
    .value("NORMAL", mmNN::ACCUMULATION_NORMAL).value("POOLING", mmNN::ACCUMULATION_POOLING)
    .value("ABS_POOLING", mmNN::ACCUMULATION_ABS_POOLING).value("EUCLIDEAN", mmNN::ACCUMULATION_EUCLIDEAN);
    
    py::enum_<mmNN::ErrorType>("ErrorType")
    .value("LOSS_SQUARED", mmNN::LOSS_SQUARED);
    
}