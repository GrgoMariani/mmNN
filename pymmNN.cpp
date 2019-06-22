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
#include <iostream>
#include <vector>
#include "mmNN.h"
#include <boost/python.hpp>

namespace py = boost::python;

class NN_FACADE{
private:
    mmNN::NeuralNetwork *_nn;
    mmNN::ErrorFunction *_error;
    mmNN::LearningRate *_learningrate;
    int input_size;
    int output_size;
    
public:
    NN_FACADE(int in_size, int out_size, int output_activation){
        this->input_size=in_size;
        this->output_size=out_size;
        this->_nn = new mmNN::NeuralNetwork(in_size, out_size, output_activation);
        this->_learningrate = new mmNN::LearningRate(1.);
        this->_error = new mmNN::ErrorFunction(mmNN::LOSS_SQUARED);
    }
    
    void changeErrorFunction(int error_type){
        delete this->_error;
        this->_error = new mmNN::ErrorFunction(error_type);
    }
    
    py::list forward(py::list li){
        std::vector<double> data;
        py::list result;
        if(len(li)!=this->input_size){
            std::cout<<"WRONG INPUT SIZE"<<std::endl;
            return result;
        }
        for(int i=0; i<len(li); i++){
            data.push_back( py::extract<double>(li[i]) );
        }
        std::vector<double> result_vec = this->_nn->forwardNetwork(data);
        for(auto& item : result_vec){
            result.append(item);
        }
        return result;
    }
    
    void backprop(py::list desired_data){
        std::vector<double> data;
        if(len(desired_data) != this->output_size){
            std::cout<<"WRONG OUTPUT SIZE"<<std::endl;
            return;
        }
        for(int i=0; i<len(desired_data); i++){
            data.push_back( py::extract<double>(desired_data[i]) );
        }
        this->_nn->backPropagateFor(data, this->_learningrate->getCurrentLearningRate(), this->_error);
    }
    
    void setLearningRate(double lr){
        this->_learningrate->setCurrentLearningRate(lr);
    }
    
    py::list getInputLayer(){
        py::list result;
        std::vector<mmNN::Neuron*> result_vec = this->_nn->getInputLayer();
        for(auto& item : result_vec){
            result.append(boost::ref(item));
        }
        return result;
    }
    
    py::list getOutputLayer(){
        py::list result;
        std::vector<mmNN::Neuron*> result_vec = this->_nn->getOutputLayer();
        for(auto& item : result_vec){
            result.append(boost::ref(item));
        }
        return result;
    }
    
    void linkToBias(mmNN::Neuron* neuronToLink, double weight){
        this->_nn->link2bias(neuronToLink, weight);
    }
    
    void linkTwoNeurons(mmNN::Neuron* neuron_before, mmNN::Neuron* neuron_after, double weight){
        this->_nn->link2Neurons(neuron_before, neuron_after, weight);
    }
    
    mmNN::Neuron* createNeuron(int activation_type, int accumulation_type){
        return this->_nn->newNeuron(activation_type, accumulation_type);
    }
    
    std::string info(){
        return this->_nn->netInfo();
    }
    
};



/*   OUR PYTHON MODULE    */
BOOST_PYTHON_MODULE(pymmNN)
{
    py::class_<mmNN::Neuron, boost::noncopyable>("Neuron", py::no_init);
    
    py::class_< NN_FACADE >("NeuralNetwork", py::init<int, int, int>())
        .def("forward", &NN_FACADE::forward)
        .def("backprop", &NN_FACADE::backprop)
        .def("createNeuron", &NN_FACADE::createNeuron, py::return_value_policy<py::manage_new_object>())
        .def("getInputLayer", &NN_FACADE::getInputLayer)
        .def("getOutputLayer", &NN_FACADE::getOutputLayer)
        .def("linkToBias", &NN_FACADE::linkToBias)
        .def("linkTwoNeurons", &NN_FACADE::linkTwoNeurons)
        .def("setLearningRate", &NN_FACADE::setLearningRate)
        .def("changeErrorFunction", &NN_FACADE::changeErrorFunction)
        .def("info", &NN_FACADE::info);
    
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