.DEFAULT_GOAL := all

BUILD_DIR = build

# compiler options
CXX = g++
BASICOPTS = -shared -Wl,--export-dynamic
CXXFLAGS = -std=c++11 -Wall

# Choose version and set python header folder
PYTHON_VERSION = 2.7
SHORT = $(subst .,,$(subst m,,$(PYTHON_VERSION)))
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# location of the Boost Python include files and library
BOOST_INC = /usr/include
BOOST_LIB = /usr/lib

MMNN_SOURCES = Net/Error/ErrorBase.cpp Net/Error/SquaredError.cpp \
		Net/Neuron/Activations/Accumulation/Accumulation.cpp Net/Neuron/Activations/Accumulation/AbsPooling.cpp Net/Neuron/Activations/Accumulation/Euclidean.cpp Net/Neuron/Activations/Accumulation/Pooling.cpp \
		Net/Neuron/Activations/ActivationFunction/ActivationFunctionBase.cpp Net/Neuron/Activations/ActivationFunction/ArcTan.cpp Net/Neuron/Activations/ActivationFunction/Binary.cpp Net/Neuron/Activations/ActivationFunction/ISRU.cpp Net/Neuron/Activations/ActivationFunction/LeakyReLU.cpp \
		Net/Neuron/Activations/ActivationFunction/Linear.cpp Net/Neuron/Activations/ActivationFunction/ReLU.cpp Net/Neuron/Activations/ActivationFunction/SoftSign.cpp Net/Neuron/Activations/ActivationFunction/SoftStep.cpp Net/Neuron/Activations/ActivationFunction/TanH.cpp \
		Net/Neuron/Activations/Activation.cpp Net/Neuron/Neuron.cpp Net/Neuron/Synapse.cpp Net/Utils/Utils.cpp Net/ErrorFunction.cpp Net/LearningRate.cpp Net/NeuralNetwork.cpp Net/Evolution.cpp

MMNN_OBJS=$(patsubst %.cpp,$(BUILD_DIR)/%.o,$(MMNN_SOURCES))

# libraries

LIBRARIES = -L$(BOOST_LIB) -l:libboost_python-py$(SHORT).so -lboost_python \
	-L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION)
	
INCLUDES = -I$(PYTHON_INCLUDE) -I$(BOOST_INC)

# Set your target here
TARGET = pymmNN


%/.: 
	mkdir -p $(@D)

# Our build directory

$(BUILD_DIR):
	@echo ""
	@echo "Creating build directory"
	mkdir -p $(BUILD_DIR)
	@echo "Done"


$(BUILD_DIR)/$(TARGET).so: $(BUILD_DIR)/$(TARGET).o $(MMNN_OBJS) | $(BUILD_DIR)/.
	@echo ""
	@echo "Linking .so file"
	$(CXX) $(BASICOPTS) $(BUILD_DIR)/$(TARGET).o $(MMNN_OBJS) $(LIBRARIES) -o $(BUILD_DIR)/$(TARGET).so $(CXXFLAGS)
	@echo "Done"

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)/.
	@mkdir -p $(@D)
	@echo ""
	@echo "Compiling $<"
	$(CXX) $(INCLUDES) -fPIC -c $< $(CXXFLAGS) -o $@
	@echo "Done"

all: $(BUILD_DIR) $(BUILD_DIR)/$(TARGET).so 
	@echo "Target all done"
	
install: $(BUILD_DIR) $(BUILD_DIR)/$(TARGET).so
	@echo ""
	@echo "Copying .so to python lib directory"
	cp $(BUILD_DIR)/$(TARGET).so /usr/lib/python$(subst m,,$(PYTHON_VERSION))/$(TARGET).so
	@echo "Done"

uninstall:
	@echo ""
	@echo "Removing .so from python lib directory"
	rm /usr/lib/python$(subst m,,$(PYTHON_VERSION))/$(TARGET).so
	@echo "Done"
	
clean:
	@echo ""
	@echo "Cleaning build files"
	rm -f $(BUILD_DIR)/$(TARGET).o $(MMNN_OBJS) $(BUILD_DIR)/$(TARGET).so
	@echo "Done"
	
distclean:
	@echo ""
	@echo "Cleaning build files and directory"
	rm -f $(BUILD_DIR)/$(TARGET).o $(BUILD_DIR)/$(TARGET).so
	rm -r $(BUILD_DIR)
	@echo "Done"