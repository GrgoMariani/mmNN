.DEFAULT_GOAL := all

# compiler options
CXX = g++
BASICOPTS = -shared -Wl,--export-dynamic
CXXFLAGS = -std=c++11

# Choose version and set python header folder
PYTHON_VERSION = 3.4m
SHORT = $(subst .,,$(subst m,,$(PYTHON_VERSION)))
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# location of the Boost Python include files and library
BOOST_INC = /usr/include
BOOST_LIB = /usr/lib

# libraries

LIBRARIES = -L$(BOOST_LIB) -l:libboost_python-py$(SHORT).so -lboost_python \
	-L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION)
	
INCLUDES = -I$(PYTHON_INCLUDE) -I$(BOOST_INC)

# Set your target here
TARGET = pymmNN


# Our build directory
BUILD_DIR = build
$(BUILD_DIR):
	@echo ""
	@echo "Creating build directory"
	mkdir -p $(BUILD_DIR)
	@echo "Done"


$(BUILD_DIR)/$(TARGET).so: $(BUILD_DIR)/$(TARGET).o
	@echo ""
	@echo "Linking .so file"
	$(CXX) $(BASICOPTS) $(BUILD_DIR)/$(TARGET).o $(LIBRARIES) -o $(BUILD_DIR)/$(TARGET).so $(CXXFLAGS)
	@echo "Done"

$(BUILD_DIR)/$(TARGET).o: $(TARGET).cpp
	@echo ""
	@echo "Compiling .o file"
	$(CXX) $(INCLUDES) -fPIC -c $(TARGET).cpp $(CXXFLAGS) -o $(BUILD_DIR)/$(TARGET).o
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
	rm -f $(BUILD_DIR)/$(TARGET).o $(BUILD_DIR)/$(TARGET).so
	@echo "Done"
	
distclean:
	@echo ""
	@echo "Cleaning build files and directory"
	rm -f $(BUILD_DIR)/$(TARGET).o $(BUILD_DIR)/$(TARGET).so
	rm -r $(BUILD_DIR)
	@echo "Done"