

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/*
    Light weight CNN class that hold all metadata we need
    This will be fill during construction of ssdFPGA;
*/
typedef struct lwCNN {
    std::string name;
    std::string description;
    int num_layers;
    int num_params;
    int input_size;
    int output_size;
} lwCNN;

class ssdFPGA {

private:
    std::string _device;    // running device
    std::string _xml;       // path to OpenVino xml file
    lwCNN metadata;
public:
    // default constructor
    ssdFPGA() {
        // we should follow the RAII
        // Acquire proper resouces during constructor
    }
    // default destructor
    ~ssdFPGA() {
        // release every resource
    }
};