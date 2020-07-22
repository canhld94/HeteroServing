

// C include
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
// C++
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <memory>
#include <thread>
#include <csignal>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include <ext_list.hpp>

// we use stl for most of our container 
// just avoid so many mess std::
using std::cout;
using std::string;
using std::vector;
using std::endl;

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/*
    This header implement three namespace:
    - common: common data structures
    - helper: utilities function that will be use by the main namespace
    - ncl: implement the core inference logic
    Why so much namespace? Who know, I like it
*/

namespace common {
    /*
    Light weight CNN class that hold all metadata we need
    This will be fill during construction of ssdFPGA;
    */
    class lwCNN {
    public:
        string name;
        string description;
        int num_layers;
        int num_params;
        int input_size;
        int output_size;
    };

    /*
    Bounding box class the represent a prediction
    The bounding box is (c[0],c[1]) --> bottom left, (c[2],c[3]) --> top right
    */

    class bbox {
    public:
        int label_id;       // label id
        string label;       // class name
        double prop;        // confidence score
        int c[4];           // coordinates of bounding box
    };
} // namespace commom


namespace helper {
    using namespace common;

    /*
        !Testing: return a random bbox vector
    */
    vector<bbox> runtest(const char *data) {
        int n = 3;
        vector<bbox> ret(n);
        for (int i = 0; i < n; ++i) {
            // generate a fake bbox
            ret[i].label_id = rand()%91;
            ret[i].label = "lalaland_" + std::to_string(i);
            ret[i].prop = 100.0/(rand()%100);
            for (int j = 0; j < 4; ++j) {
                ret[i].c[j] = rand()%1000;
            }
        }
        return ret;
    }
}

namespace ncl {
    using namespace helper;
    using namespace common;

    /*
        SSD class that implement everything we need
    */

    class ssdFPGA {
    private:
        string _device;    // running device
        string _xml;       // path to OpenVino xml file
        lwCNN metadata;
    public:
        // default constructor
        ssdFPGA();
        // default destructor
        ~ssdFPGA();
        // run inference with input is stream from network
        // return a vector of bbox
        vector<bbox> run(const char*);
        // get the metadata
        lwCNN get();
    };

//-------------------------------------------------------------------------

    ssdFPGA::ssdFPGA() {
        // we should follow the RAII
        // Acquire proper resouces during constructor
        // TODO: discover and lock FPGA card, reprogramming the device and ready to launch

    }

    ssdFPGA::~ssdFPGA() {
        // release every resource
        // TODO: release any resource that we accquire before
    }

    lwCNN ssdFPGA::get() {
        // TODO
        lwCNN ret;
        return ret;
    }


    vector<bbox> ssdFPGA::run(const char *data) {
        // TODO: implement
        return runtest(data);
    }
} // namespace ncl