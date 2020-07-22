// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Copyright (C) 2020 canhld@kaist.ac.kr
// SPDX-License-Identifier: Apache-2.0

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_ssd_async/main.cpp
* \example object_detection_demo_ssd_async/main.cpp
*/
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
// websocket
#include <memory>
#include <thread>
#include <csignal>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "ssd_standalone.hpp"
#include <ext_list.hpp>

typedef struct color {
    int r;
    int g;
    int b;
} color;

// pid: that handler web server
using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
       showUsage();
       return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

void frameToBlob(const cv::Mat& frame,
                 InferRequest::Ptr& inferRequest,
                 const std::string& inputName) {
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

// base64 encode to send image over network
static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";


bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i)
  {
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = ( char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

    for (j = 0; (j < i + 1); j++)
      ret += base64_chars[char_array_4[j]];

    while((i++ < 3))
      ret += '=';

  }

  return ret;

}

std::string base64_decode(std::string const& encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  std::string ret;

  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_]; in_++;
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = base64_chars.find(char_array_4[i]);

      char_array_3[0] = ( char_array_4[0] << 2       ) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

      for (i = 0; (i < 3); i++)
        ret += char_array_3[i];
      i = 0;
    }
  }

  if (i) {
    for (j = 0; j < i; j++)
      char_array_4[j] = base64_chars.find(char_array_4[j]);

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

    for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
  }

  return ret;
}


int main(int argc, char *argv[]) {
    try {
        /** This demo covers certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher().getPluginByDevice(FLAGS_d);
        printPluginVersion(plugin, std::cout);

        /** Load extensions for the plugin **/

        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            plugin.AddExtension(extension_ptr);
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }

        /** Webserver **/
        struct sockaddr_in serv_addr;
            //Set ctrl-c handler
        int sockfd = -1, len;
        if (FLAGS_ws) {
            const char* ip = FLAGS_ws_ip.c_str();
            const char* port = FLAGS_ws_port.c_str();
            slog::info << "Websocket enable, create streaming socket at IP " << FLAGS_ws_ip << " and port " << FLAGS_ws_port << slog::endl;
            // create web server
            sockfd = socket(AF_INET, SOCK_STREAM, 0);
            if (sockfd == -1) {
                throw std::logic_error("Create socket server failed\n");
                exit(1);
            }
            memset(&serv_addr, 0, sizeof(serv_addr));
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_addr.s_addr = inet_addr(ip);
            serv_addr.sin_port = htons(atoi(port));
            if(connect(sockfd, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) == -1){
                printf("Cannot connect to server %s:%s\n", ip, port);
                exit(1);
            }

        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        slog::info << "Batch size is forced to  1." << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        /** Read labels (if any)**/
        std::string labelFileName = "/home/canhld/workplace/MEC_FPGA_DEMO/models/object_detection/common/ssd.labels";
        std::vector<std::string> labels;
        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        std::cout << "Read label file with " << labels.size() << " objects" << std::endl;
        // bouding box
        std::vector<color> bb(labels.size());
        for (auto &v:bb) {
            v.r = rand()%255;
            v.g = rand()%255;
            v.b = rand()%255;
        }
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks having only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        auto inputName = inputInfo.begin()->first;
        input->setPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            input->getInputData()->setLayout(Layout::NHWC);
        } else {
            input->getInputData()->setLayout(Layout::NCHW);
        }
        // --------------------------- Prepare output blobs -----------------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks having only one output");
        }
        DataPtr& output = outputInfo.begin()->second;
        auto outputName = outputInfo.begin()->first;
        const int num_classes = netReader.getNetwork().getLayerByName(outputName.c_str())->GetParamAsInt("num_classes");
        std::cout << "Network has " << num_classes << " classes" << std::endl;
        if (static_cast<int>(labels.size()) < num_classes) {
            if (static_cast<int>(labels.size()) == (num_classes - 1))  // if network assumes default "background" class, having no label
                labels.insert(labels.begin(), "fake");
            else
                labels.clear();
        }
        const SizeVector outputDims = output->getTensorDesc().getDims();
        const int maxProposalCount = outputDims[2];
        const int objectSize = outputDims[3];
        if (objectSize != 7) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }
        output->setPrecision(Precision::FP32);
        output->setLayout(Layout::NCHW);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;
        ExecutableNetwork network = plugin.LoadNetwork(netReader.getNetwork(), {});
        if (FLAGS_d == "CPU") {
            usleep(5000000);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest::Ptr async_infer_request_next = network.CreateInferRequestPtr();
        InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Read the input file --------------------------------------------------

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        if (!((FLAGS_i == "cam") ? cap.open(0) : cap.open(FLAGS_i.c_str()))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        const double totalFrame = cap.get(cv::CAP_PROP_FRAME_COUNT);

        // read input (video) frame
        cv::Mat curr_frame;  cap >> curr_frame;
        cv::Mat next_frame;

        if (!cap.grab()) {
            throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                                   "Failed getting next frame from the " + FLAGS_i);
        }

        // cv::VideoWriter video("outcpp.avi",cv::VideoWriter::fourcc('M','J','P','G'),20, cv::Size(width,height));

        // Data to send over network when websocket is enable
        std::ostringstream out_s;
        std::string info_str_cpp;
        std::string encoded;
        const char* encoded_c;
        size_t framesCounter = 0;

        int params[3] = {cv::IMWRITE_JPEG_QUALITY, 100, 0};
        std::vector<unsigned char> buf;
        unsigned char* ress;

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Do inference ---------------------------------------------------------
        slog::info << "Start inference " << slog::endl;

        bool isLastFrame = false;
        bool isAsyncMode = false;  // execution is always started using SYNC mode
        bool isModeChanged = false;  // set to TRUE when execution mode is changed (SYNC<->ASYNC)

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double ocv_decode_time = 0, ocv_render_time = 0;

        std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
        while (true) {
            // std::cout << framesCounter << std::endl;
            ++framesCounter;
            auto t0 = std::chrono::high_resolution_clock::now();
            // Here is the first asynchronous point:
            // in the async mode we capture frame to populate the NEXT infer request
            // in the regular mode we capture frame to the CURRENT infer request
            if (!cap.read(next_frame)) {
                if (next_frame.empty()) {
                    isLastFrame = true;  // end of video file
                } else {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            }
            if (isAsyncMode) {
                if (isModeChanged) {
                    frameToBlob(curr_frame, async_infer_request_curr, inputName);
                }
                if (!isLastFrame) {
                    frameToBlob(next_frame, async_infer_request_next, inputName);
                }
            } else if (!isModeChanged) {
                frameToBlob(curr_frame, async_infer_request_curr, inputName);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
            // in the regular mode we start the CURRENT request and immediately wait for it's completion
            if (isAsyncMode) {
                if (isModeChanged) {
                    async_infer_request_curr->StartAsync();
                }
                if (!isLastFrame) {
                    async_infer_request_next->StartAsync();
                }
            } else if (!isModeChanged) {
                async_infer_request_curr->StartAsync();
            }
            ms detection;
            ms wall;
            // std::cout << framesCounter << std::endl;
            if (async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY) == OK) {
                t1 = std::chrono::high_resolution_clock::now();
                detection = std::chrono::duration_cast<ms>(t1 - t0);

                t0 = std::chrono::high_resolution_clock::now();
                wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                wallclock = t0;

                t0 = std::chrono::high_resolution_clock::now();

                // ---------------------------Process output blobs--------------------------------------------------
                // Processing results of the CURRENT request
                const float *detections = async_infer_request_curr->GetBlob(outputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
                for (int i = 0; i < maxProposalCount; i++) {
                    float image_id = detections[i * objectSize + 0];
                    if (image_id < 0) {
                        // std::cout << "Only " << i << " proposals found" << std::endl;
                        break;
                    }

                    float confidence = detections[i * objectSize + 2];
                    auto label = static_cast<int>(detections[i * objectSize + 1]);
                    float xmin = detections[i * objectSize + 3] * width;
                    float ymin = detections[i * objectSize + 4] * height;
                    float xmax = detections[i * objectSize + 5] * width;
                    float ymax = detections[i * objectSize + 6] * height;

                    if (FLAGS_r) {
                        std::cout << "[" << i << "," << label << "] element, prob = " << confidence <<
                                  "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")"
                                  << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
                    }

                    if (confidence > FLAGS_t) {
                        /** Drawing only objects when > confidence_threshold probability **/
                        std::ostringstream conf;
                        conf << ":" << std::fixed << std::setprecision(3) << confidence;
                        color c = bb[label-1];
                        cv::putText(curr_frame,
                                    (static_cast<size_t>(label) < labels.size() ?
                                    labels[label-1] : std::string("label #") + std::to_string(label)) + conf.str(),
                                    cv::Point2f(xmin, ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    cv::Scalar(c.r,c.g,c.b));
                        cv::rectangle(curr_frame, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), cv::Scalar(c.r,c.g,c.b),3);
                    }
                }
            }
            if (!FLAGS_ws) {
                cv::imshow("Detection results", curr_frame);
            }
            else {
                out_s.str("");
                out_s << std::fixed << std::setprecision(2) << "{"
                    << "\"cv\": " << ocv_render_time << ","
                    << "\"pre\": " << ocv_decode_time << ","
                    << "\"yolo\": " << detection.count() << ","
                    << "\"total\": " << wall.count() << ","
                    << "\"now_frame\": " << framesCounter << ","
                    << "\"total_frame\": " << totalFrame
                    << "}";
                info_str_cpp = out_s.str();

                cv::imencode(".jpeg", curr_frame, buf, std::vector<int>(params,params+2));
                ress = reinterpret_cast<unsigned char*>(&buf[0]);
                encoded = base64_encode(ress, buf.size())+info_str_cpp+"\n";
                encoded_c = encoded.c_str();
                len = send(sockfd, encoded_c, sizeof(char)*encoded.size(), 0);
                if (len <=0 ) {
                    std::cout << "Sending " << len << " bytes over network" << std::endl;
                }
            }
            // video.write(curr_frame);

            t1 = std::chrono::high_resolution_clock::now();
            ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            if (isLastFrame) {
                break;
            }

            if (isModeChanged) {
                isModeChanged = false;
            }

            // Final point:
            // in the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
            curr_frame = next_frame;
            next_frame = cv::Mat();
            if (isAsyncMode) {
                async_infer_request_curr.swap(async_infer_request_next);
            }

            const int key = cv::waitKey(1);
            if (27 == key)  // Esc
                break;
            if (9 == key) {  // Tab
                isAsyncMode ^= true;
                isModeChanged = true;
            }
        }
        cap.release();
        // video.release();
        // -----------------------------------------------------------------------------------------------------
        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "Total Inference time: " << total.count() << std::endl;

        /** Show performace results **/
        if (FLAGS_pc) {
            printPerformanceCounts(*async_infer_request_curr, std::cout);
        }
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}