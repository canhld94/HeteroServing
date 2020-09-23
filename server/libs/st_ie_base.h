/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file is a generic inference engine interface. It provide an uniform 
 * interface of inference engine, so the worker don't need to worry about the underlying 
 * implementation of deep learning algorithm in OpenVino or any DL framework. The only 
 * thing that workers care is a class with constructor and an invoking method.
 ***************************************************************************************/

#pragma once 
#include <vector>
#include <string>
#include <memory>
#include <iterator>
#include <fstream>

/*
TensorRT and OpenVINO Anatomy
    OpenVino                    TensorRT (Tensorflow target)
    Plugin                  IPlugin
IR (xml + bin)              Builder, Parser              
    Network                 Network Definition
Executable Network          ICudaEngine
Inference Request           Execution Context
    Layers                  ILayers
    Tensors                 ITensor
    Blobs                   Buffer Manager

TensorRT workflows:

- Build the network:
    - prepare the config of the network (a.k.a IR in openvino)
    - create inference builder
    - create network definition
    - create parser
    - construct the network from config with parser and builder
    - create the cuda engine (a.k.a executable network)
- Running inference:
    - create inference context (a.k.a inference request)
    - create input and ouput buffer (a.k.a blobs)
    - run inference
    - processing the output

In openvino, we have very little thing to deal with input and output blobs
*/

namespace st {
namespace ie {

    /**
     * @brief Bouding box object
     * @details Basic bouding box object that can use in any recognition task
     */
    struct bbox {
        int label_id;       //!< label id
        std::string label;  //!< class name
        float prop;         //!< confidence score
        int c[4];           //!< coordinates of bounding box
    };

    /**
     * @brief Generic Inference Engine Interface
     * 
     */
    class inference_engine {
    public:
        /*******************************************/
        /* Public interface of an inference engine */
        /*******************************************/
        /**
         * @brief Run object detection
         * 
         * @param data 
         * @param size 
         * @return std::vector<bbox> 
         */
        virtual std::vector<bbox> run_detection (const char* data, int size) = 0;

        /**
         * @brief Run classification
         * 
         * @param data 
         * @param size 
         * @return std::vector<int> 
         */
        virtual std::vector<int> run_classification (const char* data, int size) = 0;

        /**
         * @brief default shared pointer
         * 
         */
        using ptr = std::shared_ptr<inference_engine>;
    protected:
        std::vector<std::string> labels;
        /**
         * @brief Construct a new inference engine object
         * 
         */
        inference_engine() {}
        /**
         * @brief Destroy the inference engine object
         * 
         */
        virtual ~inference_engine() {};
        /**
         * @brief Set the labels object
         * 
         * @param label 
         */
        void set_labels(const std::string& label) {
            std::ifstream inputFile(label);
            std::copy(std::istream_iterator<std::string>(inputFile),
                std::istream_iterator<std::string>(),
                std::back_inserter(labels));
        }
    };
} // namespace st
} // namespace ie