/***************************************************************************************
 * Copyright (C) 2020 canhld@.kaist.ac.kr
 * SPDX-License-Identifier: Apache-2.0
 * @b About: This file implement the synchronization template that will be use in producer
 * - consumer execution model
 ***************************************************************************************/

#pragma once 
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <mutex>
#include <condition_variable>
#include "st_ie_base.h"

using st::ie::bbox;



namespace st {
namespace sync {
    /**
     * @brief Simple bell using in queue theorem.
     * @details In producer-consumer model with event-loop, producer will pass job to consumer, 
     * then block and wait for the data. Consumer will process the request and notify the producer 
     * once it's data is ready. Bell is the effective mechanism to implement this strategy
     * @tparam CondVar condition variable type
     * @tparam Mutex mutex type
     * @tparam Key key type
     * @tparam Lock lock type
     * @tparam LiteralKey literal type that Key must have constructor from that
     * @tparam reset_state the reset state of key
     */
    template <class Key, class LiteralKey, LiteralKey reset_state,
              class CondVar = std::condition_variable, 
              class Mutex = std::mutex, 
              class Lock = std::unique_lock<std::mutex>>
    class simple_bell {
    private:
        CondVar cv;             //!< Conditional variable that producer will wait for 
        Mutex mtx;              //!< Associated mutex
        Key key = reset_state;  //!< A key to prevent surpicious wake-up.
    public:
        /**
         * @brief Construct a new simple_bell object
         * 
         */
        simple_bell() = default;
        /**
         * @brief Construct a new simple_bell object
         * 
         * @param other 
         */
        simple_bell(const simple_bell& other) = delete;
        /**
         * @brief Construct a new simple_bell object
         * 
         * @param other 
         */
        simple_bell(simple_bell&& other) = delete;
        /**
         * @brief 
         * 
         * @param rhs 
         * @return simple_bell& 
         */
        simple_bell& operator=(const simple_bell& rhs) = delete; 
        /**
         * @brief 
         * 
         * @param rhs 
         * @return simple_bell& 
         */
        simple_bell& operator=(const simple_bell&& rhs) = delete;
        /**
         * @brief Destroy the simple_bell object
         * 
         */
        ~simple_bell() {};
        /**
         * @brief Wait for sb ring the bell
         * @details This function should be called by producer after submiting the job. Once called,
         * producer will go to sleeping state and wait for cv. To perevent surpicious wake-up, producer 
         * will go out of sleeping state if and only if key is at desired state.
         * @param desired_state the desired state that producer will wait for
         */
        void wait(Key&& desired_state) {
            // assert(*key == 0);
            auto lk = lock();
            cv.wait(lk,[&](){return (key == desired_state);});
            key = reset_state;
        }
        /**
         * @brief Get the lock that associate with the mutex
         * @details We want to use mutext in the RAII manner to prevent resouce leak, so we should
         * wrapper it with a lock
         * @return Lock 
         */
        Lock lock() {
            return Lock(mtx);
        }
        /**
         * @brief Ring the bell
         * @details This function should be call by consumer to wake up the owner of the bell
         * @param set_state 
         */
        void ring(Key&& set_state) {
            // assert(key == 0);
            auto lk = lock();
            key = set_state;
            lk.unlock();
            cv.notify_one();
        }
        using ptr = std::shared_ptr<simple_bell>;
    };
    /**
     * @brief Single bell type
     * @details With single bell, each producer will have a bell and all user need to do is ring 
     * the bell to notify the consumer
     */
    using single_bell = simple_bell<int,int,0>;

    /**
     * @brief Shared bell type
     * @details With shared bell, all producer share a bell, and each producer will have its
     * own key. Before ringing the bell, consumer set the key
     * @tparam reset_state 
     */
    template <const char* reset_state>
    using shared_bell = simple_bell<std::string,const char*,reset_state>;
    
    /**
     * @brief A messagge template that producer and consumer will use to communicate
     * @tparam DataPtr 
     * @tparam Ssize 
     * @tparam ResponsePtr 
     * @tparam BellPtr 
     */
    template <class DataPtr, class Ssize, class ResponsePtr, class simple_bell>
    class message {
        using BellPtr = typename simple_bell::ptr;
    public:
        DataPtr data;               //!< The pointer that hold actual data
        Ssize size;                 //!< Size of the data
        ResponsePtr predictions;    //!< The prediction, inference engine will write the result here
        BellPtr bell;               //!< The bell object that consumer will used to notify producer
        /**
         * @brief Construct a new message object
         * 
         */
        message(): data(nullptr), size(-1), predictions(nullptr), bell(nullptr) {}
        /**
         * @brief Construct a new message object
         * 
         * @param _data 
         * @param _size 
         * @param _predictions 
         * @param _bell 
         */
        message(DataPtr& _data, Ssize& _size, ResponsePtr _predictions, BellPtr& _bell):
            data(_data), size(_size), predictions(_predictions), bell(_bell) {}
        /**
         * @brief 
         * 
         * @param rhs 
         * @return message& 
         */
        message& operator=(const message& rhs) {
            if (this != &rhs) {
                data = rhs.data;
                size = rhs.size;
                predictions = rhs.predictions;
                bell = rhs.bell;
            }
            return *this;
        }
        /**
         * @brief Construct a new message object
         * 
         * @param other 
         */
        message(const message& other) {
            *this = other;
        }
        /**
         * @brief 
         * 
         * @param rhs 
         * @return message& 
         */
        message& operator=(const message&& rhs) {
            if (this != rhs) {
                this = rhs;
                rhs.data = nullptr;
                rhs.size = -1;
                rhs.predictions = nullptr;
                rhs.bell = nullptr;
            }
        }
        /**
         * @brief Construct a new message object
         * 
         * @param other 
         */
        message(const message&& other) {
            *this = other;
        }
    };

    /**
     * @brief Message template that can hold object detection result
     * 
     * @tparam simple_bell 
     */
    template <class simple_bell>
    using obj_detection_msg = message<const char*, int, std::vector<bbox>*, simple_bell>;

    /**
     * @brief Message template that can old classification result
     * 
     * @tparam simple_bell 
     */
    template <class simple_bell>
    using classification_msg =  message<const char*, int, std::vector<int>*, simple_bell>;

    /**
     * @brief The communication channel between producer and consumer
     * @details Producer and Consumer will send and recieve messagge throught this channel. A channel 
     * must be in with a message type.
     * @tparam Message Message type
     * @tparam DeQue Queue type, default std::deque
     * @tparam CondVar 
     * @tparam Mutex 
     * @tparam Lock 
     */
    template <class Message, class DeQue = std::deque<Message>, 
              class CondVar = std::condition_variable, 
              class Mutex = std::mutex, 
              class Lock = std::unique_lock<std::mutex>>
    class blocking_queue {
    private:
        DeQue queue;    //!< The actual channel
        CondVar cv;     //!< Convar that used to block poping the queue when queue is empty
        Mutex mtx;      //!< Associated mutex 
    
    public:
        /**
         * @brief Push an item to queue
         * 
         * @param item 
         */
        void push(const Message& item) {
            {
                Lock lk{mtx};
                queue.push_back(item);
            }
            cv.notify_one();
        }
        /**
         * @brief Push an rvalue item to queue
         * 
         * @param item 
         */
        void push(Message&& item) {
            {
                Lock lk{mtx};
                queue.push_back(std::move(item));
            }
            cv.notify_one();
        }
        /**
         * @brief Pop an item from queue
         * 
         * @return Message 
         */
        Message pop() {
            Lock lk(mtx);
            cv.wait(lk,[&](){return queue.size() > 0;});
            Message ret = std::move(queue.front());
            queue.pop_front();
            return ret;
        }
        /**
         * @brief Get current number of item in queue
         * 
         * @return int 
         */
        int size() {
            Lock lk{mtx};
            return queue.size();
        }
        using ptr = std::shared_ptr<blocking_queue>;
    };

    /**
     * @brief Object detection message queue that can be used to exchange object detection message
     * 
     * @tparam simple_bell 
     */
    template <class simple_bell>
    using object_detection_mq = blocking_queue<obj_detection_msg<simple_bell>>;

    /**
     * @brief Classification message queue than can be used to exchange the classification message
     * 
     * @tparam simple_bell 
     */
    template <class simple_bell>
    using classification_mq = blocking_queue<classification_msg<simple_bell>>;
    }
}