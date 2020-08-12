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
#include <tbb/concurrent_queue.h>   // Intel tbb concurent queue
#include "ssdFPGA.h"                // from ncl

using ncl::bbox;



namespace st {
namespace internal {
    /**
     * @brief simple bell using in queue theorem
     *  
     * @tparam CondVar condition variable type
     * @tparam Mutex mutex type
     * @tparam Key key type
     * @tparam Lock lock type
     * @tparam LiteralKey literal type that Key must have constructor from that
     * @tparam reset_state the reset state of key
     */

    template <class CondVar, class Mutex, class Key, 
            class Lock, class LiteralKey, LiteralKey reset_state>
    class Bell {
    private:
        CondVar cv;
        Mutex mtx;
        Key key = reset_state;
    public:
        /**
         * @brief Construct a new Bell object
         * 
         */
        Bell() = default;
        /**
         * @brief Construct a new Bell object
         * 
         * @param other 
         */
        Bell(const Bell& other) = delete;
        /**
         * @brief Construct a new Bell object
         * 
         * @param other 
         */
        Bell(Bell&& other) = delete;
        /**
         * @brief 
         * 
         * @param rhs 
         * @return Bell& 
         */
        Bell& operator=(const Bell& rhs) = delete; 
        /**
         * @brief 
         * 
         * @param rhs 
         * @return Bell& 
         */
        Bell& operator=(const Bell&& rhs) = delete;
        /**
         * @brief Destroy the Bell object
         * 
         */
        ~Bell() {};
        /**
         * @brief 
         * 
         * @param desired_state 
         */
        void wait(Key&& desired_state) {
            // assert(*key == 0);
            auto lk = lock();
            cv.wait(lk,[&](){return (key == desired_state);});
            key = reset_state;
        }
        /**
         * @brief 
         * 
         * @return Lock 
         */
        Lock lock() {
            return Lock(mtx);
        }
        /**
         * @brief 
         * 
         * @param set_state 
         */
        void ring(Key&& set_state) {
            // assert(key == 0);
            auto lk = lock();
            key = set_state;
            lk.unlock();
            cv.notify_one();
        }
        using Ptr = std::shared_ptr<Bell>;
    };
    
    using single_bell = Bell<std::condition_variable, std::mutex, int, std::unique_lock<std::mutex>,int,0>;
    template <const char* reset_state>
    using shared_bell = Bell<std::condition_variable, std::mutex, std::string, std::unique_lock<std::mutex>,const char*,reset_state>;
    
    /**
     * @brief 
     * 
     * @tparam DataPtr 
     * @tparam Ssize 
     * @tparam ResponsePtr 
     * @tparam BellPtr 
     */
    template <class DataPtr, class Ssize, class ResponsePtr, class BellPtr>
    class msg {
    public:
        DataPtr data;               // the pointer that hold actual data
        Ssize size;                       // size of the data
        ResponsePtr predictions; // the prediction, inference engine will write the result here
        BellPtr bell;
        /**
         * @brief Construct a new msg object
         * 
         */
        msg(): data(nullptr), size(-1), predictions(nullptr), bell(nullptr) {}
        /**
         * @brief Construct a new msg object
         * 
         * @param _data 
         * @param _size 
         * @param _predictions 
         * @param _bell 
         */
        msg(DataPtr& _data, Ssize& _size, ResponsePtr _predictions, BellPtr& _bell):
            data(_data), size(_size), predictions(_predictions), bell(_bell) {}
        /**
         * @brief 
         * 
         * @param rhs 
         * @return msg& 
         */
        msg& operator=(const msg& rhs) {
            if (this != &rhs) {
                data = rhs.data;
                size = rhs.size;
                predictions = rhs.predictions;
                bell = rhs.bell;
            }
            return *this;
        }
        /**
         * @brief Construct a new msg object
         * 
         * @param other 
         */
        msg(const msg& other) {
            *this = other;
        }
        /**
         * @brief 
         * 
         * @param rhs 
         * @return msg& 
         */
        msg& operator=(const msg&& rhs) {
            if (this != rhs) {
                this = rhs;
                rhs.data = nullptr;
                rhs.size = -1;
                rhs.predictions = nullptr;
                rhs.bell = nullptr;
            }
        }
        /**
         * @brief Construct a new msg object
         * 
         * @param other 
         */
        msg(const msg&& other) {
            *this = other;
        }
    };
    /**
     * @brief 
     * 
     * @tparam BellPtr 
     */
    template <class BellPtr>
    using obj_detection_msg = msg<const char*, int, std::vector<bbox>*, BellPtr>;
    /**
     * @brief 
     * 
     * @tparam BellPtr 
     */
    template <class BellPtr>
    using classification_msg =  msg<const char*, int, int*, BellPtr>;

    /**
     * @brief 
     * 
     * @tparam Message 
     * @tparam DeQue 
     * @tparam CondVar 
     * @tparam Mutex 
     * @tparam Lock 
     */
    template <class Message, class DeQue, class CondVar, class Mutex, class Lock>
    class BlockingQueue {
    private:
        DeQue queue;
        CondVar cv;
        Mutex mtx;
    
    public:
        /**
         * @brief 
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
         * @brief 
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
         * @brief 
         * 
         * @return int 
         */
        int size() {
            Lock lk{mtx};
            return queue.size();
        }
        using Ptr = std::shared_ptr<BlockingQueue>;
    };
    /**
     * @brief 
     * 
     * @tparam BellPtr 
     */
    template <class BellPtr>
    using object_detection_mq = BlockingQueue<obj_detection_msg<BellPtr>,std::deque<obj_detection_msg<BellPtr> >, std::condition_variable, std::mutex,std::unique_lock<std::mutex> >;
    /**
     * @brief 
     * 
     * @tparam BellPtr 
     */
    template <class BellPtr>
    using classification_mq = BlockingQueue<classification_msg<BellPtr>,std::deque<obj_detection_msg<BellPtr> >, std::condition_variable, std::mutex,std::unique_lock<std::mutex> >;
    }
}