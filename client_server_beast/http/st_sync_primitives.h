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
namespace sync {
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

    template <class Key, class LiteralKey, LiteralKey reset_state,
              class CondVar = std::condition_variable, 
              class Mutex = std::mutex, 
              class Lock = std::unique_lock<std::mutex>>
    class simple_bell {
    private:
        CondVar cv;
        Mutex mtx;
        Key key = reset_state;
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
        using ptr = std::shared_ptr<simple_bell>;
    };
    
    using single_bell = simple_bell<int,int,0>;
    template <const char* reset_state>
    using shared_bell = simple_bell<std::string,const char*,reset_state>;
    
    /**
     * @brief 
     * 
     * @tparam DataPtr 
     * @tparam Ssize 
     * @tparam ResponsePtr 
     * @tparam BellPtr 
     */
    template <class DataPtr, class Ssize, class ResponsePtr, class simple_bell>
    class message {
        using BellPtr = typename simple_bell::ptr;
    public:
        DataPtr data;               // the pointer that hold actual data
        Ssize size;                       // size of the data
        ResponsePtr predictions; // the prediction, inference engine will write the result here
        BellPtr bell;
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
     * @brief 
     * 
     * @tparam simple_bell 
     */
    template <class simple_bell>
    using obj_detection_msg = message<const char*, int, std::vector<bbox>*, simple_bell>;
    /**
     * @brief 
     * 
     * @tparam simple_bell 
     */
    template <class simple_bell>
    using classification_msg =  message<const char*, int, int*, simple_bell>;

    /**
     * @brief 
     * 
     * @tparam Message 
     * @tparam DeQue 
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
        using ptr = std::shared_ptr<blocking_queue>;
    };
    /**
     * @brief 
     * 
     * @tparam simple_bell 
     */
    template <class simple_bell>
    using object_detection_mq = blocking_queue<obj_detection_msg<simple_bell>>;
    /**
     * @brief 
     * 
     * @tparam simple_bell 
     */
    template <class simple_bell>
    using classification_mq = blocking_queue<classification_msg<simple_bell>>;
    }
}