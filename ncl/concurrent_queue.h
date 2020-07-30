// Copyright (C) 2020 canhld@.kaist.ac.kr
// SPDX-License-Identifier: Apache-2.0
//


// Yet another implementation of concurent queue
// Try to do it lock-free

#include <mutex>
#include <queue>
#include <thread>

namespace ncl {
    template <class T>
    class concurrent_queue {
    public:
        // default constructor
        concurrent_queue();
        // default destructor
        ~concurrent_queue();
        // push an element to queue
        void push(T&&);
        // get the front element of queue
        T front();
        // pop the element from queue --> may want to try_pop()?
        void pop();
        // check if the queue is empty
        bool empty();
        // return size of the queue
        ssize_t size();

    private:
        // use this lock to protect all all operator
        std::mutex mtx;
        // the actuall queue
        std::queue<T> qe;
    };

    template <class T>
    concurrent_queue<T>::concurrent_queue() {

    }
    template <class T>
    concurrent_queue<T>::~concurrent_queue() {

    }

    template <class T>
    void concurrent_queue<T>::push(T&& val) {
        std::lock_guard<std::mutex> lock(mtx);
        qe.push(val);
    }

    template <class T>
    T concurrent_queue<T>::front() {
        std::lock_guard<std::mutex> lock(mtx);
        return qe.front();
    }
    
    template <class T>
    void concurrent_queue<T>::pop() {
        std::lock_guard<std::mutex> lock(mtx);
        qe.pop();
    }

    template <class T>
    bool concurrent_queue<T>::empty() {
        std::lock_guard<std::mutex> lock(mtx);
        return qe.empty();
    }

    template <class T>
    ssize_t concurrent_queue<T>::size() {
        std::lock_guard<std::mutex> lock(mtx);
        return qe.size();
    }

} // namespace ncl