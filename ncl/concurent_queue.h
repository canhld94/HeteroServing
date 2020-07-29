// Copyright (C) 2020 canhld@.kaist.ac.kr
// SPDX-License-Identifier: Apache-2.0
//


// Yet another implementation of concurent queue
// Try to do it lock-free

#include <mutex>
#include <queue>

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
        bool isEmpty();
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

    }

    template <class T>
    T concurrent_queue<T>::front() {

    }
    
    template <class T>
    void concurrent_queue<T>::pop() {

    }

    template <class T>
    bool concurrent_queue<T>::isEmpty() {

    }

    template <class T>
    ssize_t concurrent_queue<T>::size() {
        
    }

} // namespace ncl