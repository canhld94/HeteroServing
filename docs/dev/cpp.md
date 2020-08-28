# Some note on C++ std features

## Smart pointer / Dynamic Memory Management

__Smart pointer__ wraps a pointer in RAII manner: it will be free when the pointer object destructor is called.

- `shared_ptr`: 
- `unique_prt`:
- `make_shared`:

## Std sync methods and termilogy

### Mutex

Basic type of lock, atomic resouce that only one thread can access at a time

Type of lock in STL:

- `lock_guard`: the most basic one one of locking, it only wrap the mutex in the RAII manner: it locks the supplied mutex during constructor and release it in destructor.
  - Example of usages: when we want to enter an critcal section
  - Disadtantage: work with only one mutex
  - In case of multiple mutex, we can create an array of `lock_guard`, but it can cause deadlock due to unordered excution, e.g. bellow. To lock multiple mutex, use `scoped_lock` container, or `std::lock` algorithm which is deadlock-free locking algorithm.

  ```C++
    std::mutex m0;
    std::mutex m1;

    void thread0 () {
        std::lock_guard<std::mutex> lk0(m0); // <-- thread 0 accquire m0
        std::lock_guard<std::mutex> lk0(m1); // <-- waiting for m1
    }
    void thread1 () {
        std::lock_guard<std::mutex> lk0(m1); // <-- thread 1 accquire m1
        std::lock_guard<std::mutex> lk1(m0); // <-- waiting for m0
    }
  ```

- `scoped_lock`: simialar to `lock_guard` but can lock multiple mutexes with deadlock-free algorithm. Both `lock_guard` and `scoped_lock` locks the supplied mutex(es) at the constructor and release them in the destrutor.
- `unique_lock`: general purpose wrapping of mutex, allow moveable ownership lock, and does not require to lock the mutex at the constructor
- `shared_lock`:

### Conditional Variable

## Utilities

- `optional<T>`: manage an optional value, i.e. value that may or may not exist --> similar to `enum`?