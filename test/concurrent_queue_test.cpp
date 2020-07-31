
#include <concurrent_queue.h>
#include <gtest/gtest.h>
#include <thread>
#include <functional>


namespace {
    TEST(enqueue,normal) {
        ncl::concurrent_queue<int> Q;
        EXPECT_EQ(true,Q.empty());
        Q.push(0);
        Q.push(1);
        EXPECT_EQ(0,Q.front());
        Q.pop();
        EXPECT_EQ(1,Q.front());
        Q.pop();
        EXPECT_EQ(true,Q.empty());
    }

    TEST(enqueue,concurrent) {
        ncl::concurrent_queue<int> Q;
        int i = 0;
        auto func = [&]() {
            int ii = i+1;
            Q.push(std::move(ii));
            ++i;
        };
        std::thread t1(std::bind(func));
        std::thread t2(std::bind(func));
        t1.join();
        t2.join();
        EXPECT_EQ(Q.front(),1);
        Q.pop();
        EXPECT_EQ(Q.front(),2);
    }

    TEST(dequeue,empty) {
        ncl::concurrent_queue<int> Q;
        auto consumer = [&]() {
            while (Q.empty()) {}
            sleep(0.01);
            Q.pop();
        };
        auto producer = [&]() {
            Q.push(1);
        };
        std::thread c0(std::bind(consumer)), c1(std::bind(consumer)); // consumers
        std::thread p0(std::bind(producer)); // producers
        p0.detach();
        c0.join();
        c1.join();
        EXPECT_EQ(Q.empty(),true);
    }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}