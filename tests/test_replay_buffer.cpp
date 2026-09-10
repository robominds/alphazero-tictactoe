#include <cassert>
#include <cstdio>
#include "az/replay_buffer.hpp"

using namespace az;

namespace {
TrainingExample makeExample(float value) {
    TrainingExample ex;
    ex.encodedBoard.fill(0.0f);
    ex.targetPolicy.fill(0.0f);
    ex.targetValue = value;
    return ex;
}
}

void test_add_and_size() {
    ReplayBuffer buf(3);
    assert(buf.size() == 0);
    buf.add(makeExample(1.0f));
    assert(buf.size() == 1);
}

void test_capacity_overwrites_oldest() {
    ReplayBuffer buf(2);
    buf.add(makeExample(1.0f));
    buf.add(makeExample(2.0f));
    buf.add(makeExample(3.0f)); // overwrites the value=1.0 example
    assert(buf.size() == 2);
    auto batch = buf.sampleBatch(50);
    bool sawOne = false;
    for (const auto& ex : batch) if (ex.targetValue == 1.0f) sawOne = true;
    assert(!sawOne);
}

void test_sample_batch_returns_requested_size() {
    ReplayBuffer buf(5);
    buf.add(makeExample(1.0f));
    buf.add(makeExample(2.0f));
    auto batch = buf.sampleBatch(10);
    assert(batch.size() == 10);
}

int main() {
    test_add_and_size();
    test_capacity_overwrites_oldest();
    test_sample_batch_returns_requested_size();
    std::printf("all replay_buffer tests passed\n");
    return 0;
}
