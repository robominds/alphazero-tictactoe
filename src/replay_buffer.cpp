#include "az/replay_buffer.hpp"
#include <cassert>

namespace az {

ReplayBuffer::ReplayBuffer(size_t capacity) : capacity_(capacity) {
    buffer_.reserve(capacity);
}

void ReplayBuffer::add(const TrainingExample& example) {
    if (buffer_.size() < capacity_) {
        buffer_.push_back(example);
    } else {
        buffer_[nextIndex_] = example;
    }
    nextIndex_ = (nextIndex_ + 1) % capacity_;
}

std::vector<TrainingExample> ReplayBuffer::sampleBatch(size_t batchSize) const {
    assert(!buffer_.empty());
    std::uniform_int_distribution<size_t> dist(0, buffer_.size() - 1);
    std::vector<TrainingExample> batch;
    batch.reserve(batchSize);
    for (size_t i = 0; i < batchSize; ++i) {
        batch.push_back(buffer_[dist(rng_)]);
    }
    return batch;
}

} // namespace az
