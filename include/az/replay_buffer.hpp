#pragma once
#include <random>
#include <vector>
#include "az/network.hpp"

namespace az {

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity);

    void add(const TrainingExample& example);

    // Uniform random sampling with replacement. Precondition: size() > 0.
    std::vector<TrainingExample> sampleBatch(size_t batchSize) const;

    size_t size() const { return buffer_.size(); }

private:
    size_t capacity_;
    size_t nextIndex_ = 0;
    std::vector<TrainingExample> buffer_;
    mutable std::mt19937 rng_{std::random_device{}()};
};

} // namespace az
