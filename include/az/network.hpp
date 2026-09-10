#pragma once
#include <array>
#include <string>
#include <vector>

namespace az {

struct Prediction {
    std::array<float, 9> policy;
    float value;
};

struct TrainingExample {
    std::array<float, 18> encodedBoard;
    std::array<float, 9> targetPolicy;
    float targetValue;
};

class Network {
public:
    static constexpr int kInputSize = 18;
    static constexpr int kHiddenSize = 64;
    static constexpr int kPolicySize = 9;

    Network();

    // Full softmax over all 9 outputs -- NOT masked to legal moves. Callers
    // with board context (MCTS) mask and renormalize themselves.
    Prediction predict(const std::array<float, kInputSize>& encodedBoard) const;

    // One gradient-descent step over a batch. Returns the batch's mean
    // combined loss (value MSE + policy cross-entropy).
    float trainStep(const std::vector<TrainingExample>& batch, float learningRate);

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::array<std::array<float, kInputSize>, kHiddenSize> w1_;
    std::array<float, kHiddenSize> b1_;
    std::array<std::array<float, kHiddenSize>, kPolicySize> wPolicy_;
    std::array<float, kPolicySize> bPolicy_;
    std::array<float, kHiddenSize> wValue_;
    float bValue_;
};

} // namespace az
