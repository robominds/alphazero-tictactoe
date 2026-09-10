#include "az/network.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace az {

Network::Network() {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (auto& row : w1_) for (float& w : row) w = dist(gen);
    b1_.fill(0.0f);
    for (auto& row : wPolicy_) for (float& w : row) w = dist(gen);
    bPolicy_.fill(0.0f);
    for (float& w : wValue_) w = dist(gen);
    bValue_ = 0.0f;
}

Prediction Network::predict(const std::array<float, kInputSize>& encodedBoard) const {
    std::array<float, kHiddenSize> a1{};
    for (int h = 0; h < kHiddenSize; ++h) {
        float z = b1_[h];
        for (int i = 0; i < kInputSize; ++i) z += w1_[h][i] * encodedBoard[i];
        a1[h] = z > 0.0f ? z : 0.0f;
    }

    std::array<float, kPolicySize> logits{};
    for (int k = 0; k < kPolicySize; ++k) {
        float z = bPolicy_[k];
        for (int h = 0; h < kHiddenSize; ++h) z += wPolicy_[k][h] * a1[h];
        logits[k] = z;
    }
    float maxLogit = logits[0];
    for (float l : logits) maxLogit = std::max(maxLogit, l);
    float sumExp = 0.0f;
    std::array<float, kPolicySize> policy{};
    for (int k = 0; k < kPolicySize; ++k) {
        policy[k] = std::exp(logits[k] - maxLogit);
        sumExp += policy[k];
    }
    for (float& p : policy) p /= sumExp;

    float valuePre = bValue_;
    for (int h = 0; h < kHiddenSize; ++h) valuePre += wValue_[h] * a1[h];
    float value = std::tanh(valuePre);

    return Prediction{policy, value};
}

float Network::trainStep(const std::vector<TrainingExample>& batch, float learningRate) {
    std::array<std::array<float, kInputSize>, kHiddenSize> gradW1{};
    std::array<float, kHiddenSize> gradB1{};
    std::array<std::array<float, kHiddenSize>, kPolicySize> gradWPolicy{};
    std::array<float, kPolicySize> gradBPolicy{};
    std::array<float, kHiddenSize> gradWValue{};
    float gradBValue = 0.0f;

    float totalLoss = 0.0f;

    for (const auto& example : batch) {
        std::array<float, kHiddenSize> z1{}, a1{};
        for (int h = 0; h < kHiddenSize; ++h) {
            float z = b1_[h];
            for (int i = 0; i < kInputSize; ++i) z += w1_[h][i] * example.encodedBoard[i];
            z1[h] = z;
            a1[h] = z > 0.0f ? z : 0.0f;
        }

        std::array<float, kPolicySize> logits{};
        for (int k = 0; k < kPolicySize; ++k) {
            float z = bPolicy_[k];
            for (int h = 0; h < kHiddenSize; ++h) z += wPolicy_[k][h] * a1[h];
            logits[k] = z;
        }
        float maxLogit = logits[0];
        for (float l : logits) maxLogit = std::max(maxLogit, l);
        float sumExp = 0.0f;
        std::array<float, kPolicySize> policy{};
        for (int k = 0; k < kPolicySize; ++k) {
            policy[k] = std::exp(logits[k] - maxLogit);
            sumExp += policy[k];
        }
        for (float& p : policy) p /= sumExp;

        float valuePre = bValue_;
        for (int h = 0; h < kHiddenSize; ++h) valuePre += wValue_[h] * a1[h];
        float value = std::tanh(valuePre);

        float valueLoss = (value - example.targetValue) * (value - example.targetValue);
        float policyLoss = 0.0f;
        for (int k = 0; k < kPolicySize; ++k) {
            policyLoss -= example.targetPolicy[k] * std::log(policy[k] + 1e-8f);
        }
        totalLoss += valueLoss + policyLoss;

        float dValuePre = 2.0f * (value - example.targetValue) * (1.0f - value * value);

        std::array<float, kPolicySize> dLogits{};
        for (int k = 0; k < kPolicySize; ++k) dLogits[k] = policy[k] - example.targetPolicy[k];

        std::array<float, kHiddenSize> dA1{};
        for (int h = 0; h < kHiddenSize; ++h) {
            float grad = dValuePre * wValue_[h];
            for (int k = 0; k < kPolicySize; ++k) grad += dLogits[k] * wPolicy_[k][h];
            dA1[h] = grad;
        }
        std::array<float, kHiddenSize> dZ1{};
        for (int h = 0; h < kHiddenSize; ++h) dZ1[h] = z1[h] > 0.0f ? dA1[h] : 0.0f;

        for (int h = 0; h < kHiddenSize; ++h) {
            for (int i = 0; i < kInputSize; ++i) gradW1[h][i] += dZ1[h] * example.encodedBoard[i];
            gradB1[h] += dZ1[h];
        }
        for (int k = 0; k < kPolicySize; ++k) {
            for (int h = 0; h < kHiddenSize; ++h) gradWPolicy[k][h] += dLogits[k] * a1[h];
            gradBPolicy[k] += dLogits[k];
        }
        for (int h = 0; h < kHiddenSize; ++h) gradWValue[h] += dValuePre * a1[h];
        gradBValue += dValuePre;
    }

    float n = static_cast<float>(batch.size());
    for (int h = 0; h < kHiddenSize; ++h) {
        for (int i = 0; i < kInputSize; ++i) w1_[h][i] -= learningRate * gradW1[h][i] / n;
        b1_[h] -= learningRate * gradB1[h] / n;
    }
    for (int k = 0; k < kPolicySize; ++k) {
        for (int h = 0; h < kHiddenSize; ++h) wPolicy_[k][h] -= learningRate * gradWPolicy[k][h] / n;
        bPolicy_[k] -= learningRate * gradBPolicy[k] / n;
    }
    for (int h = 0; h < kHiddenSize; ++h) wValue_[h] -= learningRate * gradWValue[h] / n;
    bValue_ -= learningRate * gradBValue / n;

    return totalLoss / n;
}

void Network::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Network::save: cannot open " + path);
    for (const auto& row : w1_) out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(b1_.data()), b1_.size() * sizeof(float));
    for (const auto& row : wPolicy_) out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(bPolicy_.data()), bPolicy_.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(wValue_.data()), wValue_.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(&bValue_), sizeof(float));
}

void Network::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Network::load: cannot open " + path);
    for (auto& row : w1_) in.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(b1_.data()), b1_.size() * sizeof(float));
    for (auto& row : wPolicy_) in.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(bPolicy_.data()), bPolicy_.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(wValue_.data()), wValue_.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(&bValue_), sizeof(float));
    if (!in) throw std::runtime_error("Network::load: truncated file " + path);
}

} // namespace az
