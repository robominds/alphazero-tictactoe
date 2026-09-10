#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "az/network.hpp"

using namespace az;

void test_predict_output_shapes_and_ranges() {
    Network net;
    std::array<float, 18> input{};
    input[0] = 1.0f;
    Prediction pred = net.predict(input);
    float sum = 0.0f;
    for (float p : pred.policy) {
        assert(p >= 0.0f);
        sum += p;
    }
    assert(std::fabs(sum - 1.0f) < 1e-4f);
    assert(pred.value >= -1.0f && pred.value <= 1.0f);
}

void test_train_step_reduces_loss_on_fixed_batch() {
    Network net;
    TrainingExample example;
    example.encodedBoard.fill(0.0f);
    example.encodedBoard[0] = 1.0f;
    example.targetPolicy.fill(0.0f);
    example.targetPolicy[4] = 1.0f;
    example.targetValue = 1.0f;

    std::vector<TrainingExample> batch{example};

    float firstLoss = net.trainStep(batch, 0.05f);
    float lastLoss = firstLoss;
    for (int i = 0; i < 200; ++i) {
        lastLoss = net.trainStep(batch, 0.05f);
    }
    assert(lastLoss < firstLoss);
    assert(lastLoss < 0.1f);
}

int main() {
    test_predict_output_shapes_and_ranges();
    test_train_step_reduces_loss_on_fixed_batch();
    std::printf("all network tests passed\n");
    return 0;
}
