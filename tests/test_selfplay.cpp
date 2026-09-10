#include <cassert>
#include <cmath>
#include <cstdio>
#include "az/network.hpp"
#include "az/replay_buffer.hpp"
#include "az/selfplay.hpp"

using namespace az;

void test_selfplay_game_populates_buffer() {
    Network net;
    ReplayBuffer buffer(100);
    SelfPlayConfig config;
    config.numSimulations = 20;

    playSelfPlayGame(net, config, buffer);

    assert(buffer.size() >= 5 && buffer.size() <= 9);

    auto batch = buffer.sampleBatch(buffer.size());
    for (const auto& ex : batch) {
        float sum = 0.0f;
        for (float p : ex.targetPolicy) {
            assert(p >= 0.0f);
            sum += p;
        }
        assert(std::fabs(sum - 1.0f) < 1e-3f);
        assert(ex.targetValue == 1.0f || ex.targetValue == -1.0f || ex.targetValue == 0.0f);
    }
}

int main() {
    test_selfplay_game_populates_buffer();
    std::printf("all selfplay tests passed\n");
    return 0;
}
