#pragma once
#include "az/network.hpp"
#include "az/replay_buffer.hpp"

namespace az {

struct SelfPlayConfig {
    int numSimulations = 50;
    float cPuct = 1.5f;
    // Number of plies (from game start) using temperature=1.0 sampling;
    // afterward, moves are selected greedily (temperature=0).
    int temperatureMoves = 2;
};

// Plays one full self-play game guided by `network`+MCTS, pushing every
// position's (encodedBoard, mctsPolicy, outcome) into `buffer`.
void playSelfPlayGame(const Network& network, const SelfPlayConfig& config, ReplayBuffer& buffer);

} // namespace az
