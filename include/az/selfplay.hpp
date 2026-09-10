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
    // Dirichlet noise mixed into the root's priors during search, so search
    // keeps exploring every legal move even when the network is confident
    // (possibly wrongly) that some move is bad. Only used during self-play
    // -- eval/play_cli search without it, for the network's unperturbed
    // best play.
    float dirichletAlpha = 0.3f;
    float dirichletEpsilon = 0.25f;
};

// Plays one full self-play game guided by `network`+MCTS, pushing every
// position's (encodedBoard, mctsPolicy, outcome) into `buffer`.
void playSelfPlayGame(const Network& network, const SelfPlayConfig& config, ReplayBuffer& buffer);

} // namespace az
